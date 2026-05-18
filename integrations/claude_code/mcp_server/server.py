# SPDX-License-Identifier: Apache-2.0
"""FastMCP stdio server exposing arksim CLI tools to Claude Code.

Each tool has an internal ``_function`` (testable without FastMCP) and a
thin ``@mcp.tool()`` wrapper that delegates to it.

The server treats every path-shaped argument that arrives through a
tool call as untrusted: it must resolve under the server's working
directory (the project root that Claude Code launched it from), must
not equal ``$HOME`` or ``/``, and must not contain a NUL byte. The UI
subprocess is reaped via ``atexit`` and SIGTERM/SIGINT handlers so a
client disconnect does not orphan it.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from .cli_wrapper import parse_json_file, run_cli
from .security import (
    PathValidationError,
    is_inside,
    redact_secrets,
    validate_path_arg,
)

logger = logging.getLogger(__name__)

try:
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("arksim")
except ImportError:
    # FastMCP is optional (pip install arksim[claude]).
    # Internal _functions work without it; only the @mcp.tool()
    # decorators and main() require it.
    from types import SimpleNamespace

    mcp = SimpleNamespace(  # type: ignore[assignment]
        tool=lambda: lambda fn: fn,
    )

# Module-level state for the UI subprocess.
_ui_process: subprocess.Popen[str] | None = None
_ui_port: int | None = None

# Allowed pattern for CLI override keys (lowercase identifier style).
_OVERRIDE_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

# Seconds to wait before checking if the UI process exited immediately.
_UI_STARTUP_PROBE_DELAY = 0.2

# Path-shaped CLI override keys. Values for these keys are validated
# against the project root; anything else is passed through to the CLI
# unchanged (after escaping into ``--flag=value`` list form).
_PATH_OVERRIDE_KEYS = frozenset(
    {
        "scenario_file_path",
        "simulation_file_path",
        "output_file_path",
        "output_dir",
        "module_path",
        "agent_config_file_path",
        "custom_metrics_file_paths",
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_PROJECT_ROOT: Path | None = None


# Marker files that identify a directory as a real project root.
_ROOT_MARKERS = (
    ".git",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "package.json",
    "Cargo.toml",
    "go.mod",
    ".mcp.json",
)


def _resolve_project_root(start: Path | None = None) -> Path:
    """Walk parents from ``start`` to find the first project marker.

    Refuses to return the filesystem root or the user home directory
    even if a marker is found there. Falls back to the immediate
    starting directory when no marker is found within the walk.
    """
    here = (start or Path.cwd()).resolve()
    home = Path.home().resolve()

    for candidate in (here, *here.parents):
        if candidate == Path(candidate.anchor or "/"):
            break
        if candidate == home:
            break
        if any((candidate / m).exists() for m in _ROOT_MARKERS):
            return candidate

    if here == Path(here.anchor or "/"):
        raise SystemExit(
            "arksim-mcp refuses to run from the filesystem root. "
            "Launch from inside a project directory."
        )
    if here == home:
        raise SystemExit(
            "arksim-mcp refuses to run from the user home directory. "
            "Launch from inside a project directory."
        )
    return here


def _project_root() -> Path:
    """Return the cached project root resolved at startup."""
    global _PROJECT_ROOT  # noqa: PLW0603
    if _PROJECT_ROOT is None:
        _PROJECT_ROOT = _resolve_project_root()
    return _PROJECT_ROOT


def _bound_to_project(
    candidate: str | None,
    *,
    allow_none: bool = True,
    require_exists: bool = True,
    require_dir: bool = False,
    require_file: bool = False,
) -> Path | None:
    """Validate ``candidate`` and require it to live inside the project root.

    Raises :class:`PathValidationError` when the candidate fails
    validation or escapes the project. Returns ``None`` only when
    ``candidate`` is ``None`` and ``allow_none`` is true.
    """
    resolved = validate_path_arg(
        candidate,
        allow_none=allow_none,
        require_exists=require_exists,
        require_dir=require_dir,
        require_file=require_file,
    )
    if resolved is None:
        return None
    root = _project_root()
    if not is_inside(resolved, root):
        raise PathValidationError(f"path {resolved} escapes the project root {root}")
    return resolved


def _build_override_args(
    overrides: dict[str, str] | None,
) -> tuple[list[str], list[str]]:
    """Convert a dict of CLI overrides to a flat list of flag pairs.

    Keys use underscores (Python style) and are converted to hyphenated
    CLI flags. For example ``{"num_workers": "5"}`` becomes
    ``["--num-workers=5"]``. Path-shaped values are validated against
    the project root before they are passed through.

    Returns:
        A tuple of (args, skipped_keys). ``args`` contains the valid
        CLI flags; ``skipped_keys`` lists any keys that did not match
        the expected identifier pattern or whose path values fail
        validation.
    """
    if not overrides:
        return [], []
    args: list[str] = []
    skipped: list[str] = []
    for key, value in overrides.items():
        if not _OVERRIDE_KEY_PATTERN.match(key):
            logger.warning("Skipping invalid override key: %r", key)
            skipped.append(key)
            continue
        value_str = str(value)
        if "\x00" in value_str or "\n" in value_str:
            logger.warning(
                "Skipping override %s with newline or NUL byte in value", key
            )
            skipped.append(key)
            continue
        if key in _PATH_OVERRIDE_KEYS:
            try:
                resolved = _bound_to_project(
                    value_str,
                    allow_none=False,
                    require_exists=False,
                )
                value_str = str(resolved)
            except PathValidationError as exc:
                logger.warning("Skipping path override %s=%r: %s", key, value, exc)
                skipped.append(key)
                continue
        flag = f"--{key.replace('_', '-')}"
        args.append(f"{flag}={value_str}")
    return args, skipped


def _path_validation_error(exc: PathValidationError) -> dict[str, Any]:
    """Build a structured error response from a path validation failure."""
    return {"status": "error", "error_message": str(exc)}


# ---------------------------------------------------------------------------
# UI subprocess lifecycle
# ---------------------------------------------------------------------------


def _terminate_ui() -> None:
    """Stop the UI subprocess if it is still running.

    Called from atexit and from SIGTERM/SIGINT handlers so the UI is
    reaped when Claude Code disconnects or the MCP server is killed.
    """
    global _ui_process, _ui_port  # noqa: PLW0603
    if _ui_process is None:
        return
    try:
        if _ui_process.poll() is None:
            try:
                # The UI was started with start_new_session=True,
                # so it owns its own process group. Killing the
                # group reaps any children the UI spawned.
                pgid = os.getpgid(_ui_process.pid)
                os.killpg(pgid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                _ui_process.terminate()
            try:
                _ui_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    pgid = os.getpgid(_ui_process.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    _ui_process.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    _ui_process.wait(timeout=2)
    except (OSError, ValueError):
        # Process already gone or stdio in odd state; nothing to do.
        pass
    finally:
        _ui_process = None
        _ui_port = None


_signal_handlers_installed = False


def _install_signal_handlers() -> None:
    """Register atexit and signal handlers so the UI is reaped on shutdown.

    Idempotent; safe to call from ``main()`` regardless of whether the
    server has been started before in the same process.
    """
    global _signal_handlers_installed  # noqa: PLW0603
    if _signal_handlers_installed:
        return
    atexit.register(_terminate_ui)

    def _handler(signum: int, _frame: object) -> None:
        _terminate_ui()
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        # Not on the main thread or unsupported on this platform.
        with contextlib.suppress(OSError, ValueError):
            signal.signal(sig, _handler)
    _signal_handlers_installed = True


# ---------------------------------------------------------------------------
# Tool internals (tested directly, no FastMCP dependency)
# ---------------------------------------------------------------------------


def _simulate_evaluate(
    config_path: str,
    cli_overrides: dict[str, str] | None = None,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Run simulation and evaluation in a single pass."""
    try:
        config_resolved = _bound_to_project(
            config_path,
            allow_none=False,
            require_exists=True,
            require_file=True,
        )
        cwd_resolved = _bound_to_project(
            cwd,
            allow_none=True,
            require_exists=True,
            require_dir=True,
        )
    except PathValidationError as exc:
        return _path_validation_error(exc)

    override_args, skipped_keys = _build_override_args(cli_overrides)
    result = run_cli(
        ["simulate-evaluate", str(config_resolved), *override_args],
        cwd=str(cwd_resolved) if cwd_resolved is not None else None,
    )
    if result["status"] != "success":
        return {
            "status": "error",
            "error_message": result["error_message"],
            "stderr": result.get("stderr", ""),
        }
    response: dict[str, Any] = {
        "status": "success",
        "output": result["stdout"],
        "stderr": result.get("stderr", ""),
        "message": "Simulation and evaluation completed successfully.",
    }
    if skipped_keys:
        response["warnings"] = [
            f"Skipped invalid override keys: {', '.join(skipped_keys)}"
        ]
    return response


def _evaluate(
    config_path: str,
    simulation_file_path: str | None = None,
    cli_overrides: dict[str, str] | None = None,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Run evaluation on an existing simulation output."""
    try:
        config_resolved = _bound_to_project(
            config_path,
            allow_none=False,
            require_exists=True,
            require_file=True,
        )
        sim_resolved = _bound_to_project(
            simulation_file_path,
            allow_none=True,
            require_exists=True,
            require_file=True,
        )
        cwd_resolved = _bound_to_project(
            cwd,
            allow_none=True,
            require_exists=True,
            require_dir=True,
        )
    except PathValidationError as exc:
        return _path_validation_error(exc)

    overrides = dict(cli_overrides or {})
    if sim_resolved is not None:
        overrides["simulation_file_path"] = str(sim_resolved)
    override_args, skipped_keys = _build_override_args(overrides)
    result = run_cli(
        ["evaluate", str(config_resolved), *override_args],
        cwd=str(cwd_resolved) if cwd_resolved is not None else None,
    )
    if result["status"] != "success":
        return {
            "status": "error",
            "error_message": result["error_message"],
            "stderr": result.get("stderr", ""),
        }
    response: dict[str, Any] = {
        "status": "success",
        "output": result["stdout"],
        "stderr": result.get("stderr", ""),
        "message": "Evaluation completed successfully.",
    }
    if skipped_keys:
        response["warnings"] = [
            f"Skipped invalid override keys: {', '.join(skipped_keys)}"
        ]
    return response


def _list_results(output_dir: str = ".") -> dict[str, Any]:
    """Scan a directory tree for evaluation.json files and summarize each."""
    try:
        search_path = _bound_to_project(
            output_dir,
            allow_none=False,
            require_exists=False,
            require_dir=False,
        )
    except PathValidationError as exc:
        return _path_validation_error(exc)

    if search_path is None or not search_path.is_dir():
        return {"status": "success", "runs": [], "skipped": []}

    runs: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for eval_path in sorted(search_path.rglob("evaluation.json")):
        if eval_path.is_symlink():
            skipped.append(
                {
                    "file": str(eval_path),
                    "reason": "skipped: symlinks not followed",
                }
            )
            continue
        parsed = parse_json_file(str(eval_path))
        if parsed["status"] != "success":
            skipped.append(
                {
                    "file": str(eval_path),
                    "reason": parsed.get("error_message", "unknown"),
                }
            )
            continue
        data = parsed["data"]
        conversations = data.get("conversations", [])
        if not isinstance(conversations, list):
            conversations = []
        unique_errors_raw = data.get("unique_errors", [])
        if not isinstance(unique_errors_raw, list):
            unique_errors_raw = []
        passed = sum(1 for c in conversations if c.get("evaluation_status") == "Done")
        partial = sum(
            1 for c in conversations if c.get("evaluation_status") == "Partial Failure"
        )
        runs.append(
            {
                "evaluation_id": data.get("evaluation_id", ""),
                "simulation_id": data.get("simulation_id", ""),
                "generated_at": data.get("generated_at", ""),
                "file_path": str(eval_path),
                "total_conversations": len(conversations),
                "passed": passed,
                "partial": partial,
                "failed": len(conversations) - passed - partial,
                "unique_errors_count": len(unique_errors_raw),
            }
        )
    return {"status": "success", "runs": runs, "skipped": skipped}


def _read_result(result_path: str) -> dict[str, Any]:
    """Read an evaluation.json and return a structured summary."""
    try:
        resolved = _bound_to_project(
            result_path,
            allow_none=False,
            require_exists=True,
            require_file=True,
        )
    except PathValidationError as exc:
        return _path_validation_error(exc)

    parsed = parse_json_file(str(resolved))
    if parsed["status"] != "success":
        return {
            "status": "error",
            "error_message": parsed["error_message"],
        }
    data = parsed["data"]
    conversations = data.get("conversations", [])
    if not isinstance(conversations, list):
        conversations = []
    raw_unique_errors = data.get("unique_errors", [])
    if not isinstance(raw_unique_errors, list):
        raw_unique_errors = []

    # "Done" means arksim's evaluator completed successfully for that
    # conversation (all metrics scored).  Threshold-based pass/fail
    # requires comparing ``overall_agent_score`` against user-defined
    # thresholds, which are in the config, not the evaluation output.
    passed = sum(1 for c in conversations if c.get("evaluation_status") == "Done")
    partial = sum(
        1 for c in conversations if c.get("evaluation_status") == "Partial Failure"
    )
    failed = len(conversations) - passed - partial

    unique_errors = [
        {
            "error_id": e.get("unique_error_id", ""),
            "category": e.get("behavior_failure_category", ""),
            "description": e.get("unique_error_description", ""),
            "severity": e.get("severity", "medium"),
            "occurrence_count": len(e.get("occurrences", [])),
        }
        for e in raw_unique_errors
    ]

    conversation_summaries = [
        {
            "conversation_id": c.get("conversation_id", ""),
            "goal_completion_score": c.get("goal_completion_score", 0.0),
            "overall_agent_score": c.get("overall_agent_score", 0.0),
            "evaluation_status": c.get("evaluation_status", ""),
            "turn_count": len(c.get("turn_scores", [])),
        }
        for c in conversations
    ]

    return {
        "status": "success",
        "evaluation_id": data.get("evaluation_id", ""),
        "generated_at": data.get("generated_at", ""),
        "total_conversations": len(conversations),
        "passed": passed,
        "partial": partial,
        "failed": failed,
        "unique_errors": unique_errors,
        "conversations": conversation_summaries,
    }


_VALID_AGENT_TYPES = frozenset({"custom", "a2a", "chat_completions"})


def _init_project(
    agent_type: str = "custom",
    directory: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Scaffold a new arksim project."""
    if agent_type not in _VALID_AGENT_TYPES:
        return {
            "status": "error",
            "error_message": (
                f"Invalid agent_type: {agent_type!r}. "
                f"Must be one of: {', '.join(sorted(_VALID_AGENT_TYPES))}"
            ),
        }
    try:
        directory_resolved = _bound_to_project(
            directory,
            allow_none=True,
            require_exists=True,
            require_dir=True,
        )
    except PathValidationError as exc:
        return _path_validation_error(exc)

    cmd = ["init", "--agent-type", agent_type]
    if force:
        cmd.append("--force")
    result = run_cli(
        cmd,
        cwd=str(directory_resolved) if directory_resolved is not None else None,
    )
    if result["status"] != "success":
        return {
            "status": "error",
            "error_message": result["error_message"],
            "stderr": result.get("stderr", ""),
        }
    return {
        "status": "success",
        "output": result["stdout"],
        "message": f"Project initialized with agent type '{agent_type}'.",
    }


def _launch_ui(port: int = 8080) -> dict[str, Any]:
    """Start the arksim UI dashboard in a background process."""
    global _ui_process, _ui_port  # noqa: PLW0603

    _install_signal_handlers()

    if not isinstance(port, int) or isinstance(port, bool):
        return {
            "status": "error",
            "error_message": f"port must be an int, got {type(port).__name__}",
        }
    if not (1 <= port <= 65535):
        return {
            "status": "error",
            "error_message": f"Port must be between 1 and 65535, got {port}.",
        }

    if _ui_process is not None and _ui_process.poll() is None:
        if port != _ui_port:
            return {
                "status": "error",
                "error_message": (
                    f"UI is already running on port {_ui_port}. "
                    f"Stop the current UI before starting on port {port}."
                ),
            }
        return {
            "status": "success",
            "url": f"http://localhost:{_ui_port}",
            "message": "UI is already running.",
        }

    # Previous process exited; clear stale port before restarting.
    if _ui_process is not None:
        _ui_port = None

    try:
        _ui_process = subprocess.Popen(
            ["arksim", "ui", "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except FileNotFoundError:
        return {
            "status": "error",
            "error_message": (
                "arksim CLI not found. Install it with: pip install arksim[claude]"
            ),
        }

    time.sleep(_UI_STARTUP_PROBE_DELAY)
    if _ui_process.poll() is not None:
        stderr_output = ""
        if _ui_process.stderr is not None:
            try:
                stderr_output = _ui_process.stderr.read(8192) or ""
            except (OSError, ValueError):
                stderr_output = ""
        detail = redact_secrets(stderr_output.strip()) if stderr_output else ""
        message = "UI process exited immediately."
        if detail:
            message = f"{message} stderr: {detail}"
        else:
            message = f"{message} Check if the port is in use."
        return {
            "status": "error",
            "error_message": message,
        }

    # Drop the stderr pipe so it cannot deadlock the subprocess once
    # the buffer fills up. We have already consumed any startup error.
    if _ui_process.stderr is not None:
        with contextlib.suppress(OSError):
            _ui_process.stderr.close()

    _ui_port = port
    return {
        "status": "success",
        "url": f"http://localhost:{port}",
        "message": f"UI started on port {port}.",
    }


# ---------------------------------------------------------------------------
# MCP tool wrappers
# ---------------------------------------------------------------------------


@mcp.tool()
def simulate_evaluate(
    config_path: str,
    cli_overrides: dict[str, str] | None = None,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Run agent simulation and evaluation in one step.

    Executes ``arksim simulate-evaluate`` against the given config file.
    Use ``cli_overrides`` to pass additional CLI flags, for example
    ``{"model": "gpt-4o", "num_workers": "5"}``. Pass ``cwd`` if the
    config's relative paths assume a specific working directory.

    All path arguments must resolve inside the project root that the
    MCP server was launched from.
    """
    return _simulate_evaluate(config_path, cli_overrides=cli_overrides, cwd=cwd)


@mcp.tool()
def evaluate(
    config_path: str,
    simulation_file_path: str | None = None,
    cli_overrides: dict[str, str] | None = None,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Evaluate a previously completed simulation.

    Runs ``arksim evaluate`` against the config file. Optionally pass
    ``simulation_file_path`` to point at an existing simulation output.
    Pass ``cwd`` if the config's relative paths assume a specific
    working directory.

    All path arguments must resolve inside the project root that the
    MCP server was launched from.
    """
    return _evaluate(
        config_path,
        simulation_file_path=simulation_file_path,
        cli_overrides=cli_overrides,
        cwd=cwd,
    )


@mcp.tool()
def list_results(output_dir: str = ".") -> dict[str, Any]:
    """List all evaluation results under a directory.

    Recursively scans for ``evaluation.json`` files and returns a summary
    of each run including pass/fail counts and unique error counts.
    """
    return _list_results(output_dir=output_dir)


@mcp.tool()
def read_result(result_path: str) -> dict[str, Any]:
    """Read a single evaluation result file.

    Returns a structured summary including per-conversation scores,
    unique errors with categories and severity, and overall pass/fail
    counts. The ``passed`` count reflects conversations where
    ``evaluation_status == "Done"`` (evaluation completed and all
    metrics scored). For threshold-based pass/fail, compare each
    conversation's ``overall_agent_score`` against your configured
    thresholds.
    """
    return _read_result(result_path)


@mcp.tool()
def init_project(
    agent_type: str = "custom",
    directory: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Initialize a new arksim project.

    Scaffolds a project directory with config files, scenarios, and an
    agent stub. Set ``agent_type`` to ``"custom"``, ``"a2a"``, or
    ``"chat_completions"`` depending on the agent architecture. Pass
    ``force=True`` to overwrite existing files.

    ``directory`` must resolve inside the project root that the MCP
    server was launched from.
    """
    return _init_project(agent_type=agent_type, directory=directory, force=force)


@mcp.tool()
def launch_ui(port: int = 8080) -> dict[str, Any]:
    """Start the arksim evaluation dashboard UI.

    Launches a background process running ``arksim ui`` and returns the
    URL. If the UI is already running, returns the existing URL without
    starting a new process. The UI subprocess is reaped automatically
    when the MCP server exits.
    """
    return _launch_ui(port=port)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the MCP server over stdio."""
    if not hasattr(mcp, "run"):
        raise SystemExit("MCP SDK is not installed. Run: pip install arksim[claude]")
    _install_signal_handlers()
    _resolve_project_root()
    try:
        mcp.run(transport="stdio")
    finally:
        _terminate_ui()


if __name__ == "__main__":
    main()
