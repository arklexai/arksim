# SPDX-License-Identifier: Apache-2.0
"""FastMCP stdio server exposing arksim CLI tools to Claude Code.

Each tool has an internal ``_function`` (testable without FastMCP) and a
thin ``@mcp.tool()`` wrapper that delegates to it.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from integrations.claude_code.mcp_server.cli_wrapper import (
    parse_json_file,
    run_cli,
)

logger = logging.getLogger(__name__)

mcp = FastMCP("arksim")

# Module-level state for the UI subprocess.
_ui_process: subprocess.Popen[str] | None = None
_ui_port: int | None = None

# Allowed pattern for CLI override keys (lowercase identifier style).
_OVERRIDE_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_override_args(overrides: dict[str, str] | None) -> list[str]:
    """Convert a dict of CLI overrides to a flat list of flag pairs.

    Keys use underscores (Python style) and are converted to hyphenated
    CLI flags.  For example ``{"num_workers": "5"}`` becomes
    ``["--num-workers", "5"]``.

    Keys that do not match the expected identifier pattern are skipped
    with a warning log.
    """
    if not overrides:
        return []
    args: list[str] = []
    for key, value in overrides.items():
        if not _OVERRIDE_KEY_PATTERN.match(key):
            logger.warning("Skipping invalid override key: %r", key)
            continue
        flag = f"--{key.replace('_', '-')}"
        args.extend([flag, value])
    return args


# ---------------------------------------------------------------------------
# Tool internals (tested directly, no FastMCP dependency)
# ---------------------------------------------------------------------------


def _simulate_evaluate(
    config_path: str,
    cli_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run simulation and evaluation in a single pass."""
    override_args = _build_override_args(cli_overrides)
    result = run_cli(["simulate-evaluate", config_path, *override_args])
    if result["status"] != "success":
        return {
            "status": "error",
            "error_message": result["error_message"],
        }
    return {
        "status": "success",
        "output": result["stdout"],
        "stderr": result.get("stderr", ""),
        "message": "Simulation and evaluation completed successfully.",
    }


def _evaluate(
    config_path: str,
    simulation_file_path: str | None = None,
    cli_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run evaluation on an existing simulation output."""
    overrides = dict(cli_overrides or {})
    if simulation_file_path is not None:
        overrides["simulation_file_path"] = simulation_file_path
    override_args = _build_override_args(overrides)
    result = run_cli(["evaluate", config_path, *override_args])
    if result["status"] != "success":
        return {
            "status": "error",
            "error_message": result["error_message"],
        }
    return {
        "status": "success",
        "output": result["stdout"],
        "stderr": result.get("stderr", ""),
        "message": "Evaluation completed successfully.",
    }


def _list_results(output_dir: str = ".") -> dict[str, Any]:
    """Scan a directory tree for evaluation.json files and summarize each."""
    search_path = Path(output_dir)
    if not search_path.is_dir():
        return {"status": "success", "runs": [], "skipped": []}

    runs: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for eval_path in sorted(search_path.rglob("evaluation.json")):
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
        runs.append(
            {
                "evaluation_id": data.get("evaluation_id", ""),
                "generated_at": data.get("generated_at", ""),
                "file_path": str(eval_path),
                "total_conversations": len(conversations),
                "passed": passed,
                "failed": len(conversations) - passed,
                "unique_errors_count": len(unique_errors_raw),
            }
        )
    return {"status": "success", "runs": runs, "skipped": skipped}


def _read_result(result_path: str) -> dict[str, Any]:
    """Read an evaluation.json and return a structured summary."""
    parsed = parse_json_file(result_path)
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

    passed = sum(1 for c in conversations if c.get("evaluation_status") == "Done")
    failed = len(conversations) - passed

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
        "failed": failed,
        "unique_errors": unique_errors,
        "conversations": conversation_summaries,
    }


def _init_project(
    agent_type: str = "custom",
    directory: str | None = None,
) -> dict[str, Any]:
    """Scaffold a new arksim project."""
    result = run_cli(
        ["init", "--agent-type", agent_type, "--force"],
        cwd=directory,
    )
    if result["status"] != "success":
        return {
            "status": "error",
            "error_message": result["error_message"],
        }
    return {
        "status": "success",
        "output": result["stdout"],
        "message": f"Project initialized with agent type '{agent_type}'.",
    }


def _launch_ui(port: int = 8080) -> dict[str, Any]:
    """Start the arksim UI dashboard in a background process."""
    global _ui_process, _ui_port  # noqa: PLW0603

    if _ui_process is not None and _ui_process.poll() is None:
        return {
            "status": "success",
            "url": f"http://localhost:{_ui_port}",
            "message": "UI is already running.",
        }

    try:
        _ui_process = subprocess.Popen(
            ["arksim", "ui", "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return {
            "status": "error",
            "error_message": (
                "arksim CLI not found. Install it with: pip install arksim"
            ),
        }

    time.sleep(0.2)
    if _ui_process.poll() is not None:
        return {
            "status": "error",
            "error_message": (
                "UI process exited immediately. Check if the port is in use."
            ),
        }

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
) -> dict[str, Any]:
    """Run agent simulation and evaluation in one step.

    Executes ``arksim simulate-evaluate`` against the given config file.
    Use ``cli_overrides`` to pass additional CLI flags, for example
    ``{"model": "gpt-4o", "num_workers": "5"}``.
    """
    return _simulate_evaluate(config_path, cli_overrides=cli_overrides)


@mcp.tool()
def evaluate(
    config_path: str,
    simulation_file_path: str | None = None,
    cli_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate a previously completed simulation.

    Runs ``arksim evaluate`` against the config file.  Optionally pass
    ``simulation_file_path`` to point at an existing simulation output.
    """
    return _evaluate(
        config_path,
        simulation_file_path=simulation_file_path,
        cli_overrides=cli_overrides,
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
    counts.
    """
    return _read_result(result_path)


@mcp.tool()
def init_project(
    agent_type: str = "custom",
    directory: str | None = None,
) -> dict[str, Any]:
    """Initialize a new arksim project.

    Scaffolds a project directory with config files, scenarios, and an
    agent stub.  Set ``agent_type`` to ``"custom"``, ``"a2a"``, or
    ``"chat_completions"`` depending on the agent architecture.
    """
    return _init_project(agent_type=agent_type, directory=directory)


@mcp.tool()
def launch_ui(port: int = 8080) -> dict[str, Any]:
    """Start the arksim evaluation dashboard UI.

    Launches a background process running ``arksim ui`` and returns the
    URL.  If the UI is already running, returns the existing URL without
    starting a new process.
    """
    return _launch_ui(port=port)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
