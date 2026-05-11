# SPDX-License-Identifier: Apache-2.0
"""Thin wrapper around the ``arksim`` CLI for use by the MCP server.

Every public function returns a plain ``dict`` so the MCP tool layer can
serialize responses without importing domain types.

Subprocess output capture goes through ``tempfile.TemporaryFile`` rather
than ``capture_output=True`` so a large stream cannot deadlock the OS
pipe buffer (the typical arksim simulation log routinely exceeds the
64 KB pipe limit). Stderr is run through :func:`redact_secrets` before
it is returned to the LLM client to keep API keys out of conversation
transcripts.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .security import (
    MAX_CAPTURED_BYTES,
    PathValidationError,
    redact_secrets,
    tail_capture,
    validate_path_arg,
)

_MAX_JSON_SIZE = 50 * 1024 * 1024  # 50 MB


def run_cli(
    args: list[str],
    cwd: str | None = None,
    timeout: int = 600,
) -> dict[str, Any]:
    """Execute an ``arksim`` CLI command and return a structured result.

    Parameters
    ----------
    args:
        Arguments forwarded to the ``arksim`` binary
        (e.g. ``["evaluate", "config.yaml"]``).
    cwd:
        Working directory for the subprocess.  ``None`` inherits the
        current process working directory.  When provided, the path
        must exist, must be a directory, and must not be ``/``,
        ``$HOME``, or any path containing a NUL byte.
    timeout:
        Maximum wall-clock seconds before the process is killed.

    Returns
    -------
    dict
        Always contains ``status`` (``"success"`` | ``"error"``),
        ``stdout``, ``stderr``, and ``return_code``.  On error an
        ``error_message`` key is also present.  Both ``stdout`` and
        ``stderr`` are tail-truncated to roughly 256 KB and ``stderr``
        has recognised secret patterns redacted.
    """
    cwd_arg: str | None = None
    if cwd is not None:
        try:
            resolved = validate_path_arg(
                cwd,
                allow_none=False,
                require_exists=True,
                require_dir=True,
            )
        except PathValidationError as exc:
            return {
                "status": "error",
                "error_message": f"invalid cwd: {exc}",
                "stdout": "",
                "stderr": "",
                "return_code": -1,
            }
        cwd_arg = str(resolved)

    with (
        tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as out,
        tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as err,
    ):
        try:
            completed = subprocess.run(
                ["arksim", *args],
                stdout=out,
                stderr=err,
                cwd=cwd_arg,
                timeout=timeout,
                text=True,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            out.seek(0)
            err.seek(0)
            stdout_partial = tail_capture(out.read(), MAX_CAPTURED_BYTES)
            stderr_partial = redact_secrets(
                tail_capture(err.read(), MAX_CAPTURED_BYTES)
            )
            return {
                "status": "error",
                "error_message": f"Command timed out after {timeout} seconds",
                "stdout": stdout_partial or (exc.stdout or ""),
                "stderr": stderr_partial or redact_secrets(exc.stderr or ""),
                "return_code": -1,
            }
        except FileNotFoundError:
            return {
                "status": "error",
                "error_message": (
                    "arksim CLI not found. "
                    "Install arksim with: pip install arksim[claude]"
                ),
                "stdout": "",
                "stderr": "",
                "return_code": -1,
            }

        out.seek(0)
        err.seek(0)
        stdout = tail_capture(out.read(), MAX_CAPTURED_BYTES)
        stderr = redact_secrets(tail_capture(err.read(), MAX_CAPTURED_BYTES))

    if completed.returncode != 0:
        error_message = (
            stderr
            if stderr
            else f"Command failed with exit code {completed.returncode}"
        )
        return {
            "status": "error",
            "error_message": error_message,
            "stdout": stdout,
            "stderr": stderr,
            "return_code": completed.returncode,
        }

    return {
        "status": "success",
        "stdout": stdout,
        "stderr": stderr,
        "return_code": 0,
    }


def parse_json_file(path: str) -> dict[str, Any]:
    """Read and parse a JSON file, returning a structured result.

    Parameters
    ----------
    path:
        Filesystem path to the JSON file.

    Returns
    -------
    dict
        On success: ``{"status": "success", "data": <parsed>}``.
        On failure: ``{"status": "error", "error_message": ...}``.
    """
    file_path = Path(path)
    try:
        file_size = file_path.stat().st_size
    except FileNotFoundError:
        return {
            "status": "error",
            "error_message": f"File not found: {path}",
        }
    except PermissionError:
        return {
            "status": "error",
            "error_message": f"Permission denied: {path}",
        }
    if file_size > _MAX_JSON_SIZE:
        return {
            "status": "error",
            "error_message": (
                f"File too large: {path} ({file_size} bytes, max {_MAX_JSON_SIZE})"
            ),
        }
    try:
        # O_NOFOLLOW prevents a malicious symlink at the leaf
        # from leaking file contents outside the project root.
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags)
        with os.fdopen(fd, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {
            "status": "error",
            "error_message": f"File not found: {path}",
        }
    except OSError as exc:
        return {
            "status": "error",
            "error_message": f"Could not open {path}: {exc}",
        }
    except json.JSONDecodeError:
        return {
            "status": "error",
            "error_message": f"Invalid JSON in {path}",
        }
    except UnicodeDecodeError:
        return {
            "status": "error",
            "error_message": f"File is not valid UTF-8: {path}",
        }

    return {"status": "success", "data": data}
