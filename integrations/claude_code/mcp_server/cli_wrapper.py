# SPDX-License-Identifier: Apache-2.0
"""Thin wrapper around the ``arksim`` CLI for use by the MCP server.

Every public function returns a plain ``dict`` so the MCP tool layer can
serialize responses without importing domain types.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any


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
        current process working directory.
    timeout:
        Maximum wall-clock seconds before the process is killed.

    Returns
    -------
    dict
        Always contains ``status`` (``"success"`` | ``"error"``),
        ``stdout``, ``stderr``, and ``return_code``.  On error an
        ``error_message`` key is also present.
    """
    try:
        completed = subprocess.run(
            ["arksim", *args],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "error_message": (f"Command timed out after {timeout} seconds"),
            "stdout": "",
            "stderr": "",
            "return_code": -1,
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "error_message": (
                "arksim CLI not found. Install arksim with: pip install arksim"
            ),
            "stdout": "",
            "stderr": "",
            "return_code": -1,
        }

    if completed.returncode != 0:
        error_message = (
            completed.stderr
            if completed.stderr
            else f"Command failed with exit code {completed.returncode}"
        )
        return {
            "status": "error",
            "error_message": error_message,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "return_code": completed.returncode,
        }

    return {
        "status": "success",
        "stdout": completed.stdout,
        "stderr": completed.stderr,
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
    try:
        with open(path) as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {
            "status": "error",
            "error_message": f"File not found: {path}",
        }
    except json.JSONDecodeError:
        return {
            "status": "error",
            "error_message": f"Invalid JSON in {path}",
        }

    return {"status": "success", "data": data}
