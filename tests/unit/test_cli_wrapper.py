# SPDX-License-Identifier: Apache-2.0
"""Tests for the Claude Code MCP server CLI wrapper."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from integrations.claude_code.mcp_server.cli_wrapper import (
    parse_json_file,
    run_cli,
)

# ── run_cli ──────────────────────────────────────────────────


class TestRunCliSuccess:
    """run_cli returns a success result when subprocess exits 0."""

    def test_returns_success_on_zero_exit(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["arksim", "evaluate", "config.yaml"],
            returncode=0,
            stdout="all good\n",
            stderr="",
        )
        with patch("subprocess.run", return_value=completed) as mock_run:
            result = run_cli(["evaluate", "config.yaml"])

        assert result["status"] == "success"
        assert result["stdout"] == "all good\n"
        assert result["stderr"] == ""
        assert result["return_code"] == 0

        mock_run.assert_called_once_with(
            ["arksim", "evaluate", "config.yaml"],
            capture_output=True,
            text=True,
            cwd=None,
            timeout=600,
        )

    def test_passes_cwd_and_timeout(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["arksim", "version"],
            returncode=0,
            stdout="1.0.0\n",
            stderr="",
        )
        with patch("subprocess.run", return_value=completed) as mock_run:
            run_cli(["version"], cwd="/tmp/work", timeout=30)

        mock_run.assert_called_once_with(
            ["arksim", "version"],
            capture_output=True,
            text=True,
            cwd="/tmp/work",
            timeout=30,
        )


class TestRunCliNonzeroExit:
    """run_cli returns an error result on nonzero exit codes."""

    def test_returns_error_with_stderr_message(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["arksim", "evaluate", "bad.yaml"],
            returncode=1,
            stdout="",
            stderr="Config error: missing field\n",
        )
        with patch("subprocess.run", return_value=completed):
            result = run_cli(["evaluate", "bad.yaml"])

        assert result["status"] == "error"
        assert result["error_message"] == "Config error: missing field\n"
        assert result["stdout"] == ""
        assert result["stderr"] == "Config error: missing field\n"
        assert result["return_code"] == 1

    def test_returns_generic_message_when_stderr_empty(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["arksim", "evaluate"],
            returncode=2,
            stdout="partial output",
            stderr="",
        )
        with patch("subprocess.run", return_value=completed):
            result = run_cli(["evaluate"])

        assert result["status"] == "error"
        assert "exit code 2" in result["error_message"]
        assert result["return_code"] == 2


class TestRunCliTimeout:
    """run_cli handles subprocess timeout."""

    def test_returns_error_on_timeout(self) -> None:
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(
                cmd=["arksim", "evaluate"], timeout=600
            ),
        ):
            result = run_cli(["evaluate"])

        assert result["status"] == "error"
        assert "timed out" in result["error_message"]
        assert "600" in result["error_message"]
        assert result["return_code"] == -1


class TestRunCliFileNotFound:
    """run_cli handles missing arksim binary."""

    def test_returns_error_when_arksim_not_found(self) -> None:
        with patch(
            "subprocess.run",
            side_effect=FileNotFoundError(
                "[Errno 2] No such file or directory: 'arksim'"
            ),
        ):
            result = run_cli(["evaluate"])

        assert result["status"] == "error"
        assert "arksim CLI not found" in result["error_message"]
        assert result["return_code"] == -1


# ── parse_json_file ──────────────────────────────────────────


class TestParseJsonFileSuccess:
    """parse_json_file reads and parses valid JSON."""

    def test_returns_parsed_data(self, tmp_path: Path) -> None:
        payload = {"scores": [0.9, 0.85], "summary": "pass"}
        json_file = tmp_path / "results.json"
        json_file.write_text(json.dumps(payload))

        result = parse_json_file(str(json_file))

        assert result["status"] == "success"
        assert result["data"] == payload


class TestParseJsonFileMissing:
    """parse_json_file handles missing files."""

    def test_returns_error_for_missing_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.json"

        result = parse_json_file(str(missing))

        assert result["status"] == "error"
        assert "File not found" in result["error_message"]
        assert str(missing) in result["error_message"]


class TestParseJsonFileInvalid:
    """parse_json_file handles malformed JSON."""

    def test_returns_error_for_invalid_json(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "broken.json"
        bad_file.write_text("{not valid json")

        result = parse_json_file(str(bad_file))

        assert result["status"] == "error"
        assert "Invalid JSON" in result["error_message"]
        assert str(bad_file) in result["error_message"]
