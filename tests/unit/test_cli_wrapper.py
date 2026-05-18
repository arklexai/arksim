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

# ── run_cli ──────────────────────────────


def _stub_run(returncode: int = 0, stdout: str = "", stderr: str = "") -> object:
    """Build a subprocess.run mock that writes to the file objects passed.

    The new run_cli implementation passes ``stdout=tempfile.TemporaryFile()``
    and ``stderr=tempfile.TemporaryFile()`` instead of using
    ``capture_output=True``, so tests must write payload to those file
    handles rather than stash it on a CompletedProcess.
    """
    import subprocess as _sub
    from unittest.mock import MagicMock

    def _runner(argv: list[str], **kw: object) -> _sub.CompletedProcess:
        out_file = kw.get("stdout")
        err_file = kw.get("stderr")
        if hasattr(out_file, "write"):
            out_file.write(stdout)
        if hasattr(err_file, "write"):
            err_file.write(stderr)
        return _sub.CompletedProcess(
            args=argv,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    return MagicMock(side_effect=_runner)


class TestRunCliSuccess:
    """run_cli returns a success result when subprocess exits 0."""

    def test_returns_success_on_zero_exit(self) -> None:
        runner = _stub_run(returncode=0, stdout="all good\n", stderr="")
        with patch("subprocess.run", runner) as mock_run:
            result = run_cli(["evaluate", "config.yaml"])

        assert result["status"] == "success"
        assert result["stdout"] == "all good\n"
        assert result["stderr"] == ""
        assert result["return_code"] == 0
        assert mock_run.call_args.args[0] == [
            "arksim",
            "evaluate",
            "config.yaml",
        ]

    def test_passes_cwd_and_timeout(self, tmp_path: Path) -> None:
        runner = _stub_run(returncode=0, stdout="1.0.0\n", stderr="")
        with patch("subprocess.run", runner) as mock_run:
            run_cli(["version"], cwd=str(tmp_path), timeout=30)

        kwargs = mock_run.call_args.kwargs
        assert kwargs["cwd"] == str(tmp_path.resolve())
        assert kwargs["timeout"] == 30


class TestRunCliNonzeroExit:
    """run_cli returns an error result on nonzero exit codes."""

    def test_returns_error_with_stderr_message(self) -> None:
        runner = _stub_run(
            returncode=1,
            stdout="",
            stderr="Config error: missing field\n",
        )
        with patch("subprocess.run", runner):
            result = run_cli(["evaluate", "bad.yaml"])

        assert result["status"] == "error"
        assert "Config error: missing field" in result["error_message"]
        assert "Config error: missing field" in result["stderr"]
        assert result["return_code"] == 1


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
        assert result["stdout"] == ""
        assert result["stderr"] == ""

    def test_captures_partial_output_on_timeout(self) -> None:
        exc = subprocess.TimeoutExpired(cmd=["arksim", "evaluate"], timeout=60)
        exc.stdout = "partial stdout"
        exc.stderr = "partial stderr"

        with patch("subprocess.run", side_effect=exc):
            result = run_cli(["evaluate"], timeout=60)

        assert result["status"] == "error"
        assert result["stdout"] == "partial stdout"
        assert result["stderr"] == "partial stderr"


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


class TestParseJsonFileOversized:
    """parse_json_file rejects files exceeding the size limit."""

    def test_returns_error_for_oversized_file(self, tmp_path: Path) -> None:
        """A file larger than _MAX_JSON_SIZE is rejected before parsing."""
        from integrations.claude_code.mcp_server.cli_wrapper import _MAX_JSON_SIZE

        big_file = tmp_path / "huge.json"
        big_file.write_text("{}")

        # Mock stat to report a size over the limit without allocating real data.
        fake_size = _MAX_JSON_SIZE + 1
        original_stat = Path.stat

        def oversized_stat(self: Path) -> object:
            result = original_stat(self)
            if self == big_file:
                # Return a modified stat result with an inflated size.
                return type(result)(
                    (
                        result.st_mode,
                        result.st_ino,
                        result.st_dev,
                        result.st_nlink,
                        result.st_uid,
                        result.st_gid,
                        fake_size,
                        result.st_atime,
                        result.st_mtime,
                        result.st_ctime,
                    )
                )
            return result

        with patch.object(Path, "stat", oversized_stat):
            result = parse_json_file(str(big_file))

        assert result["status"] == "error"
        assert "File too large" in result["error_message"]
        assert str(big_file) in result["error_message"]
