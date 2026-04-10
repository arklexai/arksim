# SPDX-License-Identifier: Apache-2.0
"""Tests for the Claude Code MCP server tool functions.

Tests target the internal ``_function`` implementations, not the
``@mcp.tool()`` decorated wrappers, so FastMCP is never imported here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_eval_data(
    *,
    evaluation_id: str = "eval-001",
    conversations: list[dict[str, Any]] | None = None,
    unique_errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a minimal evaluation.json payload for testing."""
    if conversations is None:
        conversations = [
            {
                "conversation_id": "conv-1",
                "goal_completion_score": 0.8,
                "goal_completion_reason": "mostly done",
                "turn_success_ratio": 1.0,
                "overall_agent_score": 0.85,
                "evaluation_status": "Done",
                "turn_scores": [
                    {
                        "turn_id": 1,
                        "scores": [],
                        "turn_score": 4.0,
                        "turn_behavior_failure": "none",
                        "turn_behavior_failure_reason": "",
                        "qual_scores": [],
                        "unique_error_ids": [],
                    }
                ],
            },
            {
                "conversation_id": "conv-2",
                "goal_completion_score": 0.6,
                "goal_completion_reason": "partially done",
                "turn_success_ratio": 0.75,
                "overall_agent_score": 0.65,
                "evaluation_status": "Partial Failure",
                "turn_scores": [],
            },
            {
                "conversation_id": "conv-3",
                "goal_completion_score": 0.3,
                "goal_completion_reason": "failed",
                "turn_success_ratio": 0.5,
                "overall_agent_score": 0.4,
                "evaluation_status": "Failed",
                "turn_scores": [],
            },
        ]
    if unique_errors is None:
        unique_errors = [
            {
                "unique_error_id": "err-1",
                "behavior_failure_category": "hallucination",
                "unique_error_description": "Made up facts",
                "severity": "high",
                "occurrences": [
                    {"conversation_id": "conv-2", "turn_id": 1},
                ],
            },
        ]
    return {
        "schema_version": "1.0",
        "generated_at": "2025-01-15T10:00:00Z",
        "evaluator_version": "0.5.0",
        "evaluation_id": evaluation_id,
        "simulation_id": "sim-001",
        "conversations": conversations,
        "unique_errors": unique_errors,
        "error_scenario_mappings": [],
    }


def _cli_success(stdout: str = "") -> dict[str, Any]:
    return {
        "status": "success",
        "stdout": stdout,
        "stderr": "",
        "return_code": 0,
    }


def _cli_error(message: str = "boom") -> dict[str, Any]:
    return {
        "status": "error",
        "error_message": message,
        "stdout": "",
        "stderr": message,
        "return_code": 1,
    }


# ── _build_override_args ──────────────────────────────────────


class TestBuildOverrideArgs:
    """_build_override_args converts a dict to CLI flag pairs."""

    def test_none_returns_empty_list(self) -> None:
        from integrations.claude_code.mcp_server.server import (
            _build_override_args,
        )

        args, skipped = _build_override_args(None)
        assert args == []
        assert skipped == []

    def test_empty_dict_returns_empty_list(self) -> None:
        from integrations.claude_code.mcp_server.server import (
            _build_override_args,
        )

        args, skipped = _build_override_args({})
        assert args == []
        assert skipped == []

    def test_converts_underscores_to_hyphens(self) -> None:
        from integrations.claude_code.mcp_server.server import (
            _build_override_args,
        )

        args, skipped = _build_override_args({"num_workers": "5"})
        assert args == ["--num-workers=5"]
        assert skipped == []

    def test_multiple_overrides(self) -> None:
        from integrations.claude_code.mcp_server.server import (
            _build_override_args,
        )

        args, skipped = _build_override_args({"model": "gpt-4o", "num_workers": "5"})
        assert "--model=gpt-4o" in args
        assert "--num-workers=5" in args
        assert len(args) == 2
        assert skipped == []

    def test_skipped_keys_returned(self) -> None:
        from integrations.claude_code.mcp_server.server import (
            _build_override_args,
        )

        args, skipped = _build_override_args(
            {"model": "gpt-4o", "BAD-KEY!": "x", "0starts_digit": "y"}
        )
        assert args == ["--model=gpt-4o"]
        assert "BAD-KEY!" in skipped
        assert "0starts_digit" in skipped

    def test_value_starting_with_dashes_uses_bound_form(self) -> None:
        """Values starting with '--' must not be split into separate args."""
        from integrations.claude_code.mcp_server.server import (
            _build_override_args,
        )

        args, skipped = _build_override_args({"model": "--inject"})
        assert args == ["--model=--inject"]
        assert skipped == []


# ── simulate_evaluate ─────────────────────────────────────────


_MOD = "integrations.claude_code.mcp_server.server"


class TestSimulateEvaluateSuccess:
    """_simulate_evaluate delegates to run_cli on success."""

    def test_returns_structured_success(self) -> None:
        from integrations.claude_code.mcp_server.server import (
            _simulate_evaluate,
        )

        with patch(f"{_MOD}.run_cli", return_value=_cli_success("done")):
            result = _simulate_evaluate("config.yaml")

        assert result["status"] == "success"
        assert result["output"] == "done"
        assert "message" in result

    def test_passes_cli_overrides(self) -> None:
        from integrations.claude_code.mcp_server.server import (
            _simulate_evaluate,
        )

        with patch(f"{_MOD}.run_cli", return_value=_cli_success()) as mock:
            _simulate_evaluate(
                "config.yaml",
                cli_overrides={"model": "gpt-4o"},
            )

        args_list = mock.call_args[0][0]
        assert "simulate-evaluate" in args_list
        assert "config.yaml" in args_list
        assert "--model=gpt-4o" in args_list


class TestSimulateEvaluateFailure:
    """_simulate_evaluate returns error dict on CLI failure."""

    def test_returns_error_on_cli_failure(self) -> None:
        from integrations.claude_code.mcp_server.server import (
            _simulate_evaluate,
        )

        with patch(f"{_MOD}.run_cli", return_value=_cli_error("config invalid")):
            result = _simulate_evaluate("bad.yaml")

        assert result["status"] == "error"
        assert result["error_message"] == "config invalid"


# ── evaluate ──────────────────────────────────────────────────


class TestEvaluateSuccess:
    """_evaluate delegates to run_cli with correct args."""

    def test_returns_structured_success(self) -> None:
        from integrations.claude_code.mcp_server.server import _evaluate

        with patch(f"{_MOD}.run_cli", return_value=_cli_success("ok")):
            result = _evaluate("config.yaml")

        assert result["status"] == "success"
        assert result["output"] == "ok"

    def test_adds_simulation_file_path(self) -> None:
        from integrations.claude_code.mcp_server.server import _evaluate

        with patch(f"{_MOD}.run_cli", return_value=_cli_success()) as mock:
            _evaluate(
                "config.yaml",
                simulation_file_path="/tmp/sim.json",
            )

        args_list = mock.call_args[0][0]
        assert "--simulation-file-path=/tmp/sim.json" in args_list


class TestEvaluateFailure:
    """_evaluate returns error dict on CLI failure."""

    def test_returns_error_on_cli_failure(self) -> None:
        from integrations.claude_code.mcp_server.server import _evaluate

        with patch(f"{_MOD}.run_cli", return_value=_cli_error("eval failed")):
            result = _evaluate("bad.yaml")

        assert result["status"] == "error"
        assert result["error_message"] == "eval failed"


# ── init_project ──────────────────────────────────────────────


class TestInitProject:
    """_init_project calls CLI with agent-type and optional directory."""

    def test_calls_cli_with_agent_type(self) -> None:
        from integrations.claude_code.mcp_server.server import _init_project

        with patch(f"{_MOD}.run_cli", return_value=_cli_success()) as mock:
            result = _init_project(agent_type="a2a")

        assert result["status"] == "success"
        args_list = mock.call_args[0][0]
        assert "init" in args_list
        assert "--agent-type" in args_list
        assert "a2a" in args_list
        assert "--force" not in args_list

    def test_passes_force_flag_when_requested(self) -> None:
        from integrations.claude_code.mcp_server.server import _init_project

        with patch(f"{_MOD}.run_cli", return_value=_cli_success()) as mock:
            result = _init_project(agent_type="custom", force=True)

        assert result["status"] == "success"
        args_list = mock.call_args[0][0]
        assert "--force" in args_list

    def test_passes_directory_as_cwd(self) -> None:
        from integrations.claude_code.mcp_server.server import _init_project

        with patch(f"{_MOD}.run_cli", return_value=_cli_success()) as mock:
            _init_project(directory="/tmp/project")

        assert mock.call_args[1]["cwd"] == "/tmp/project"


class TestInitProjectFailure:
    """_init_project returns error dict on CLI failure."""

    def test_returns_error_on_cli_failure(self) -> None:
        from integrations.claude_code.mcp_server.server import _init_project

        with patch(f"{_MOD}.run_cli", return_value=_cli_error("permission denied")):
            result = _init_project(agent_type="custom")

        assert result["status"] == "error"
        assert result["error_message"] == "permission denied"

    def test_rejects_invalid_agent_type(self) -> None:
        """Invalid agent_type returns a structured error, not a crash."""
        from integrations.claude_code.mcp_server.server import _init_project

        result = _init_project(agent_type="../../etc")

        assert result["status"] == "error"
        assert "Invalid agent_type" in result["error_message"]
        assert "../../etc" in result["error_message"]


# ── launch_ui ─────────────────────────────────────────────────


class TestLaunchUi:
    """_launch_ui starts a subprocess and returns URL."""

    @pytest.fixture(autouse=True)
    def _reset_ui_state(self) -> None:
        """Reset module-level UI process state before each test."""
        from integrations.claude_code.mcp_server import server

        server._ui_process = None
        server._ui_port = None
        yield
        server._ui_process = None
        server._ui_port = None

    def test_starts_subprocess_returns_url(self) -> None:
        from integrations.claude_code.mcp_server import server

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running

        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            result = server._launch_ui(port=9090)

        assert result["status"] == "success"
        assert "9090" in result["url"]
        mock_popen.assert_called_once()
        popen_args = mock_popen.call_args[0][0]
        assert "arksim" in popen_args
        assert "ui" in popen_args
        assert "--port" in popen_args
        assert "9090" in popen_args

    def test_returns_existing_url_when_already_running(self) -> None:
        from integrations.claude_code.mcp_server import server

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        server._ui_process = mock_proc
        server._ui_port = 8080

        with patch("subprocess.Popen") as mock_popen:
            result = server._launch_ui(port=8080)

        mock_popen.assert_not_called()
        assert result["status"] == "success"
        assert "8080" in result["url"]

    def test_returns_error_when_port_mismatch(self) -> None:
        """Requesting a different port while UI is running returns an error."""
        from integrations.claude_code.mcp_server import server

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        server._ui_process = mock_proc
        server._ui_port = 8080

        with patch("subprocess.Popen") as mock_popen:
            result = server._launch_ui(port=9090)

        mock_popen.assert_not_called()
        assert result["status"] == "error"
        assert "8080" in result["error_message"]
        assert "9090" in result["error_message"]

    def test_restarts_when_process_has_exited(self) -> None:
        from integrations.claude_code.mcp_server import server

        dead_proc = MagicMock()
        dead_proc.poll.return_value = 1  # exited
        server._ui_process = dead_proc
        server._ui_port = 8080

        new_proc = MagicMock()
        new_proc.poll.return_value = None

        with patch("subprocess.Popen", return_value=new_proc) as mock_popen:
            result = server._launch_ui(port=9000)

        mock_popen.assert_called_once()
        assert result["status"] == "success"
        assert "9000" in result["url"]

    def test_includes_stderr_when_process_exits_immediately(self) -> None:
        """When the UI process crashes, stderr is included in the error."""
        from integrations.claude_code.mcp_server import server

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1  # exited immediately
        mock_proc.stderr.read.return_value = "Address already in use\n"

        with patch("subprocess.Popen", return_value=mock_proc):
            result = server._launch_ui(port=8080)

        assert result["status"] == "error"
        assert "Address already in use" in result["error_message"]

    def test_returns_error_when_arksim_not_found(self) -> None:
        from integrations.claude_code.mcp_server import server

        with patch("subprocess.Popen", side_effect=FileNotFoundError("arksim")):
            result = server._launch_ui(port=8080)

        assert result["status"] == "error"
        assert "arksim CLI not found" in result["error_message"]

    @pytest.mark.parametrize("bad_port", [0, -1, 65536, 99999])
    def test_rejects_invalid_port(self, bad_port: int) -> None:
        from integrations.claude_code.mcp_server import server

        result = server._launch_ui(port=bad_port)

        assert result["status"] == "error"
        assert "Port must be between 1 and 65535" in result["error_message"]


# ── list_results ──────────────────────────────────────────────


class TestListResults:
    """_list_results scans for evaluation.json files."""

    def test_finds_evaluation_files(self, tmp_path: Path) -> None:
        from integrations.claude_code.mcp_server.server import _list_results

        eval_data = _make_eval_data()
        eval_dir = tmp_path / "run1"
        eval_dir.mkdir()
        eval_file = eval_dir / "evaluation.json"

        import json

        eval_file.write_text(json.dumps(eval_data))

        with patch(
            f"{_MOD}.parse_json_file",
            return_value={"status": "success", "data": eval_data},
        ):
            result = _list_results(output_dir=str(tmp_path))

        assert result["status"] == "success"
        assert len(result["runs"]) == 1

        run = result["runs"][0]
        assert run["evaluation_id"] == "eval-001"
        assert run["simulation_id"] == "sim-001"
        assert run["generated_at"] == "2025-01-15T10:00:00Z"
        assert run["total_conversations"] == 3
        assert run["passed"] == 1
        assert run["partial"] == 1
        assert run["failed"] == 1
        assert run["unique_errors_count"] == 1

    def test_returns_empty_when_no_results(self, tmp_path: Path) -> None:
        from integrations.claude_code.mcp_server.server import _list_results

        result = _list_results(output_dir=str(tmp_path))

        assert result["status"] == "success"
        assert result["runs"] == []


# ── read_result ───────────────────────────────────────────────


class TestReadResult:
    """_read_result returns a structured summary of an evaluation."""

    def test_returns_structured_data(self) -> None:
        from integrations.claude_code.mcp_server.server import _read_result

        eval_data = _make_eval_data()

        with patch(
            f"{_MOD}.parse_json_file",
            return_value={"status": "success", "data": eval_data},
        ):
            result = _read_result("/tmp/evaluation.json")

        assert result["status"] == "success"
        assert result["evaluation_id"] == "eval-001"
        assert result["generated_at"] == "2025-01-15T10:00:00Z"
        assert result["total_conversations"] == 3
        assert result["passed"] == 1
        assert result["partial"] == 1
        assert result["failed"] == 1

        errors = result["unique_errors"]
        assert len(errors) == 1
        assert errors[0]["error_id"] == "err-1"
        assert errors[0]["category"] == "hallucination"
        assert errors[0]["description"] == "Made up facts"
        assert errors[0]["severity"] == "high"
        assert errors[0]["occurrence_count"] == 1

        convos = result["conversations"]
        assert len(convos) == 3
        assert convos[0]["evaluation_status"] == "Done"
        assert convos[1]["evaluation_status"] == "Partial Failure"
        assert convos[2]["evaluation_status"] == "Failed"

    def test_returns_error_for_missing_file(self) -> None:
        from integrations.claude_code.mcp_server.server import _read_result

        with patch(
            f"{_MOD}.parse_json_file",
            return_value={
                "status": "error",
                "error_message": "File not found: /tmp/missing.json",
            },
        ):
            result = _read_result("/tmp/missing.json")

        assert result["status"] == "error"
        assert "File not found" in result["error_message"]

    def test_handles_null_conversations(self) -> None:
        """Null conversations field should not crash; returns 0 conversations."""
        from integrations.claude_code.mcp_server.server import _read_result

        eval_data = _make_eval_data()
        eval_data["conversations"] = None
        eval_data["unique_errors"] = None

        with patch(
            f"{_MOD}.parse_json_file",
            return_value={"status": "success", "data": eval_data},
        ):
            result = _read_result("/tmp/evaluation.json")

        assert result["status"] == "success"
        assert result["total_conversations"] == 0
        assert result["passed"] == 0
        assert result["failed"] == 0
        assert result["unique_errors"] == []
        assert result["conversations"] == []


# ── list_results edge cases ──────────────────────────────────


class TestListResultsEdgeCases:
    """Edge-case tests for _list_results."""

    def test_returns_empty_when_output_dir_is_a_file(self, tmp_path: Path) -> None:
        """Passing a file path instead of a directory returns empty runs."""
        from integrations.claude_code.mcp_server.server import _list_results

        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("hello")

        result = _list_results(output_dir=str(file_path))

        assert result["status"] == "success"
        assert result["runs"] == []
        assert result["skipped"] == []
