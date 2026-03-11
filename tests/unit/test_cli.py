# SPDX-License-Identifier: Apache-2.0
"""Integration tests for main() in arksim.cli."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from arksim.cli import (
    EXIT_CONFIG_ERROR,
    EXIT_EVAL_FAILED,
    EXIT_INTERNAL_ERROR,
    main,
)

# ── main() input validation ───────────────────────────────


class TestMainEvaluateValidation:
    """Tests for input validation and config-loading errors in main()."""

    def test_no_command_exits_config_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exits with EXIT_CONFIG_ERROR when no subcommand is given."""
        monkeypatch.setattr(sys, "argv", ["arksim"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == EXIT_CONFIG_ERROR

    def test_unreadable_config_file_exits_config_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Exits with EXIT_CONFIG_ERROR when the config file cannot be opened."""
        nonexistent = tmp_path / "no_such_dir" / "config.yaml"
        monkeypatch.setattr(sys, "argv", ["arksim", "evaluate", str(nonexistent)])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == EXIT_CONFIG_ERROR

    def test_evaluate_missing_simulation_file_path_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Exits with EXIT_CONFIG_ERROR when simulation_file_path is missing."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("model: gpt-5.1\nprovider: openai\n")
        monkeypatch.setattr(sys, "argv", ["arksim", "evaluate", str(cfg_path)])

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == EXIT_CONFIG_ERROR

    def test_evaluate_nonexistent_simulation_file_path_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Exits with EXIT_CONFIG_ERROR when simulation_file_path does not exist."""
        missing_path = tmp_path / "missing_simulation.json"
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            f"simulation_file_path: {missing_path}\nmodel: gpt-5.1\nprovider: openai\n"
        )
        monkeypatch.setattr(sys, "argv", ["arksim", "evaluate", str(cfg_path)])

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == EXIT_CONFIG_ERROR


# ── main() exception handlers ─────────────────────────────


class TestMainExceptionHandlers:
    """Tests for the try/except handlers in main()."""

    @staticmethod
    def _eval_config(tmp_path: Path) -> Path:
        sim_file = tmp_path / "sim.json"
        sim_file.write_text("[]")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            f"simulation_file_path: {sim_file}\nmodel: gpt-5.1\nprovider: openai\n"
        )
        return cfg

    @staticmethod
    def _make_pydantic_error() -> ValidationError:
        class _M(BaseModel):
            x: int

        try:
            _M.model_validate({"x": "not_an_int"})
        except ValidationError as exc:
            return exc
        raise AssertionError("unreachable")  # pragma: no cover

    def test_validation_error_exits_config_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """ValidationError raised inside main() exits with EXIT_CONFIG_ERROR."""
        cfg = self._eval_config(tmp_path)
        monkeypatch.setattr(sys, "argv", ["arksim", "evaluate", str(cfg)])
        with (
            patch(
                "arksim.cli.run_evaluation",
                side_effect=self._make_pydantic_error(),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == EXIT_CONFIG_ERROR

    def test_internal_error_exits_internal_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Unhandled Exception exits with EXIT_INTERNAL_ERROR."""
        cfg = self._eval_config(tmp_path)
        monkeypatch.setattr(sys, "argv", ["arksim", "evaluate", str(cfg)])
        with (
            patch(
                "arksim.cli.run_evaluation",
                side_effect=RuntimeError("unexpected failure"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == EXIT_INTERNAL_ERROR


# ── main() simulate / evaluate threshold exit codes ───────


class TestMainEvaluateThresholds:
    """Integration tests for threshold-based exit codes."""

    @staticmethod
    def _eval_config(tmp_path: Path, extra: str = "") -> Path:
        sim_file = tmp_path / "sim.json"
        sim_file.write_text("[]")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            f"simulation_file_path: {sim_file}\nmodel: gpt-5.1\nprovider: openai\n{extra}"
        )
        return cfg

    @staticmethod
    def _sim_eval_config(tmp_path: Path, extra: str = "") -> Path:
        scenario_file = tmp_path / "scenarios.json"
        scenario_file.write_text("[]")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            f"scenario_file_path: {scenario_file}\n"
            "model: gpt-5.1\nprovider: openai\n"
            "agent_config:\n"
            "  agent_type: chat_completions\n"
            "  agent_name: test-agent\n"
            "  api_config:\n"
            "    endpoint: http://localhost/chat\n"
            "    body:\n"
            "      messages: []\n"
            f"{extra}"
        )
        return cfg

    @staticmethod
    def _mock_eval(goal_completion: float = 0.9) -> MagicMock:
        c = MagicMock()
        c.conversation_id = "c1"
        c.overall_agent_score = goal_completion
        c.goal_completion_score = goal_completion
        c.turn_scores = []
        ev = MagicMock()
        ev.conversations = [c]
        return ev

    def test_simulate_runs_normally(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """simulate command completes without error when mocked."""
        cfg = self._sim_eval_config(tmp_path)
        monkeypatch.setattr(sys, "argv", ["arksim", "simulate", str(cfg)])
        with patch("arksim.cli.asyncio.run", return_value=MagicMock()):
            main()  # should not raise SystemExit

    def test_numeric_threshold_failure_exits_eval_failed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Exits with EXIT_EVAL_FAILED when a numeric threshold is not met."""
        cfg = self._eval_config(
            tmp_path, "numeric_thresholds:\n  goal_completion: 0.9\n"
        )
        monkeypatch.setattr(sys, "argv", ["arksim", "evaluate", str(cfg)])
        with (
            patch("arksim.cli.run_evaluation", return_value=self._mock_eval(0.5)),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == EXIT_EVAL_FAILED

    def test_qualitative_threshold_failure_exits_eval_failed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Exits with EXIT_EVAL_FAILED when a qualitative threshold is not met."""
        cfg = self._eval_config(
            tmp_path,
            'qualitative_thresholds:\n  agent_behavior_failure: "no failure"\n',
        )
        monkeypatch.setattr(sys, "argv", ["arksim", "evaluate", str(cfg)])
        turn = MagicMock()
        turn.turn_behavior_failure = "false information"
        turn.qual_scores = []
        c = MagicMock()
        c.conversation_id = "c1"
        c.overall_agent_score = 0.9
        c.goal_completion_score = 0.9
        c.turn_scores = [turn]
        ev = MagicMock()
        ev.conversations = [c]
        with (
            patch("arksim.cli.run_evaluation", return_value=ev),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == EXIT_EVAL_FAILED

    def test_all_thresholds_pass_returns_normally(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No SystemExit when all thresholds pass."""
        cfg = self._eval_config(
            tmp_path,
            "score_threshold: 0.5\nnumeric_thresholds:\n  goal_completion: 0.5\n",
        )
        monkeypatch.setattr(sys, "argv", ["arksim", "evaluate", str(cfg)])
        with patch("arksim.cli.run_evaluation", return_value=self._mock_eval(0.9)):
            main()  # should not raise SystemExit

    def test_simulate_evaluate_threshold_failure_exits_eval_failed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Exits with EXIT_EVAL_FAILED when simulate-evaluate numeric threshold is not met."""
        cfg = self._sim_eval_config(
            tmp_path, "numeric_thresholds:\n  goal_completion: 0.9\n"
        )
        monkeypatch.setattr(sys, "argv", ["arksim", "simulate-evaluate", str(cfg)])
        with (
            patch("arksim.cli.asyncio.run", return_value=MagicMock()),
            patch("arksim.cli.run_evaluation", return_value=self._mock_eval(0.5)),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == EXIT_EVAL_FAILED

    def test_simulate_evaluate_all_thresholds_pass_returns_normally(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No SystemExit when simulate-evaluate thresholds all pass."""
        cfg = self._sim_eval_config(
            tmp_path, "numeric_thresholds:\n  goal_completion: 0.5\n"
        )
        monkeypatch.setattr(sys, "argv", ["arksim", "simulate-evaluate", str(cfg)])
        with (
            patch("arksim.cli.asyncio.run", return_value=MagicMock()),
            patch("arksim.cli.run_evaluation", return_value=self._mock_eval(0.9)),
        ):
            main()  # should not raise SystemExit
