# SPDX-License-Identifier: Apache-2.0
"""Tests for CLI argument parsing and utility functions."""

from __future__ import annotations

import io
import os
import sys
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from arksim.cli import (
    EXIT_CONFIG_ERROR,
    EXIT_EVAL_FAILED,
    EXIT_INTERNAL_ERROR,
    EXIT_NETWORK_ERROR,
    _check_numeric_thresholds,
    _check_qualitative_thresholds,
    _check_score_threshold,
    _merge_cli_overrides,
    _parse_value,
    _run_examples,
    build_parser,
    main,
    parse_extra_args,
    validate_overrides,
)

# ── _parse_value ─────────────────────────────────────────


class TestParseValue:
    """Tests for _parse_value type coercion."""

    def test_true_strings(self) -> None:
        """Parses 'true' and 'yes' as True."""
        assert _parse_value("true") is True
        assert _parse_value("True") is True
        assert _parse_value("yes") is True
        assert _parse_value("YES") is True

    def test_false_strings(self) -> None:
        """Parses 'false' and 'no' as False."""
        assert _parse_value("false") is False
        assert _parse_value("False") is False
        assert _parse_value("no") is False
        assert _parse_value("NO") is False

    def test_integer(self) -> None:
        """Parses integer strings."""
        assert _parse_value("42") == 42
        assert _parse_value("0") == 0
        assert _parse_value("-1") == -1

    def test_float(self) -> None:
        """Parses float strings."""
        assert _parse_value("3.14") == 3.14
        assert _parse_value("0.5") == 0.5

    def test_string_fallback(self) -> None:
        """Returns string when no other type matches."""
        assert _parse_value("hello") == "hello"
        assert _parse_value("./path/to/file") == "./path/to/file"


# ── parse_extra_args ─────────────────────────────────────


class TestParseExtraArgs:
    """Tests for parse_extra_args CLI override parser."""

    def test_empty_list(self) -> None:
        """Empty input returns empty dict."""
        assert parse_extra_args([]) == {}

    def test_key_value_pair(self) -> None:
        """Parses --key value format."""
        result = parse_extra_args(["--model", "gpt-4"])
        assert result == {"model": "gpt-4"}

    def test_key_equals_value(self) -> None:
        """Parses --key=value format."""
        result = parse_extra_args(["--seed=42"])
        assert result == {"seed": 42}

    def test_boolean_flag(self) -> None:
        """Bare --flag is treated as True."""
        result = parse_extra_args(["--verbose"])
        assert result == {"verbose": True}

    def test_dash_to_underscore(self) -> None:
        """Dashes in key names are converted to underscores."""
        result = parse_extra_args(["--max-turns", "5"])
        assert result == {"max_turns": 5}

    def test_multiple_args(self) -> None:
        """Parses multiple arguments together."""
        result = parse_extra_args(
            ["--model", "gpt-4", "--max-turns", "10", "--verbose"]
        )
        assert result == {"model": "gpt-4", "max_turns": 10, "verbose": True}

    def test_type_coercion(self) -> None:
        """Values are coerced to appropriate types."""
        result = parse_extra_args(["--count", "5", "--rate", "0.5", "--flag", "true"])
        assert result == {"count": 5, "rate": 0.5, "flag": True}

    def test_skips_non_flag_args(self) -> None:
        """Non-flag arguments are silently skipped."""
        result = parse_extra_args(["positional", "--key", "val"])
        assert result == {"key": "val"}


# ── _merge_cli_overrides ────────────────────────────────


class TestMergeCliOverrides:
    """Tests for _merge_cli_overrides."""

    def test_override_replaces_yaml(self) -> None:
        """CLI values override YAML values."""
        yaml = {"model": "gpt-3.5", "max_turns": 5}
        cli = {"model": "gpt-4"}
        result = _merge_cli_overrides(yaml, cli)
        assert result["model"] == "gpt-4"
        assert result["max_turns"] == 5

    def test_none_values_skipped(self) -> None:
        """None values in overrides are ignored."""
        yaml = {"model": "gpt-3.5"}
        cli = {"model": None}
        result = _merge_cli_overrides(yaml, cli)
        assert result["model"] == "gpt-3.5"

    def test_new_keys_added(self) -> None:
        """New keys from CLI are added to result."""
        yaml = {"model": "gpt-3.5"}
        cli = {"verbose": True}
        result = _merge_cli_overrides(yaml, cli)
        assert result == {"model": "gpt-3.5", "verbose": True}

    def test_original_not_mutated(self) -> None:
        """Original YAML dict is not modified."""
        yaml = {"model": "gpt-3.5"}
        cli = {"model": "gpt-4"}
        _merge_cli_overrides(yaml, cli)
        assert yaml["model"] == "gpt-3.5"


# ── validate_overrides ──────────────────────────────────


class TestValidateOverrides:
    """Tests for validate_overrides."""

    def test_valid_keys_pass(self) -> None:
        """No exit when all keys are valid."""
        validate_overrides({"model": "gpt-4"}, {"model", "provider"})

    def test_invalid_keys_exit(self) -> None:
        """Exits with EXIT_CONFIG_ERROR when unknown keys are present."""
        with pytest.raises(SystemExit) as exc_info:
            validate_overrides({"bad_key": "val"}, {"model"})
        assert exc_info.value.code == EXIT_CONFIG_ERROR

    def test_empty_overrides_pass(self) -> None:
        """Empty overrides always pass."""
        validate_overrides({}, {"model"})


# ── _check_score_threshold ──────────────────────────────


class TestCheckScoreThreshold:
    """Tests for _check_score_threshold."""

    @staticmethod
    def _make_evaluation(scores: list[float]) -> MagicMock:
        """Build a mock Evaluation with given scores."""
        convos = []
        for i, score in enumerate(scores):
            c = MagicMock()
            c.conversation_id = f"convo-{i}"
            c.overall_agent_score = score
            convos.append(c)
        ev = MagicMock()
        ev.conversations = convos
        return ev

    def test_none_threshold_passes(self) -> None:
        """None threshold always returns True."""
        ev = self._make_evaluation([0.1])
        assert _check_score_threshold(ev, None) is True

    def test_all_above_threshold(self) -> None:
        """Returns True when all scores pass."""
        ev = self._make_evaluation([0.9, 0.8, 0.85])
        assert _check_score_threshold(ev, 0.7) is True

    def test_some_below_threshold(self) -> None:
        """Returns False when any score is below threshold."""
        ev = self._make_evaluation([0.9, 0.3, 0.8])
        assert _check_score_threshold(ev, 0.5) is False

    def test_exact_threshold_passes(self) -> None:
        """Score equal to threshold passes."""
        ev = self._make_evaluation([0.7])
        assert _check_score_threshold(ev, 0.7) is True


# ── build_parser ────────────────────────────────────────


class TestBuildParser:
    """Tests for build_parser subcommand structure."""

    def test_all_subcommands_registered(self) -> None:
        """Parser has all expected subcommands."""
        parser = build_parser()
        # Parse each command to verify it's registered
        for cmd in [
            "simulate",
            "evaluate",
            "simulate-evaluate",
            "show-prompts",
            "examples",
            "ui",
        ]:
            args = parser.parse_args([cmd])
            assert args.command == cmd

    def test_simulate_with_config_and_overrides(self) -> None:
        """Simulate parses config_file and REMAINDER overrides."""
        parser = build_parser()
        args = parser.parse_args(["simulate", "config.yaml", "--max-turns", "3"])
        assert args.command == "simulate"
        assert args.config_file == "config.yaml"
        assert args.additional_args == ["--max-turns", "3"]

    def test_simulate_without_config(self) -> None:
        """Simulate with no config sets config_file to None."""
        parser = build_parser()
        args = parser.parse_args(["simulate"])
        assert args.config_file is None

    def test_examples_list_flag(self) -> None:
        """Examples --list sets list_only to True."""
        parser = build_parser()
        args = parser.parse_args(["examples", "--list"])
        assert args.command == "examples"
        assert args.list_only is True
        assert args.name is None

    def test_examples_with_name(self) -> None:
        """Examples with a name sets args.name."""
        parser = build_parser()
        args = parser.parse_args(["examples", "bank-insurance"])
        assert args.name == "bank-insurance"
        assert args.list_only is False

    def test_show_prompts_category(self) -> None:
        """Show-prompts --category sets args.category."""
        parser = build_parser()
        args = parser.parse_args(["show-prompts", "--category", "faithfulness"])
        assert args.category == "faithfulness"

    def test_show_prompts_default_category(self) -> None:
        """Show-prompts without --category defaults to None."""
        parser = build_parser()
        args = parser.parse_args(["show-prompts"])
        assert args.category is None

    def test_ui_custom_port(self) -> None:
        """UI --port sets custom port."""
        parser = build_parser()
        args = parser.parse_args(["ui", "--port", "9090"])
        assert args.port == 9090

    def test_ui_default_port(self) -> None:
        """UI without --port defaults to 8080."""
        parser = build_parser()
        args = parser.parse_args(["ui"])
        assert args.port == 8080


# ── _run_examples ───────────────────────────────────────


def _make_tarball(members: dict[str, str]) -> bytes:
    """Build an in-memory .tar.gz with given name→content entries."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in members.items():
            data = content.encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class TestRunExamples:
    """Tests for _run_examples."""

    @pytest.fixture()
    def fake_tarball(self) -> bytes:
        """A minimal tarball mimicking the GitHub archive."""
        return _make_tarball(
            {
                "arksim-main/examples/demo/config.yaml": "model: gpt-4",
                "arksim-main/examples/demo/scenario.json": "[]",
                "arksim-main/examples/bank-insurance/config.yaml": "model: gpt-4",
                "arksim-main/README.md": "# arksim",
            }
        )

    def test_list_only(
        self, fake_tarball: bytes, capsys: pytest.CaptureFixture
    ) -> None:
        """--list prints available example names."""
        resp = MagicMock()
        resp.read.return_value = fake_tarball
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=resp):
            _run_examples(list_only=True)

        out = capsys.readouterr().out
        assert "demo" in out
        assert "bank-insurance" in out

    def test_unknown_name_exits(self, fake_tarball: bytes) -> None:
        """Unknown example name exits with EXIT_CONFIG_ERROR."""
        resp = MagicMock()
        resp.read.return_value = fake_tarball
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=resp):
            with pytest.raises(SystemExit) as exc_info:
                _run_examples(name="nonexistent")
            assert exc_info.value.code == EXIT_CONFIG_ERROR

    def test_dest_exists_exits(self, fake_tarball: bytes, temp_dir: str) -> None:
        """Exits with EXIT_CONFIG_ERROR when destination already exists."""
        resp = MagicMock()
        resp.read.return_value = fake_tarball
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)

        # Create the destination so it already exists
        dest = os.path.join(temp_dir, "examples")
        os.makedirs(dest)

        with (
            patch("urllib.request.urlopen", return_value=resp),
            patch("os.path.exists", return_value=True),
            pytest.raises(SystemExit) as exc_info,
        ):
            _run_examples()
        assert exc_info.value.code == EXIT_CONFIG_ERROR

    def test_path_traversal_skipped(self, temp_dir: str) -> None:
        """Members with '..' or absolute paths are skipped."""
        tarball = _make_tarball(
            {
                "arksim-main/examples/demo/safe.txt": "ok",
                "arksim-main/examples/demo/../../etc/passwd": "evil",
                "arksim-main/examples/demo//tmp/evil": "evil",
            }
        )
        resp = MagicMock()
        resp.read.return_value = tarball
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)

        with (
            patch("urllib.request.urlopen", return_value=resp),
            patch("os.path.exists", return_value=False),
            patch("tarfile.TarFile.extract") as mock_extract,
        ):
            _run_examples(name="demo")

        # Only the safe member should have been extracted
        extracted_names = [call.args[0].name for call in mock_extract.call_args_list]
        for name in extracted_names:
            assert ".." not in name.split("/")
            assert not os.path.isabs(name)

        # Verify filter="data" is passed on 3.12+ and omitted on older versions
        for call in mock_extract.call_args_list:
            if sys.version_info >= (3, 12):
                assert call.kwargs.get("filter") == "data"
            else:
                assert "filter" not in call.kwargs


class TestMainEvaluateValidation:
    """Focused tests for standalone evaluate path validation."""

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


# ── _check_numeric_thresholds ────────────────────────────


class TestCheckNumericThresholds:
    """Tests for _check_numeric_thresholds."""

    @staticmethod
    def _make_convo(
        convo_id: str,
        metric_scores: dict[str, list[float]],
        goal_completion_score: float = -1.0,
    ) -> MagicMock:
        """Build a mock ConversationEvaluation with per-metric turn scores."""
        turns = []
        # Collect all turn ids across metrics
        max_turns = max((len(v) for v in metric_scores.values()), default=0)
        for i in range(max_turns):
            turn = MagicMock()
            turn.turn_id = i
            scores = []
            for name, values in metric_scores.items():
                if i < len(values):
                    r = MagicMock()
                    r.name = name
                    r.value = values[i]
                    scores.append(r)
            turn.scores = scores
            turns.append(turn)
        convo = MagicMock()
        convo.conversation_id = convo_id
        convo.turn_scores = turns
        convo.goal_completion_score = goal_completion_score
        return convo

    @staticmethod
    def _make_evaluation(convos: list) -> MagicMock:
        ev = MagicMock()
        ev.conversations = convos
        return ev

    def test_none_thresholds_passes(self) -> None:
        """None thresholds always returns True."""
        ev = self._make_evaluation([self._make_convo("c1", {"faithfulness": [3.0]})])
        assert _check_numeric_thresholds(ev, None) is True

    def test_empty_thresholds_passes(self) -> None:
        """Empty dict always returns True."""
        ev = self._make_evaluation([self._make_convo("c1", {"faithfulness": [3.0]})])
        assert _check_numeric_thresholds(ev, {}) is True

    def test_all_above_threshold(self) -> None:
        """Returns True when all conversations pass."""
        # mean >= 3.5
        ev = self._make_evaluation(
            [
                self._make_convo("c1", {"faithfulness": [4.0, 3.5]}),
                self._make_convo("c2", {"faithfulness": [3.5, 3.5]}),
            ]
        )
        assert _check_numeric_thresholds(ev, {"faithfulness": 3.5}) is True

    def test_some_below_threshold(self) -> None:
        """Returns False when any conversation fails."""
        # c2 mean = (2.0+2.0)/2 = 2.0 < 3.5
        ev = self._make_evaluation(
            [
                self._make_convo("c1", {"faithfulness": [4.0, 4.0]}),
                self._make_convo("c2", {"faithfulness": [2.0, 2.0]}),
            ]
        )
        assert _check_numeric_thresholds(ev, {"faithfulness": 3.5}) is False

    def test_exact_threshold_passes(self) -> None:
        """Score exactly equal to threshold passes."""
        # mean = 4.0 == 4.0
        ev = self._make_evaluation([self._make_convo("c1", {"faithfulness": [4.0]})])
        assert _check_numeric_thresholds(ev, {"faithfulness": 4.0}) is True

    def test_goal_completion_uses_convo_score(self) -> None:
        """goal_completion reads the per-conversation score, not turn scores."""
        # goal_completion_score=0.9, threshold=0.8 → pass
        ev = self._make_evaluation(
            [self._make_convo("c1", {}, goal_completion_score=0.9)]
        )
        assert _check_numeric_thresholds(ev, {"goal_completion": 0.8}) is True

    def test_goal_completion_not_computed_skips(self) -> None:
        """goal_completion_score == -1 (not computed) is skipped with a warning."""
        ev = self._make_evaluation(
            [self._make_convo("c1", {}, goal_completion_score=-1.0)]
        )
        # Should pass (skipped, not failed)
        assert _check_numeric_thresholds(ev, {"goal_completion": 0.8}) is True

    def test_metric_not_found_skips_with_warning(self) -> None:
        """Metric absent from all turns is skipped — does not cause failure."""
        ev = self._make_evaluation([self._make_convo("c1", {"helpfulness": [3.0]})])
        assert _check_numeric_thresholds(ev, {"not_exist": 4.0}) is True

    def test_multiple_metrics_both_fail_reported(self) -> None:
        """Both failing metrics are reported before returning False."""
        # faithfulness mean = 1.0 < 4.0; helpfulness mean = 1.0 < 3.0
        ev = self._make_evaluation(
            [self._make_convo("c1", {"faithfulness": [1.0], "helpfulness": [1.0]})]
        )
        assert (
            _check_numeric_thresholds(ev, {"faithfulness": 4.0, "helpfulness": 3.0})
            is False
        )

    def test_scores_with_skipped_turns_excluded(self) -> None:
        """Turn scores with value < 0 (SCORE_NOT_COMPUTED) are excluded from mean."""
        # Only the 4.0 turn counts; -1 is excluded → mean = 4.0 >= 3.5
        ev = self._make_evaluation(
            [self._make_convo("c1", {"faithfulness": [4.0, -1.0]})]
        )
        assert _check_numeric_thresholds(ev, {"faithfulness": 3.5}) is True


# ── _check_qualitative_thresholds ───────────────────────


class TestCheckQualitativeThresholds:
    """Tests for _check_qualitative_thresholds."""

    @staticmethod
    def _make_convo(
        convo_id: str,
        abf_labels: list[str] | None = None,
        qual_scores: dict[str, list[str]] | None = None,
    ) -> MagicMock:
        """Build a mock ConversationEvaluation.

        abf_labels: per-turn agent_behavior_failure labels.
        qual_scores: dict of metric_name -> per-turn label values.
        """
        turns = []
        max_turns = max(
            len(abf_labels) if abf_labels else 0,
            *(len(v) for v in (qual_scores or {}).values()),
            0,
        )
        for i in range(max_turns):
            turn = MagicMock()
            turn.turn_id = i
            turn.turn_behavior_failure = (
                abf_labels[i] if abf_labels and i < len(abf_labels) else "no failure"
            )
            qs = []
            for name, labels in (qual_scores or {}).items():
                if i < len(labels):
                    q = MagicMock()
                    q.name = name
                    q.value = labels[i]
                    qs.append(q)
            turn.qual_scores = qs
            turns.append(turn)
        convo = MagicMock()
        convo.conversation_id = convo_id
        convo.turn_scores = turns
        return convo

    @staticmethod
    def _make_evaluation(convos: list) -> MagicMock:
        ev = MagicMock()
        ev.conversations = convos
        return ev

    def test_none_thresholds_passes(self) -> None:
        """None thresholds always returns True."""
        ev = self._make_evaluation([self._make_convo("c1", abf_labels=["no failure"])])
        assert _check_qualitative_thresholds(ev, None) is True

    def test_all_turns_match_required_label(self) -> None:
        """Returns True when every turn has the required label."""
        ev = self._make_evaluation(
            [
                self._make_convo("c1", abf_labels=["no failure", "no failure"]),
            ]
        )
        assert (
            _check_qualitative_thresholds(ev, {"agent_behavior_failure": "no failure"})
            is True
        )

    def test_one_turn_fails_required_label(self) -> None:
        """Returns False when any turn has a different label."""
        ev = self._make_evaluation(
            [
                self._make_convo("c1", abf_labels=["no failure", "false information"]),
            ]
        )
        assert (
            _check_qualitative_thresholds(ev, {"agent_behavior_failure": "no failure"})
            is False
        )

    def test_skip_outcomes_are_ignored(self) -> None:
        """System skip outcomes are not counted as failures."""
        ev = self._make_evaluation(
            [
                self._make_convo(
                    "c1",
                    abf_labels=["skipped_good_performance", "evaluation_run_failed"],
                ),
            ]
        )
        assert (
            _check_qualitative_thresholds(ev, {"agent_behavior_failure": "no failure"})
            is True
        )

    def test_agent_behavior_failure_uses_turn_field(self) -> None:
        """agent_behavior_failure reads turn.turn_behavior_failure, not qual_scores."""
        ev = self._make_evaluation(
            [
                self._make_convo(
                    "c1",
                    abf_labels=["no failure"],
                    qual_scores={"agent_behavior_failure": ["disobey user request"]},
                ),
            ]
        )
        # Should use turn_behavior_failure ("no failure"), not qual_scores
        assert (
            _check_qualitative_thresholds(ev, {"agent_behavior_failure": "no failure"})
            is True
        )

    def test_custom_qual_metric_uses_qual_scores(self) -> None:
        """Custom qualitative metrics are read from turn.qual_scores."""
        ev = self._make_evaluation(
            [
                self._make_convo(
                    "c1",
                    qual_scores={"prohibited_statements": ["clean", "clean"]},
                ),
            ]
        )
        assert (
            _check_qualitative_thresholds(ev, {"prohibited_statements": "clean"})
            is True
        )

    def test_metric_absent_from_turn_skips(self) -> None:
        """Turns where the metric is absent are skipped — not counted as failures."""
        ev = self._make_evaluation(
            [
                self._make_convo("c1", qual_scores={}),
            ]
        )
        assert (
            _check_qualitative_thresholds(ev, {"prohibited_statements": "clean"})
            is True
        )

    def test_multiple_metrics_both_fail_reported(self) -> None:
        """All failing metrics are reported before returning False."""
        ev = self._make_evaluation(
            [
                self._make_convo(
                    "c1",
                    abf_labels=["false information"],
                    qual_scores={"prohibited_statements": ["violated"]},
                ),
            ]
        )
        assert (
            _check_qualitative_thresholds(
                ev,
                {
                    "agent_behavior_failure": "no failure",
                    "prohibited_statements": "clean",
                },
            )
            is False
        )


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

    def test_network_error_exits_network_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """httpx.NetworkError exits with EXIT_NETWORK_ERROR."""
        cfg = self._eval_config(tmp_path)
        monkeypatch.setattr(sys, "argv", ["arksim", "evaluate", str(cfg)])
        with (
            patch(
                "arksim.cli.run_evaluation",
                side_effect=httpx.NetworkError("connection refused"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == EXIT_NETWORK_ERROR

    def test_timeout_error_exits_network_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """httpx.TimeoutException exits with EXIT_NETWORK_ERROR."""
        cfg = self._eval_config(tmp_path)
        monkeypatch.setattr(sys, "argv", ["arksim", "evaluate", str(cfg)])
        with (
            patch(
                "arksim.cli.run_evaluation",
                side_effect=httpx.TimeoutException("timed out"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == EXIT_NETWORK_ERROR

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


# ── main() evaluate threshold exit codes ─────────────────


class TestMainEvaluateThresholds:
    """Integration tests for threshold-based exit codes in the evaluate command."""

    @staticmethod
    def _config(tmp_path: Path, extra: str = "") -> Path:
        sim_file = tmp_path / "sim.json"
        sim_file.write_text("[]")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            f"simulation_file_path: {sim_file}\nmodel: gpt-5.1\nprovider: openai\n{extra}"
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

    def test_numeric_threshold_failure_exits_eval_failed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Exits with EXIT_EVAL_FAILED when a numeric threshold is not met."""
        cfg = self._config(tmp_path, "numeric_thresholds:\n  goal_completion: 0.9\n")
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
        cfg = self._config(
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
        cfg = self._config(
            tmp_path,
            "score_threshold: 0.5\nnumeric_thresholds:\n  goal_completion: 0.5\n",
        )
        monkeypatch.setattr(sys, "argv", ["arksim", "evaluate", str(cfg)])
        with patch("arksim.cli.run_evaluation", return_value=self._mock_eval(0.9)):
            main()  # should not raise SystemExit

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
