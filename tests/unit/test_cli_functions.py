# SPDX-License-Identifier: Apache-2.0
"""Unit tests for pure helper functions in arksim.cli."""

from __future__ import annotations

from unittest.mock import MagicMock

from arksim.cli import (
    _check_score_threshold,
    _coerce_list_overrides,
    _merge_cli_overrides,
    _parse_value,
    parse_extra_args,
)
from arksim.evaluator import check_numeric_thresholds, check_qualitative_failure_labels
from arksim.evaluator.entities import ConversationEvaluation, Evaluation

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


# ── _merge_cli_overrides ─────────────────────────────────


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


# ── _coerce_list_overrides ───────────────────────────────


class TestCoerceListOverrides:
    def test_string_split_into_list(self) -> None:
        """Scalar string is comma-split into a list."""
        from arksim.evaluator.entities import EvaluationInput

        overrides = {"custom_metrics_file_paths": "a.py,b.py"}
        _coerce_list_overrides(overrides, EvaluationInput)
        assert overrides["custom_metrics_file_paths"] == ["a.py", "b.py"]

    def test_single_string_wrapped(self) -> None:
        from arksim.evaluator.entities import EvaluationInput

        overrides = {"custom_metrics_file_paths": "a.py"}
        _coerce_list_overrides(overrides, EvaluationInput)
        assert overrides["custom_metrics_file_paths"] == ["a.py"]

    def test_already_list_unchanged(self) -> None:
        from arksim.evaluator.entities import EvaluationInput

        overrides = {"custom_metrics_file_paths": ["a.py", "b.py"]}
        _coerce_list_overrides(overrides, EvaluationInput)
        assert overrides["custom_metrics_file_paths"] == ["a.py", "b.py"]

    def test_non_list_field_ignored(self) -> None:
        from arksim.evaluator.entities import EvaluationInput

        overrides = {"model": "gpt-4"}
        _coerce_list_overrides(overrides, EvaluationInput)
        assert overrides["model"] == "gpt-4"

    def test_optional_list_field_coerced(self) -> None:
        """Fields annotated as list[str] | None are still detected as list types."""
        from pydantic import BaseModel, Field

        class _Model(BaseModel):
            items: list[str] | None = Field(default=None)

        overrides = {"items": "x,y"}
        _coerce_list_overrides(overrides, _Model)
        assert overrides["items"] == ["x", "y"]


# ── _check_score_threshold ───────────────────────────────


def _make_score_evaluation(scores: list[float]) -> Evaluation:
    convos = [
        ConversationEvaluation(
            conversation_id=f"conv-{i}",
            goal_completion_score=s,
            goal_completion_reason="ok",
            turn_success_ratio=s,
            overall_agent_score=s,
            evaluation_status="Done",
            turn_scores=[],
        )
        for i, s in enumerate(scores)
    ]
    return Evaluation(
        schema_version="v1",
        generated_at="2024-01-01T00:00:00Z",
        evaluator_version="v1",
        evaluation_id="eval-1",
        simulation_id="sim-1",
        conversations=convos,
        unique_errors=[],
    )


class TestCheckScoreThreshold:
    def test_none_threshold_always_passes(self) -> None:
        assert _check_score_threshold(_make_score_evaluation([0.1]), None) is True

    def test_all_pass(self) -> None:
        assert _check_score_threshold(_make_score_evaluation([0.8, 0.9]), 0.7) is True

    def test_one_fails(self) -> None:
        assert _check_score_threshold(_make_score_evaluation([0.8, 0.5]), 0.7) is False

    def test_all_fail(self) -> None:
        assert _check_score_threshold(_make_score_evaluation([0.1, 0.2]), 0.5) is False

    def test_exact_threshold_passes(self) -> None:
        assert _check_score_threshold(_make_score_evaluation([0.7]), 0.7) is True


# ── check_numeric_thresholds ────────────────────────────


class TestCheckNumericThresholds:
    """Tests for check_numeric_thresholds."""

    @staticmethod
    def _make_convo(
        convo_id: str,
        metric_scores: dict[str, list[float]],
        goal_completion_score: float = -1.0,
    ) -> MagicMock:
        """Build a mock ConversationEvaluation with per-metric turn scores."""
        turns = []
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
        assert check_numeric_thresholds(ev, None) is True

    def test_empty_thresholds_passes(self) -> None:
        """Empty dict always returns True."""
        ev = self._make_evaluation([self._make_convo("c1", {"faithfulness": [3.0]})])
        assert check_numeric_thresholds(ev, {}) is True

    def test_all_above_threshold(self) -> None:
        """Returns True when all conversations pass."""
        ev = self._make_evaluation(
            [
                self._make_convo("c1", {"faithfulness": [4.0, 3.5]}),
                self._make_convo("c2", {"faithfulness": [3.5, 3.5]}),
            ]
        )
        assert check_numeric_thresholds(ev, {"faithfulness": 3.5}) is True

    def test_some_below_threshold(self) -> None:
        """Returns False when any conversation fails."""
        ev = self._make_evaluation(
            [
                self._make_convo("c1", {"faithfulness": [4.0, 4.0]}),
                self._make_convo("c2", {"faithfulness": [2.0, 2.0]}),
            ]
        )
        assert check_numeric_thresholds(ev, {"faithfulness": 3.5}) is False

    def test_exact_threshold_passes(self) -> None:
        """Score exactly equal to threshold passes."""
        ev = self._make_evaluation([self._make_convo("c1", {"faithfulness": [4.0]})])
        assert check_numeric_thresholds(ev, {"faithfulness": 4.0}) is True

    def test_goal_completion_uses_convo_score(self) -> None:
        """goal_completion reads the per-conversation score, not turn scores."""
        ev = self._make_evaluation(
            [self._make_convo("c1", {}, goal_completion_score=0.9)]
        )
        assert check_numeric_thresholds(ev, {"goal_completion": 0.8}) is True

    def test_goal_completion_not_computed_skips(self) -> None:
        """goal_completion_score == -1 (not computed) is skipped with a warning."""
        ev = self._make_evaluation(
            [self._make_convo("c1", {}, goal_completion_score=-1.0)]
        )
        assert check_numeric_thresholds(ev, {"goal_completion": 0.8}) is True

    def test_metric_not_found_skips_with_warning(self) -> None:
        """Metric absent from all turns is skipped — does not cause failure."""
        ev = self._make_evaluation([self._make_convo("c1", {"helpfulness": [3.0]})])
        assert check_numeric_thresholds(ev, {"not_exist": 4.0}) is True

    def test_multiple_metrics_both_fail_reported(self) -> None:
        """Both failing metrics are reported before returning False."""
        ev = self._make_evaluation(
            [self._make_convo("c1", {"faithfulness": [1.0], "helpfulness": [1.0]})]
        )
        assert (
            check_numeric_thresholds(ev, {"faithfulness": 4.0, "helpfulness": 3.0})
            is False
        )

    def test_scores_with_skipped_turns_excluded(self) -> None:
        """Turn scores with value < 0 (SCORE_NOT_COMPUTED) are excluded from mean."""
        ev = self._make_evaluation(
            [self._make_convo("c1", {"faithfulness": [4.0, -1.0]})]
        )
        assert check_numeric_thresholds(ev, {"faithfulness": 3.5}) is True


# ── check_qualitative_failure_labels ────────────────────────


class TestCheckQualitativeThresholds:
    """Tests for check_qualitative_failure_labels."""

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
        assert check_qualitative_failure_labels(ev, None) is True

    def test_no_turns_have_failure_label(self) -> None:
        """Returns True when no turn has a failure label."""
        ev = self._make_evaluation(
            [self._make_convo("c1", abf_labels=["no failure", "no failure"])]
        )
        assert (
            check_qualitative_failure_labels(
                ev, {"agent_behavior_failure": ["false information"]}
            )
            is True
        )

    def test_one_turn_has_failure_label(self) -> None:
        """Returns False when any turn has a label in the failure list."""
        ev = self._make_evaluation(
            [self._make_convo("c1", abf_labels=["no failure", "false information"])]
        )
        assert (
            check_qualitative_failure_labels(
                ev, {"agent_behavior_failure": ["false information"]}
            )
            is False
        )

    def test_skip_outcomes_are_ignored(self) -> None:
        """System skip outcomes are not counted as failures."""
        ev = self._make_evaluation(
            [
                self._make_convo(
                    "c1",
                    abf_labels=["skipped_good_performance", "evaluation_run_failed"],
                )
            ]
        )
        assert (
            check_qualitative_failure_labels(
                ev, {"agent_behavior_failure": ["false information"]}
            )
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
                )
            ]
        )
        assert (
            check_qualitative_failure_labels(
                ev, {"agent_behavior_failure": ["disobey user request"]}
            )
            is True
        )

    def test_custom_qual_metric_uses_qual_scores(self) -> None:
        """Custom qualitative metrics are read from turn.qual_scores."""
        ev = self._make_evaluation(
            [
                self._make_convo(
                    "c1", qual_scores={"prohibited_statements": ["clean", "clean"]}
                )
            ]
        )
        assert (
            check_qualitative_failure_labels(
                ev, {"prohibited_statements": ["violated"]}
            )
            is True
        )

    def test_metric_absent_from_turn_skips(self) -> None:
        """Turns where the metric is absent are skipped — not counted as failures."""
        ev = self._make_evaluation([self._make_convo("c1", qual_scores={})])
        assert (
            check_qualitative_failure_labels(
                ev, {"prohibited_statements": ["violated"]}
            )
            is True
        )

    def test_custom_qual_metric_not_in_turn_qual_scores_skips(self) -> None:
        """Turn exists but checked metric is not in its qual_scores — skipped, not failed."""
        ev = self._make_evaluation(
            [self._make_convo("c1", qual_scores={"other_metric": ["ok"]})]
        )
        assert (
            check_qualitative_failure_labels(
                ev, {"prohibited_statements": ["violated"]}
            )
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
                )
            ]
        )
        assert (
            check_qualitative_failure_labels(
                ev,
                {
                    "agent_behavior_failure": ["false information"],
                    "prohibited_statements": ["violated"],
                },
            )
            is False
        )
