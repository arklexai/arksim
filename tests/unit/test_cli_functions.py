# SPDX-License-Identifier: Apache-2.0
"""Tests for pure functions in arksim.cli."""

from __future__ import annotations

from arksim.cli import (
    _check_score_threshold,
    _coerce_list_overrides,
    _merge_cli_overrides,
    _parse_value,
    parse_extra_args,
)
from arksim.evaluator.entities import ConversationEvaluation, Evaluation


def _make_evaluation(scores: list[float]) -> Evaluation:
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
        assert _check_score_threshold(_make_evaluation([0.1]), None) is True

    def test_all_pass(self) -> None:
        assert _check_score_threshold(_make_evaluation([0.8, 0.9]), 0.7) is True

    def test_one_fails(self) -> None:
        assert _check_score_threshold(_make_evaluation([0.8, 0.5]), 0.7) is False

    def test_all_fail(self) -> None:
        assert _check_score_threshold(_make_evaluation([0.1, 0.2]), 0.5) is False

    def test_exact_threshold_passes(self) -> None:
        assert _check_score_threshold(_make_evaluation([0.7]), 0.7) is True


class TestMergeCliOverrides:
    def test_cli_overrides_yaml(self) -> None:
        result = _merge_cli_overrides({"a": 1, "b": 2}, {"b": 99})
        assert result == {"a": 1, "b": 99}

    def test_none_values_ignored(self) -> None:
        result = _merge_cli_overrides({"a": 1}, {"a": None, "b": None})
        assert result == {"a": 1}

    def test_new_keys_added(self) -> None:
        result = _merge_cli_overrides({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_empty_both(self) -> None:
        assert _merge_cli_overrides({}, {}) == {}


class TestParseValue:
    def test_true_strings(self) -> None:
        assert _parse_value("true") is True
        assert _parse_value("yes") is True
        assert _parse_value("True") is True

    def test_false_strings(self) -> None:
        assert _parse_value("false") is False
        assert _parse_value("no") is False

    def test_integer(self) -> None:
        assert _parse_value("42") == 42

    def test_float(self) -> None:
        assert _parse_value("3.14") == 3.14

    def test_string_passthrough(self) -> None:
        assert _parse_value("hello") == "hello"


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


class TestParseExtraArgs:
    def test_key_value_pairs(self) -> None:
        result = parse_extra_args(["--model", "gpt-4", "--max-turns", "10"])
        assert result == {"model": "gpt-4", "max_turns": 10}

    def test_equals_format(self) -> None:
        result = parse_extra_args(["--model=gpt4"])
        assert result == {"model": "gpt4"}

    def test_boolean_flag(self) -> None:
        result = parse_extra_args(["--verbose"])
        assert result == {"verbose": True}

    def test_empty(self) -> None:
        assert parse_extra_args([]) == {}

    def test_mixed(self) -> None:
        result = parse_extra_args(["--flag", "--key", "val", "--eq=stuff"])
        assert result == {"flag": True, "key": "val", "eq": "stuff"}
