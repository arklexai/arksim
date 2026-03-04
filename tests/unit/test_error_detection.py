# SPDX-License-Identifier: Apache-2.0
"""Tests for error_detection: collect_agent_behavior_failure_reasoning."""

from __future__ import annotations

from arksim.evaluator.entities import ConversationEvaluation, TurnEvaluation
from arksim.evaluator.error_detection import collect_agent_behavior_failure_reasoning
from arksim.evaluator.utils.enums import EvaluationOutcomes


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def _make_turn(turn_id: int, failure_label: str, failure_reason: str) -> TurnEvaluation:
    """Build a TurnEvaluation with minimal fields."""
    return TurnEvaluation(
        turn_id=turn_id,
        scores=[],
        turn_score=-1,
        turn_behavior_failure=failure_label,
        turn_behavior_failure_reason=failure_reason,
    )


def _make_conv_eval(conversation_id: str, turns: list) -> ConversationEvaluation:
    """Build a ConversationEvaluation with minimal fields."""
    return ConversationEvaluation(
        conversation_id=conversation_id,
        goal_completion_score=0.5,
        goal_completion_reason="OK",
        turn_success_ratio=0.5,
        overall_agent_score=0.5,
        evaluation_status="Done",
        turn_scores=turns,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestCollectAgentBehaviorFailureReasoning:
    """Tests for collect_agent_behavior_failure_reasoning."""

    def test_collects_failure_turns(self) -> None:
        """Turns with a matching failure label are collected."""
        turn = _make_turn(0, "lack of specific information", "Agent missed key details")
        conv = _make_conv_eval("conv-1", [turn])
        result = collect_agent_behavior_failure_reasoning(
            [conv], ["lack of specific information"]
        )
        assert len(result) == 1
        assert "conv-1_0" in result[0]
        assert "lack of specific information" in result[0]
        assert "Agent missed key details" in result[0]

    def test_skips_no_failure(self) -> None:
        """Turns with 'no failure' label are always skipped."""
        turn = _make_turn(0, "no failure", "All good")
        conv = _make_conv_eval("conv-1", [turn])
        result = collect_agent_behavior_failure_reasoning(
            [conv], ["lack of specific information"]
        )
        assert len(result) == 0

    def test_skips_non_matching_categories(self) -> None:
        """Turns whose label is not in failure_categories are skipped."""
        turn = _make_turn(0, "repetition", "Repeated same response")
        conv = _make_conv_eval("conv-1", [turn])
        result = collect_agent_behavior_failure_reasoning(
            [conv], ["lack of specific information"]
        )
        assert len(result) == 0

    def test_empty_conversations(self) -> None:
        """Empty conversation list returns empty result."""
        result = collect_agent_behavior_failure_reasoning([], ["repetition"])
        assert result == []

    def test_empty_categories(self) -> None:
        """Empty categories list matches nothing."""
        turn = _make_turn(0, "repetition", "Repeated")
        conv = _make_conv_eval("conv-1", [turn])
        result = collect_agent_behavior_failure_reasoning([conv], [])
        assert len(result) == 0

    def test_multiple_turns_multiple_convos(self) -> None:
        """Collects from multiple turns across multiple conversations."""
        turn1 = _make_turn(0, "repetition", "Repeated response")
        turn2 = _make_turn(1, "false information", "Wrong facts provided")
        conv1 = _make_conv_eval("conv-1", [turn1, turn2])
        turn3 = _make_turn(0, "no failure", "All good")
        conv2 = _make_conv_eval("conv-2", [turn3])
        categories = ["repetition", "false information"]
        result = collect_agent_behavior_failure_reasoning([conv1, conv2], categories)
        assert len(result) == 2

    def test_skips_special_outcomes(self) -> None:
        """skipped_good_performance, evaluation_run_failed, agent_api_error are skipped."""
        skip_labels = [
            EvaluationOutcomes.SKIPPED_GOOD_PERFORMANCE.value,
            EvaluationOutcomes.EVALUATION_RUN_FAILED.value,
            EvaluationOutcomes.AGENT_API_ERROR.value,
        ]
        turns = [_make_turn(i, label, "reason") for i, label in enumerate(skip_labels)]
        conv = _make_conv_eval("conv-1", turns)
        result = collect_agent_behavior_failure_reasoning([conv], skip_labels)
        assert len(result) == 0

    def test_result_format(self) -> None:
        """Each result string encodes conv_id, turn_id, label and reason."""
        turn = _make_turn(3, "repetition", "Said the same thing twice")
        conv = _make_conv_eval("abc-xyz", [turn])
        result = collect_agent_behavior_failure_reasoning([conv], ["repetition"])
        assert len(result) == 1
        assert "abc-xyz_3" in result[0]
        assert "repetition" in result[0]
        assert "Said the same thing twice" in result[0]

    def test_only_failure_turns_collected(self) -> None:
        """Only failing turns within a conversation are collected, not successful ones."""
        turn_fail = _make_turn(0, "repetition", "Repeated")
        turn_ok = _make_turn(1, "no failure", "Fine")
        conv = _make_conv_eval("conv-1", [turn_fail, turn_ok])
        result = collect_agent_behavior_failure_reasoning([conv], ["repetition"])
        assert len(result) == 1
        assert "conv-1_0" in result[0]
