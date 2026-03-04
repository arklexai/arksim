# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.evaluator.evaluate (_should_run, evaluate_goal_completion)."""

from __future__ import annotations

from unittest.mock import MagicMock

from arksim.evaluator.base_metric import ChatMessage, QuantResult
from arksim.evaluator.entities import (
    ConvoItem,
    TurnEvaluation,
)
from arksim.evaluator.evaluate import _should_run, evaluate_goal_completion
from arksim.evaluator.utils.constants import (
    EVALUATION_PARTIAL_FAILURE_THRESHOLD,
    GOAL_COMPLETION_SCORE_WEIGHT,
    TURN_SUCCESS_RATIO_SCORE_WEIGHT,
)
from arksim.evaluator.utils.enums import EvaluationOutcomes, EvaluationStatus
from arksim.evaluator.utils.schema import ScoreSchema


class TestShouldRun:
    def test_none_always_true(self) -> None:
        assert _should_run("anything", None) is True

    def test_matching_name(self) -> None:
        assert _should_run("helpfulness", ["helpfulness", "coherence"]) is True

    def test_non_matching_name(self) -> None:
        assert _should_run("verbosity", ["helpfulness"]) is False

    def test_empty_list_runs_all(self) -> None:
        assert _should_run("helpfulness", []) is True


def _mock_llm(score: int = 4, reason: str = "ok") -> MagicMock:
    llm = MagicMock()
    llm.call.return_value = ScoreSchema(score=score, reason=reason)
    return llm


def _convo_item(turns: int = 2) -> ConvoItem:
    return ConvoItem(
        chat_id="conv-1",
        chat_history=[
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="hello"),
        ],
        system_prompt="sys",
        knowledge=["k1"],
        profile="",
        user_goal="help the user",
        turns=turns,
    )


def _turn_eval(
    turn_id: int = 0,
    failure: str = EvaluationOutcomes.SKIPPED_GOOD_PERFORMANCE.value,
) -> TurnEvaluation:
    return TurnEvaluation(
        turn_id=turn_id,
        scores=[QuantResult(name="helpfulness", value=4)],
        turn_score=4.0,
        turn_behavior_failure=failure,
        turn_behavior_failure_reason="reason",
    )


class TestEvaluateGoalCompletion:
    def test_goal_completion_runs(self) -> None:
        llm = _mock_llm(score=5, reason="perfect")
        turns = [_turn_eval(0), _turn_eval(1)]
        result = evaluate_goal_completion(llm, _convo_item(turns=2), turns)
        assert result.goal_completion_score == 5
        assert result.goal_completion_reason == "perfect"
        assert result.conversation_id == "conv-1"

    def test_goal_completion_skipped(self) -> None:
        llm = _mock_llm()
        turns = [_turn_eval(0)]
        result = evaluate_goal_completion(
            llm, _convo_item(turns=1), turns, metrics_to_run=["helpfulness"]
        )
        assert result.goal_completion_score == -1
        assert result.overall_agent_score == result.turn_success_ratio

    def test_turn_success_ratio_with_failures(self) -> None:
        llm = _mock_llm(score=3)
        turns = [
            _turn_eval(0),
            _turn_eval(1, failure="repetition"),
        ]
        result = evaluate_goal_completion(llm, _convo_item(turns=2), turns)
        assert result.turn_success_ratio == 0.5

    def test_status_done(self) -> None:
        llm = _mock_llm(score=4)
        convo = _convo_item(turns=1)
        turn = _turn_eval(0)
        result = evaluate_goal_completion(llm, convo, [turn])
        expected_score = (
            1.0 * TURN_SUCCESS_RATIO_SCORE_WEIGHT + 4 * GOAL_COMPLETION_SCORE_WEIGHT
        )
        assert result.overall_agent_score == expected_score
        if expected_score == 1.0:
            assert result.evaluation_status == EvaluationStatus.DONE.value
        elif expected_score >= EVALUATION_PARTIAL_FAILURE_THRESHOLD:
            assert result.evaluation_status == EvaluationStatus.PARTIAL_FAILURE.value

    def test_status_failed(self) -> None:
        llm = _mock_llm(score=1)
        turns = [
            _turn_eval(0, failure="false information"),
            _turn_eval(1, failure="repetition"),
        ]
        result = evaluate_goal_completion(llm, _convo_item(turns=2), turns)
        expected = (
            0.0 * TURN_SUCCESS_RATIO_SCORE_WEIGHT + 1 * GOAL_COMPLETION_SCORE_WEIGHT
        )
        assert result.overall_agent_score == expected
        assert result.evaluation_status == EvaluationStatus.FAILED.value

    def test_no_turns(self) -> None:
        llm = _mock_llm(score=3)
        result = evaluate_goal_completion(llm, _convo_item(turns=0), [])
        assert result.turn_success_ratio == 1.0
