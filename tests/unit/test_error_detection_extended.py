# SPDX-License-Identifier: Apache-2.0
"""Extended tests for arksim.evaluator.error_detection (detect_agent_error, UniqueErrors)."""

from __future__ import annotations

from unittest.mock import MagicMock

from arksim.evaluator.base_metric import QuantResult
from arksim.evaluator.entities import ConversationEvaluation, TurnEvaluation
from arksim.evaluator.error_detection import UniqueErrors, detect_agent_error
from arksim.evaluator.utils.schema import UniqueErrorSchema, UniqueErrorsSchema


def _turn(
    turn_id: int, failure: str = "no failure", reason: str = "ok"
) -> TurnEvaluation:
    return TurnEvaluation(
        turn_id=turn_id,
        scores=[QuantResult(name="helpfulness", value=4)],
        turn_score=4.0,
        turn_behavior_failure=failure,
        turn_behavior_failure_reason=reason,
    )


def _conv(conv_id: str, turns: list[TurnEvaluation]) -> ConversationEvaluation:
    return ConversationEvaluation(
        conversation_id=conv_id,
        user_goal_completion_score=0.5,
        user_goal_completion_reason="ok",
        turn_success_ratio=0.5,
        overall_agent_score=0.5,
        evaluation_status="Done",
        turn_scores=turns,
    )


class TestDetectAgentError:
    def test_no_failures_returns_empty(self) -> None:
        llm = MagicMock()
        conv = _conv("c1", [_turn(0)])
        result = detect_agent_error(llm, [conv])
        assert result == []

    def test_with_failures_calls_llm(self) -> None:
        llm = MagicMock()
        llm.call.return_value = UniqueErrorsSchema(
            unique_errors=[
                UniqueErrorSchema(
                    agent_behavior_failure_category="repetition",
                    unique_error_description="Agent repeats response",
                    occurrences=["c1_0"],
                )
            ]
        )
        conv = _conv("c1", [_turn(0, failure="repetition", reason="repeated")])
        result = detect_agent_error(llm, [conv])
        assert len(result) == 1
        assert result[0].behavior_failure_category == "repetition"
        assert result[0].severity == "low"
        assert len(result[0].occurrences) == 1
        assert result[0].occurrences[0].conversation_id == "c1"
        assert result[0].occurrences[0].turn_id == 0

    def test_bad_occurrence_format_skipped(self) -> None:
        llm = MagicMock()
        llm.call.return_value = UniqueErrorsSchema(
            unique_errors=[
                UniqueErrorSchema(
                    agent_behavior_failure_category="false information",
                    unique_error_description="wrong facts",
                    occurrences=["malformed", "c1_0", "c1_notanint"],
                )
            ]
        )
        conv = _conv("c1", [_turn(0, failure="false information", reason="wrong")])
        result = detect_agent_error(llm, [conv])
        assert len(result) == 1
        assert len(result[0].occurrences) == 1

    def test_exception_returns_empty(self) -> None:
        llm = MagicMock()
        llm.call.side_effect = RuntimeError("LLM down")
        conv = _conv("c1", [_turn(0, failure="repetition", reason="r")])
        result = detect_agent_error(llm, [conv])
        assert result == []


class TestUniqueErrors:
    def test_empty_returns_empty(self) -> None:
        llm = MagicMock()
        ue = UniqueErrors(llm)
        assert ue.evaluate([]) == []

    def test_calls_llm(self) -> None:
        llm = MagicMock()
        llm.call.return_value = UniqueErrorsSchema(
            unique_errors=[
                UniqueErrorSchema(
                    agent_behavior_failure_category="repetition",
                    unique_error_description="repeats",
                    occurrences=["c1_0"],
                )
            ]
        )
        ue = UniqueErrors(llm)
        result = ue.evaluate(["Item c1_0: repetition: repeated response"])
        assert len(result) == 1
        llm.call.assert_called_once()
