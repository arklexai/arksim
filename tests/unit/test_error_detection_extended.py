# SPDX-License-Identifier: Apache-2.0
"""Extended tests for arksim.evaluator.error_detection (detect_agent_error, UniqueErrors)."""

from __future__ import annotations

from unittest.mock import MagicMock

from arksim.evaluator.assertions import AssertionResult
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
        goal_completion_score=0.5,
        goal_completion_reason="ok",
        turn_success_ratio=0.5,
        overall_agent_score=0.5,
        evaluation_status="Done",
        turn_scores=turns,
    )


def _failed(label: str, reason: str) -> AssertionResult:
    return AssertionResult(type="llm_judge", label=label, status="fail", reason=reason)


class TestAssertionFailures:
    def test_failed_assertions_become_errors_without_the_metric(self) -> None:
        llm = MagicMock()
        llm.call.return_value = UniqueErrorsSchema(
            unique_errors=[
                UniqueErrorSchema(
                    agent_behavior_failure_category="assertion failed",
                    unique_error_description="Refunded without looking up the order",
                    occurrences=["c1"],
                )
            ]
        )
        conv = _conv("c1", [_turn(0)])
        conv.assertion_results = [
            _failed("Looks up the order first", "Refunded straight away."),
            AssertionResult(
                type="llm_judge", label="Stays polite", status="inconclusive"
            ),
        ]

        result = detect_agent_error(llm, [conv])

        prompt = llm.call.call_args[0][0][0]["content"]
        assert (
            "Item c1: assertion failed: Looks up the order first: "
            "Refunded straight away." in prompt
        )
        assert "Stays polite" not in prompt
        assert result[0].behavior_failure_category == "assertion failed"
        assert result[0].source == "assertion"
        assert result[0].severity == "high"
        assert result[0].occurrences[0].conversation_id == "c1"
        assert result[0].occurrences[0].turn_id is None

    def test_assertion_errors_come_before_behavior_errors(self) -> None:
        llm = MagicMock()
        llm.call.return_value = UniqueErrorsSchema(
            unique_errors=[
                UniqueErrorSchema(
                    agent_behavior_failure_category="false information",
                    unique_error_description="Invented a delivery date",
                    occurrences=["c1_0"],
                ),
                UniqueErrorSchema(
                    agent_behavior_failure_category="Assertion failed",
                    unique_error_description="Refunded without looking up the order",
                    occurrences=["c1", "c1_1"],
                ),
            ]
        )
        conv = _conv(
            "c1",
            [
                _turn(0, failure="false information", reason="made up"),
                _turn(1, failure="unsafe action", reason="refunded"),
            ],
        )
        conv.assertion_results = [_failed("Looks up the order first", "No lookup.")]

        result = detect_agent_error(llm, [conv])

        assert [e.behavior_failure_category for e in result] == [
            "assertion failed",
            "false information",
        ]
        # A merged behavior failure keeps its turn.
        assert [o.turn_id for o in result[0].occurrences] == [None, 1]


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


def test_collect_metric_failures_low_scores_and_labels() -> None:
    from arksim.evaluator.base_metric import QualResult
    from arksim.evaluator.error_detection import collect_metric_failures

    turn = TurnEvaluation(
        turn_id=0,
        scores=[
            QuantResult(name="helpfulness", value=2, reason="vague"),
            QuantResult(name="relevance", value=5, reason="on topic"),
            # Not below 0.6 of the max, so not a behavior failure either.
            QuantResult(name="coherence", value=3, reason="fine"),
            QuantResult(name="custom", value=4, reason="weak", metadata={}),
        ],
        turn_score=3.0,
        turn_behavior_failure="false information",
        turn_behavior_failure_reason="[Tool call] made up a date",
        qual_scores=[
            QualResult(name="Agent Behavior Failure", value="false information"),
            QualResult(
                name="Tool call behavior",
                value="false information",
                reason="made up a date",
            ),
            QualResult(name="Tone", value="no failure"),
        ],
    )
    conv = _conv("c1", [turn]).model_copy(
        update={
            "goal_completion_score": 0.0,
            "goal_completion_reason": "not done",
            "convo_qual_scores": [
                QualResult(name="Outcome", value="unresolved", reason="left")
            ],
        }
    )

    items = collect_metric_failures([conv], {"custom": (0, 10)})

    # Lowest relative score first; labels after every score.
    assert items == [
        "Item c1: goal_completion scored 0 of 1: not done",
        "Item c1_0: helpfulness scored 2 of 5: vague",
        "Item c1_0: custom scored 4 of 10: weak",
        "Item c1: Outcome label unresolved: left",
    ]


def test_detect_agent_error_tags_metric_patterns() -> None:
    llm = MagicMock()
    llm.call.return_value = UniqueErrorsSchema(
        unique_errors=[
            UniqueErrorSchema(
                agent_behavior_failure_category="helpfulness",
                unique_error_description="Answered without the order details",
                occurrences=["c1_0"],
            ),
            UniqueErrorSchema(
                agent_behavior_failure_category="false information",
                unique_error_description="Quoted a made-up policy",
                occurrences=["c1_0"],
            ),
        ]
    )
    turn = TurnEvaluation(
        turn_id=0,
        scores=[QuantResult(name="Helpfulness", value=1, reason="no details")],
        turn_score=1.0,
        turn_behavior_failure="no failure",
        turn_behavior_failure_reason="ok",
    )

    errors = detect_agent_error(llm, [_conv("c1", [turn])])

    assert (
        "Helpfulness scored 1 of 5: no details"
        in llm.call.call_args[0][0][0]["content"]
    )
    # The category comes back as the metric is named, with its source.
    assert errors[0].behavior_failure_category == "Helpfulness"
    assert errors[0].source == "metric"
    assert errors[1].source == "behavior"
    assert errors[0].severity == "medium"
    assert errors[0].occurrences[0].turn_id == 0
