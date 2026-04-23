# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.evaluator.evaluate (_should_run, evaluate_conversation)."""

from __future__ import annotations

from unittest.mock import MagicMock

from arksim.evaluator.base_metric import (
    ChatMessage,
    QualitativeMetric,
    QualResult,
    QuantitativeMetric,
    QuantResult,
    ScoreInput,
)
from arksim.evaluator.entities import (
    ConvoItem,
    TurnEvaluation,
)
from arksim.evaluator.evaluate import _should_run, evaluate_conversation
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


class TestEvaluateConversation:
    def test_goal_completion_runs(self) -> None:
        llm = _mock_llm(score=5, reason="perfect")
        turns = [_turn_eval(0), _turn_eval(1)]
        result = evaluate_conversation(llm, _convo_item(turns=2), turns)
        assert result.goal_completion_score == 5
        assert result.goal_completion_reason == "perfect"
        assert result.conversation_id == "conv-1"

    def test_goal_completion_skipped(self) -> None:
        llm = _mock_llm()
        turns = [_turn_eval(0)]
        result = evaluate_conversation(
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
        result = evaluate_conversation(llm, _convo_item(turns=2), turns)
        assert result.turn_success_ratio == 0.5

    def test_status_done(self) -> None:
        llm = _mock_llm(score=4)
        convo = _convo_item(turns=1)
        turn = _turn_eval(0)
        result = evaluate_conversation(llm, convo, [turn])
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
        result = evaluate_conversation(llm, _convo_item(turns=2), turns)
        expected = (
            0.0 * TURN_SUCCESS_RATIO_SCORE_WEIGHT + 1 * GOAL_COMPLETION_SCORE_WEIGHT
        )
        assert result.overall_agent_score == expected
        assert result.evaluation_status == EvaluationStatus.FAILED.value

    def test_no_turns(self) -> None:
        llm = _mock_llm(score=3)
        result = evaluate_conversation(llm, _convo_item(turns=0), [])
        assert result.turn_success_ratio == 1.0

    def test_custom_quant_metrics_run(self) -> None:
        """Custom quantitative metrics are executed and returned in scores."""

        class StubQuant(QuantitativeMetric):
            def __init__(self) -> None:
                super().__init__(name="stub_quant", score_range=(0, 10))

            def score(self, score_input: ScoreInput) -> QuantResult:
                return QuantResult(name=self.name, value=7.0, reason="good")

        llm = _mock_llm(score=4)
        turns = [_turn_eval(0)]
        result = evaluate_conversation(
            llm,
            _convo_item(turns=1),
            turns,
            custom_metrics=[StubQuant()],
        )
        assert len(result.scores) == 1
        assert result.scores[0].name == "stub_quant"
        assert result.scores[0].value == 7.0

    def test_custom_qual_metrics_run(self) -> None:
        """Custom qualitative metrics are executed and returned in qual_scores."""

        class StubQual(QualitativeMetric):
            def __init__(self) -> None:
                super().__init__(name="stub_qual")

            def evaluate(self, score_input: ScoreInput) -> QualResult:
                return QualResult(name=self.name, value="pass", reason="ok")

        llm = _mock_llm(score=4)
        turns = [_turn_eval(0)]
        result = evaluate_conversation(
            llm,
            _convo_item(turns=1),
            turns,
            custom_qualitative_metrics=[StubQual()],
        )
        assert len(result.qual_scores) == 1
        assert result.qual_scores[0].name == "stub_qual"
        assert result.qual_scores[0].value == "pass"

    def test_custom_metric_failure_logged_not_raised(self) -> None:
        """A failing custom metric is skipped, not propagated."""

        class BrokenMetric(QuantitativeMetric):
            def __init__(self) -> None:
                super().__init__(name="broken")

            def score(self, score_input: ScoreInput) -> QuantResult:
                raise RuntimeError("boom")

        llm = _mock_llm(score=4)
        turns = [_turn_eval(0)]
        result = evaluate_conversation(
            llm,
            _convo_item(turns=1),
            turns,
            custom_metrics=[BrokenMetric()],
        )
        assert result.scores == []

    def test_no_custom_metrics_returns_empty(self) -> None:
        """Without custom metrics, scores and qual_scores are empty."""
        llm = _mock_llm(score=4)
        turns = [_turn_eval(0)]
        result = evaluate_conversation(llm, _convo_item(turns=1), turns)
        assert result.scores == []
        assert result.qual_scores == []

    def test_custom_quant_receives_additional_input(self) -> None:
        """additional_input on a quant metric is forwarded into ScoreInput."""
        captured: dict = {}

        class CapturingMetric(QuantitativeMetric):
            def __init__(self) -> None:
                super().__init__(
                    name="capturing",
                    additional_input={"threshold": 0.8},
                )

            def score(self, score_input: ScoreInput) -> QuantResult:
                captured["extra"] = score_input.model_extra
                return QuantResult(name=self.name, value=3.0, reason="ok")

        llm = _mock_llm(score=4)
        evaluate_conversation(
            llm,
            _convo_item(turns=1),
            [_turn_eval(0)],
            custom_metrics=[CapturingMetric()],
        )
        assert captured["extra"]["threshold"] == 0.8

    def test_custom_qual_receives_additional_input(self) -> None:
        """additional_input on a qual metric is forwarded into ScoreInput."""
        captured: dict = {}

        class CapturingQual(QualitativeMetric):
            def __init__(self) -> None:
                super().__init__(name="capturing_qual")
                self.additional_input = {"labels": ["a", "b"]}

            def evaluate(self, score_input: ScoreInput) -> QualResult:
                captured["extra"] = score_input.model_extra
                return QualResult(name=self.name, value="a", reason="ok")

        llm = _mock_llm(score=4)
        evaluate_conversation(
            llm,
            _convo_item(turns=1),
            [_turn_eval(0)],
            custom_qualitative_metrics=[CapturingQual()],
        )
        assert captured["extra"]["labels"] == ["a", "b"]

    def test_custom_metrics_receive_conversation_context(self) -> None:
        """Custom metrics receive chat_history, knowledge, user_goal, profile."""
        captured: dict = {}

        class InspectingMetric(QuantitativeMetric):
            def __init__(self) -> None:
                super().__init__(name="inspector")

            def score(self, score_input: ScoreInput) -> QuantResult:
                captured["user_goal"] = score_input.user_goal
                captured["profile"] = score_input.profile
                captured["knowledge"] = score_input.knowledge
                captured["history_len"] = len(score_input.chat_history)
                return QuantResult(name=self.name, value=5.0, reason="ok")

        convo = ConvoItem(
            chat_id="c1",
            chat_history=[
                ChatMessage(role="user", content="q1"),
                ChatMessage(role="assistant", content="a1"),
                ChatMessage(role="user", content="q2"),
                ChatMessage(role="assistant", content="a2"),
            ],
            system_prompt="sys",
            knowledge=["doc1", "doc2"],
            profile="admin",
            user_goal="solve the issue",
            turns=2,
        )
        llm = _mock_llm(score=4)
        evaluate_conversation(
            llm,
            convo,
            [_turn_eval(0), _turn_eval(1)],
            custom_metrics=[InspectingMetric()],
        )
        assert captured["user_goal"] == "solve the issue"
        assert captured["profile"] == "admin"
        assert captured["history_len"] == 4
        assert "doc1" in captured["knowledge"]

    def test_multiple_quant_and_qual_metrics_together(self) -> None:
        """Both quant and qual custom metrics run and populate their fields."""

        class Quant1(QuantitativeMetric):
            def __init__(self) -> None:
                super().__init__(name="q1")

            def score(self, score_input: ScoreInput) -> QuantResult:
                return QuantResult(name=self.name, value=3.0, reason="r1")

        class Quant2(QuantitativeMetric):
            def __init__(self) -> None:
                super().__init__(name="q2")

            def score(self, score_input: ScoreInput) -> QuantResult:
                return QuantResult(name=self.name, value=4.0, reason="r2")

        class Qual1(QualitativeMetric):
            def __init__(self) -> None:
                super().__init__(name="ql1")

            def evaluate(self, score_input: ScoreInput) -> QualResult:
                return QualResult(name=self.name, value="pass", reason="ok")

        llm = _mock_llm(score=4)
        result = evaluate_conversation(
            llm,
            _convo_item(turns=1),
            [_turn_eval(0)],
            custom_metrics=[Quant1(), Quant2()],
            custom_qualitative_metrics=[Qual1()],
        )
        quant_names = {s.name for s in result.scores}
        assert quant_names == {"q1", "q2"}
        assert len(result.qual_scores) == 1
        assert result.qual_scores[0].name == "ql1"

    def test_broken_qual_metric_skipped(self) -> None:
        """A failing qualitative metric is skipped, not propagated."""

        class BrokenQual(QualitativeMetric):
            def __init__(self) -> None:
                super().__init__(name="broken_qual")

            def evaluate(self, score_input: ScoreInput) -> QualResult:
                raise RuntimeError("qual boom")

        llm = _mock_llm(score=4)
        result = evaluate_conversation(
            llm,
            _convo_item(turns=1),
            [_turn_eval(0)],
            custom_qualitative_metrics=[BrokenQual()],
        )
        assert result.qual_scores == []
