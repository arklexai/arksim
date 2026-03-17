# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.evaluator.evaluate.evaluate_turn."""

from __future__ import annotations

from unittest.mock import MagicMock

from arksim.evaluator.base_metric import ChatMessage
from arksim.evaluator.entities import TurnItem
from arksim.evaluator.evaluate import evaluate_turn
from arksim.evaluator.utils.enums import EvaluationOutcomes
from arksim.evaluator.utils.schema import QualSchema, ScoreSchema


def _mock_llm(score: int = 4) -> MagicMock:
    """Mock LLM that returns ScoreSchema for quant calls and QualSchema for qual calls."""
    llm = MagicMock()

    def _side_effect(
        messages: list, schema: type | None = None, **kw: object
    ) -> object:
        if schema is QualSchema:
            return QualSchema(label="no failure", reason="fine")
        return ScoreSchema(score=score, reason="ok")

    llm.call.side_effect = _side_effect
    return llm


def _turn_item() -> TurnItem:
    msgs = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="hello"),
    ]
    return TurnItem(
        chat_id="c1",
        turn_id=0,
        current_turn=msgs,
        conversation_history=msgs,
        system_prompt="sys",
        knowledge=["k1"],
        profile="profile",
        user_goal="goal",
    )


class TestEvaluateTurn:
    def test_all_builtins_run(self) -> None:
        llm = _mock_llm(score=4)
        result = evaluate_turn(llm, _turn_item())
        assert result.turn_id == 0
        assert len(result.scores) == 5
        assert result.turn_score > 0
        names = {s.name for s in result.scores}
        assert "helpfulness" in names
        assert "verbosity" in names

    def test_metrics_to_run_filters(self) -> None:
        llm = _mock_llm(score=4)
        result = evaluate_turn(
            llm, _turn_item(), metrics_to_run=["helpfulness", "coherence"]
        )
        names = {s.name for s in result.scores}
        assert names == {"helpfulness", "coherence"}

    def test_good_scores_skip_behavior_failure(self) -> None:
        # Score=3 means verbosity inverts to 3 (6-3), all scores >= 3 = 0.6*5
        llm = _mock_llm(score=3)
        result = evaluate_turn(llm, _turn_item())
        assert (
            result.turn_behavior_failure
            == EvaluationOutcomes.SKIPPED_GOOD_PERFORMANCE.value
        )

    def test_low_scores_trigger_behavior_failure(self) -> None:
        llm = MagicMock()
        # Return low score for quant metrics, then QualSchema for behavior failure
        llm.call.side_effect = [
            ScoreSchema(score=1, reason="bad"),  # helpfulness
            ScoreSchema(score=1, reason="bad"),  # coherence
            ScoreSchema(score=1, reason="bad"),  # verbosity
            ScoreSchema(score=1, reason="bad"),  # relevance
            ScoreSchema(score=1, reason="bad"),  # faithfulness
            QualSchema(label="repetition", reason="repeated"),  # behavior failure
        ]
        result = evaluate_turn(llm, _turn_item())
        assert result.turn_behavior_failure == "repetition"

    def test_no_metrics(self) -> None:
        llm = _mock_llm()
        result = evaluate_turn(llm, _turn_item(), metrics_to_run=["nonexistent"])
        assert result.scores == []
        assert result.turn_score == -1

    def test_num_workers_limits_concurrency(self) -> None:
        llm = _mock_llm(score=4)
        result = evaluate_turn(llm, _turn_item(), num_workers=2)
        assert len(result.scores) == 5

    def test_metrics_use_full_conversation_history(self) -> None:
        """Metrics should receive the full conversation history, not just the current turn."""
        history = [
            ChatMessage(role="user", content="previous question"),
            ChatMessage(role="assistant", content="previous answer"),
            ChatMessage(role="user", content="follow-up question"),
            ChatMessage(role="assistant", content="follow-up answer"),
        ]
        current_turn = [
            ChatMessage(role="user", content="follow-up question"),
            ChatMessage(role="assistant", content="follow-up answer"),
        ]
        turn_item = TurnItem(
            chat_id="c1",
            turn_id=1,
            current_turn=current_turn,
            conversation_history=history,
            system_prompt="sys",
            knowledge=["k1"],
            profile="profile",
            user_goal="goal",
        )

        llm = _mock_llm(score=4)
        evaluate_turn(llm, turn_item, metrics_to_run=["helpfulness"])

        call_args = llm.call.call_args_list[0]
        messages_sent = call_args[0][0]
        user_prompt = next(m["content"] for m in messages_sent if m["role"] == "user")

        assert "previous question" in user_prompt
        assert "previous answer" in user_prompt
