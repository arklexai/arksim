# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.evaluator.evaluate.evaluate_turn."""

from __future__ import annotations

from unittest.mock import MagicMock

from arksim.evaluator.base_metric import ChatMessage
from arksim.evaluator.entities import TurnItem
from arksim.evaluator.evaluate import evaluate_turn
from arksim.evaluator.utils.enums import AgentBehaviorFailureType, EvaluationOutcomes
from arksim.evaluator.utils.schema import ConstraintViolationSchema, QualSchema, ScoreSchema


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


# ── helpers for constraint violation tests ──────────────────────────────────

CONSTRAINT = "Agent must not provide medical advice"
EXPECTED_BEHAVIOR = "Agent should decline out-of-scope requests"
CV = AgentBehaviorFailureType.CONSTRAINT_VIOLATION.value


def _constrained_turn_item(
    agent_constraints: list[str] | None = None,
    expected_behavior: str = "",
) -> TurnItem:
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
        knowledge=[],
        profile="",
        user_goal="goal",
        agent_constraints=agent_constraints or [],
        expected_behavior=expected_behavior,
    )


def _mock_llm_cv(
    score: int = 3,
    behavior_label: str = "no failure",
    cv_violated: list[str] | None = None,
    cv_fulfilled: list[str] | None = None,
) -> MagicMock:
    """Mock LLM with controllable responses for quant, qual, and constraint schemas.

    score=3 keeps all metrics at or above threshold so AgentBehaviorFailureMetric
    is skipped by default (turn_behavior_failure = SKIPPED_GOOD_PERFORMANCE).
    Use score=1 when you need the behavior failure check to fire.
    """
    llm = MagicMock()

    def _side(messages: list, schema: type | None = None, **kw: object) -> object:
        if schema is ConstraintViolationSchema:
            return ConstraintViolationSchema(
                violated_constraints=cv_violated or [],
                fulfilled_constraints=cv_fulfilled or [],
                reason="constraint result",
            )
        if schema is QualSchema:
            return QualSchema(label=behavior_label, reason="behavior result")
        return ScoreSchema(score=score, reason="ok")

    llm.call.side_effect = _side
    return llm


# ── constraint violation merging tests ──────────────────────────────────────


class TestConstraintViolationMerging:
    """Tests for severity-based merging of constraint_violation into turn_behavior_failure.

    Covered cases (evaluate.py:252-267):
    - No constraints → constraint check never runs
    - Constraints present, all fulfilled → no label change, constraints_fulfilled populated
    - Constraint violated, base label is a skip outcome → constraint_violation wins
    - Constraint violated + lower-severity existing label → constraint_violation wins
    - Constraint violated + higher-severity existing label → existing label wins, reason appended
    - expected_behavior alone (no agent_constraints) is sufficient to trigger the check
    - Both agent_constraints and expected_behavior are merged into one prompt call
    - agent_behavior_failure absent from metrics_to_run → constraint check also skipped
    """

    def test_no_constraints_skips_check(self) -> None:
        schemas_seen: list[type] = []
        llm = MagicMock()

        def _side(messages: list, schema: type | None = None, **kw: object) -> object:
            if schema is not None:
                schemas_seen.append(schema)
            if schema is QualSchema:
                return QualSchema(label="no failure", reason="fine")
            return ScoreSchema(score=3, reason="ok")

        llm.call.side_effect = _side
        result = evaluate_turn(llm, _constrained_turn_item())

        assert ConstraintViolationSchema not in schemas_seen
        assert result.constraints_fulfilled == []
        assert result.turn_behavior_failure != CV

    def test_constraint_fulfilled_no_label_change(self) -> None:
        # score=3 → no threshold failure → base label is SKIPPED_GOOD_PERFORMANCE
        llm = _mock_llm_cv(
            score=3,
            cv_violated=[],
            cv_fulfilled=[CONSTRAINT],
        )
        result = evaluate_turn(
            llm, _constrained_turn_item(agent_constraints=[CONSTRAINT])
        )

        assert result.turn_behavior_failure != CV
        assert result.constraints_fulfilled == [CONSTRAINT]

    def test_constraint_violated_beats_skip_outcome(self) -> None:
        # score=3 → base label is SKIPPED_GOOD_PERFORMANCE (a skip outcome)
        # constraint violation should take over
        llm = _mock_llm_cv(score=3, cv_violated=[CONSTRAINT])
        result = evaluate_turn(
            llm, _constrained_turn_item(agent_constraints=[CONSTRAINT])
        )

        assert result.turn_behavior_failure == CV
        assert "[Constraint]" in result.turn_behavior_failure_reason

    def test_constraint_violated_beats_no_failure_label(self) -> None:
        # score=1 → threshold fires → behavior failure check runs → returns "no failure"
        # "no failure" is a skip outcome so constraint_violation should still win
        llm = _mock_llm_cv(score=1, behavior_label="no failure", cv_violated=[CONSTRAINT])
        result = evaluate_turn(
            llm, _constrained_turn_item(agent_constraints=[CONSTRAINT])
        )

        assert result.turn_behavior_failure == CV

    def test_constraint_beats_lower_severity_label(self) -> None:
        # constraint_violation = high (rank 1), repetition = low (rank 3)
        # constraint_violation should replace repetition as the top label
        llm = _mock_llm_cv(score=1, behavior_label="repetition", cv_violated=[CONSTRAINT])
        result = evaluate_turn(
            llm, _constrained_turn_item(agent_constraints=[CONSTRAINT])
        )

        assert result.turn_behavior_failure == CV

    def test_higher_severity_label_beats_constraint(self) -> None:
        # false information = critical (rank 0), constraint_violation = high (rank 1)
        # false information keeps the top slot; constraint reason is appended
        llm = _mock_llm_cv(
            score=1, behavior_label="false information", cv_violated=[CONSTRAINT]
        )
        result = evaluate_turn(
            llm, _constrained_turn_item(agent_constraints=[CONSTRAINT])
        )

        assert result.turn_behavior_failure == "false information"
        assert "[Constraint]" in result.turn_behavior_failure_reason

    def test_equal_severity_existing_label_kept(self) -> None:
        # disobey user request = high (rank 1), constraint_violation = high (rank 1)
        # existing label wins on tie (cv_sev < agent_sev is False when equal)
        llm = _mock_llm_cv(
            score=1, behavior_label="disobey user request", cv_violated=[CONSTRAINT]
        )
        result = evaluate_turn(
            llm, _constrained_turn_item(agent_constraints=[CONSTRAINT])
        )

        assert result.turn_behavior_failure == "disobey user request"
        assert "[Constraint]" in result.turn_behavior_failure_reason

    def test_expected_behavior_alone_triggers_check(self) -> None:
        # No agent_constraints — expected_behavior from agent_response assertion is enough
        llm = _mock_llm_cv(score=3, cv_violated=[EXPECTED_BEHAVIOR])
        result = evaluate_turn(
            llm, _constrained_turn_item(expected_behavior=EXPECTED_BEHAVIOR)
        )

        assert result.turn_behavior_failure == CV

    def test_global_and_scenario_constraints_merged_in_one_call(self) -> None:
        # Both agent_constraints and expected_behavior should appear in a single
        # ConstraintViolationSchema prompt call (not two separate calls).
        cv_calls: list[list] = []
        llm = MagicMock()

        def _side(messages: list, schema: type | None = None, **kw: object) -> object:
            if schema is ConstraintViolationSchema:
                cv_calls.append(messages)
                return ConstraintViolationSchema(
                    violated_constraints=[],
                    fulfilled_constraints=[CONSTRAINT, EXPECTED_BEHAVIOR],
                    reason="ok",
                )
            if schema is QualSchema:
                return QualSchema(label="no failure", reason="fine")
            return ScoreSchema(score=3, reason="ok")

        llm.call.side_effect = _side
        result = evaluate_turn(
            llm,
            _constrained_turn_item(
                agent_constraints=[CONSTRAINT],
                expected_behavior=EXPECTED_BEHAVIOR,
            ),
        )

        assert len(cv_calls) == 1
        user_msg = next(m["content"] for m in cv_calls[0] if m["role"] == "user")
        assert CONSTRAINT in user_msg
        assert EXPECTED_BEHAVIOR in user_msg
        assert result.constraints_fulfilled == [CONSTRAINT, EXPECTED_BEHAVIOR]

    def test_constraint_check_skipped_when_abf_not_in_metrics(self) -> None:
        # agent_behavior_failure absent from metrics_to_run → constraint check also skipped
        schemas_seen: list[type] = []
        llm = MagicMock()

        def _side(messages: list, schema: type | None = None, **kw: object) -> object:
            if schema is not None:
                schemas_seen.append(schema)
            return ScoreSchema(score=1, reason="ok")

        llm.call.side_effect = _side
        result = evaluate_turn(
            llm,
            _constrained_turn_item(agent_constraints=[CONSTRAINT]),
            metrics_to_run=["helpfulness"],
        )

        assert ConstraintViolationSchema not in schemas_seen
        assert result.turn_behavior_failure != CV
