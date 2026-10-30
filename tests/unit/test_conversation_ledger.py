# SPDX-License-Identifier: Apache-2.0
"""Tests for the V2 turn-loop conversation ledger."""

from __future__ import annotations

from arksim.scenario import (
    BehaviorParams,
    GoalChecklistItem,
    KnowledgeItem,
    Scenario,
)
from arksim.simulation_engine.core.state import (
    ConversationLedger,
    SubGoalCoverage,
    TurnJudgeVerdict,
)


def _scenario(**overrides: object) -> Scenario:
    fields: dict = {
        "scenario_id": "s1",
        "user_id": "u1",
        "goal": "learn the maintenance interval",
        "agent_context": "ctx",
        "user_profile": "profile",
        "knowledge": [KnowledgeItem(id="kc_01", content="Interval is 6 months")],
        "behavior": BehaviorParams(patience=2),
        "goal_checklist": [
            GoalChecklistItem(
                description="Learn the interval",
                satisfied_when="The agent states an interval",
            ),
            GoalChecklistItem(
                description="Learn the price",
                satisfied_when="The agent states a price",
                mandatory=False,
            ),
        ],
    }
    fields.update(overrides)
    return Scenario(**fields)


class TestFromScenario:
    def test_seeds_patience_and_checklist(self) -> None:
        ledger = ConversationLedger.from_scenario(_scenario())

        assert ledger.patience_initial == 2
        assert ledger.patience_remaining == 2
        assert [s.coverage for s in ledger.checklist] == ["none", "none"]
        assert ledger.termination_reason is None
        assert ledger.blocked_reason is None

    def test_missing_behavior_defaults_patience(self) -> None:
        ledger = ConversationLedger.from_scenario(_scenario(behavior=None))
        assert ledger.patience_remaining == 3


class TestApply:
    def test_coverage_transition_by_pending_position(self) -> None:
        ledger = ConversationLedger.from_scenario(_scenario())
        verdict = TurnJudgeVerdict(
            coverage=[SubGoalCoverage(item_number=2, coverage="complete")],
            responsiveness="answered",
        )

        ledger.apply(verdict, turn=1)

        assert ledger.checklist[0].coverage == "none"
        assert ledger.checklist[1].coverage == "complete"
        assert ledger.checklist[1].coverage_updated_at_turn == 1

    def test_numbers_reference_pending_items_only(self) -> None:
        """Once item 1 is complete, pending position 1 is the old item 2."""
        ledger = ConversationLedger.from_scenario(_scenario())
        ledger.apply(
            TurnJudgeVerdict(
                coverage=[SubGoalCoverage(item_number=1, coverage="complete")],
                responsiveness="answered",
            ),
            turn=0,
        )

        ledger.apply(
            TurnJudgeVerdict(
                coverage=[SubGoalCoverage(item_number=1, coverage="partial")],
                responsiveness="answered",
            ),
            turn=1,
        )

        assert ledger.checklist[1].coverage == "partial"
        assert ledger.checklist[1].coverage_updated_at_turn == 1

    def test_coverage_is_monotonic(self) -> None:
        """A regression report (complete -> partial) never applies."""
        ledger = ConversationLedger.from_scenario(_scenario())
        ledger.apply(
            TurnJudgeVerdict(
                coverage=[SubGoalCoverage(item_number=1, coverage="complete")],
                responsiveness="answered",
            ),
            turn=0,
        )

        # Item 1 is complete and out of the pending list now; position 1 is
        # item 2. Feed a regression at position 1 anyway via a fabricated
        # verdict referencing the (now nonexistent) old position 1 slot -
        # instead, directly assert item 2 (still "none") can't be told to
        # regress below "none" (a no-op, not a real regression case), and
        # that a genuine partial->none report on item 2 is dropped.
        ledger.apply(
            TurnJudgeVerdict(
                coverage=[SubGoalCoverage(item_number=1, coverage="partial")],
                responsiveness="answered",
            ),
            turn=1,
        )
        ledger.apply(
            TurnJudgeVerdict(
                coverage=[SubGoalCoverage(item_number=1, coverage="none")],
                responsiveness="deflected",
            ),
            turn=2,
        )

        assert ledger.checklist[1].coverage == "partial"
        assert ledger.checklist[1].coverage_updated_at_turn == 1

    def test_out_of_range_numbers_are_ignored(self) -> None:
        ledger = ConversationLedger.from_scenario(_scenario())
        verdict = TurnJudgeVerdict(
            coverage=[SubGoalCoverage(item_number=5, coverage="complete")],
            responsiveness="answered",
        )

        ledger.apply(verdict, turn=0)

        assert all(s.coverage == "none" for s in ledger.checklist)

    def test_no_transition_decrements_patience(self) -> None:
        ledger = ConversationLedger.from_scenario(_scenario())

        ledger.apply(TurnJudgeVerdict(responsiveness="deflected"), turn=0)

        assert ledger.patience_remaining == 1

    def test_transition_increments_patience_capped_at_initial(self) -> None:
        ledger = ConversationLedger.from_scenario(_scenario())

        ledger.apply(
            TurnJudgeVerdict(
                coverage=[SubGoalCoverage(item_number=1, coverage="partial")],
                responsiveness="answered",
            ),
            turn=0,
        )

        assert ledger.patience_remaining == 2  # capped, was already at initial

    def test_clarification_costs_nothing(self) -> None:
        ledger = ConversationLedger.from_scenario(_scenario())

        ledger.apply(TurnJudgeVerdict(responsiveness="asked_clarification"), turn=0)

        assert ledger.patience_remaining == 2

    def test_repeat_clarification_still_costs_nothing(self) -> None:
        """Gathering information is never charged, however often it happens."""
        ledger = ConversationLedger.from_scenario(_scenario())

        for turn in range(4):
            ledger.apply(
                TurnJudgeVerdict(responsiveness="asked_clarification"), turn=turn
            )

        assert ledger.patience_remaining == 2

    def test_contradicted_clarification_still_costs_a_point(self) -> None:
        """The free clarification does not absorb the contradiction penalty."""
        ledger = ConversationLedger.from_scenario(_scenario())

        ledger.apply(
            TurnJudgeVerdict(
                responsiveness="asked_clarification",
                contradicted=True,
                contradiction_reason="contradicted its own earlier answer",
            ),
            turn=0,
        )

        assert ledger.patience_remaining == 1

    def test_contradiction_costs_an_extra_point(self) -> None:
        ledger = ConversationLedger.from_scenario(_scenario())

        ledger.apply(
            TurnJudgeVerdict(
                responsiveness="deflected",
                contradicted=True,
                contradiction_reason="claimed the user owns a Model Y",
            ),
            turn=0,
        )

        assert ledger.patience_remaining == 0

    def test_refused_turn_sets_blocked_reason_without_patience_cost(self) -> None:
        ledger = ConversationLedger.from_scenario(_scenario())

        ledger.apply(TurnJudgeVerdict(responsiveness="refused"), turn=0)

        assert ledger.blocked_reason == "refused"
        assert ledger.patience_remaining == 2

    def test_transferred_turn_sets_blocked_reason(self) -> None:
        ledger = ConversationLedger.from_scenario(_scenario())

        ledger.apply(TurnJudgeVerdict(responsiveness="transferred"), turn=0)

        assert ledger.blocked_reason == "transferred"
        assert ledger.patience_remaining == 2

    def test_unable_turn_sets_blocked_reason(self) -> None:
        ledger = ConversationLedger.from_scenario(_scenario())

        ledger.apply(TurnJudgeVerdict(responsiveness="unable"), turn=0)

        assert ledger.blocked_reason == "unable"
        assert ledger.patience_remaining == 2

    def test_patience_never_goes_negative(self) -> None:
        ledger = ConversationLedger.from_scenario(_scenario())
        for turn in range(4):
            ledger.apply(TurnJudgeVerdict(responsiveness="deflected"), turn=turn)

        assert ledger.patience_remaining == 0


class TestTurnRecord:
    def test_records_one_entry_per_apply_call(self) -> None:
        ledger = ConversationLedger.from_scenario(_scenario())

        ledger.apply(TurnJudgeVerdict(responsiveness="deflected"), turn=0)
        ledger.apply(TurnJudgeVerdict(responsiveness="answered"), turn=1)

        assert [r.turn for r in ledger.turns] == [0, 1]
        assert [r.responsiveness for r in ledger.turns] == ["deflected", "answered"]

    def test_captures_contradiction_and_patience_snapshot(self) -> None:
        ledger = ConversationLedger.from_scenario(_scenario())

        ledger.apply(
            TurnJudgeVerdict(
                responsiveness="deflected",
                contradicted=True,
                contradiction_reason="claimed the user owns a Model Y",
            ),
            turn=0,
        )

        record = ledger.turns[0]
        assert record.contradicted is True
        assert record.contradiction_reason == "claimed the user owns a Model Y"
        assert record.patience_remaining == 0

    def test_checklist_snapshot_reflects_state_as_of_that_turn(self) -> None:
        ledger = ConversationLedger.from_scenario(_scenario())
        ledger.apply(
            TurnJudgeVerdict(
                coverage=[SubGoalCoverage(item_number=1, coverage="complete")],
                responsiveness="answered",
            ),
            turn=0,
        )

        ledger.apply(
            TurnJudgeVerdict(
                coverage=[SubGoalCoverage(item_number=1, coverage="partial")],
                responsiveness="answered",
            ),
            turn=1,
        )

        first, second = ledger.turns
        assert [s.coverage for s in first.checklist] == ["complete", "none"]
        assert [s.coverage for s in second.checklist] == ["complete", "partial"]

    def test_earlier_snapshot_is_unaffected_by_later_mutations(self) -> None:
        """The deep-copy guarantee: turn 0's snapshot must not silently
        change once turn 1 mutates the live ledger's checklist."""
        ledger = ConversationLedger.from_scenario(_scenario())

        ledger.apply(TurnJudgeVerdict(responsiveness="deflected"), turn=0)
        first_snapshot = ledger.turns[0]
        assert first_snapshot.checklist[1].coverage == "none"

        ledger.apply(
            TurnJudgeVerdict(
                coverage=[SubGoalCoverage(item_number=2, coverage="partial")],
                responsiveness="asked_clarification",
            ),
            turn=1,
        )

        # The object retrieved before turn 1 must still show turn 0's state.
        assert first_snapshot.checklist[1].coverage == "none"
        # ...while the live ledger and the new snapshot reflect the change.
        assert ledger.checklist[1].coverage == "partial"
        assert ledger.turns[1].checklist[1].coverage == "partial"

    def test_blocked_turn_records_responsiveness_and_unchanged_patience(self) -> None:
        """No separate blocked_reason on TurnRecord - responsiveness IS it."""
        ledger = ConversationLedger.from_scenario(_scenario())

        ledger.apply(TurnJudgeVerdict(responsiveness="refused"), turn=0)

        record = ledger.turns[0]
        assert record.responsiveness == "refused"
        assert record.patience_remaining == 2


class TestShouldTerminate:
    def test_running_conversation_continues(self) -> None:
        ledger = ConversationLedger.from_scenario(_scenario())
        assert ledger.should_terminate() is None

    def test_all_mandatory_complete_is_user_satisfied(self) -> None:
        """The non-mandatory item stays pending; only mandatory items gate."""
        ledger = ConversationLedger.from_scenario(_scenario())
        ledger.apply(
            TurnJudgeVerdict(
                coverage=[SubGoalCoverage(item_number=1, coverage="complete")],
                responsiveness="answered",
            ),
            turn=0,
        )

        assert ledger.should_terminate() == "user_satisfied"

    def test_zero_patience_is_gave_up(self) -> None:
        ledger = ConversationLedger.from_scenario(_scenario())
        ledger.apply(TurnJudgeVerdict(responsiveness="deflected"), turn=0)
        ledger.apply(TurnJudgeVerdict(responsiveness="deflected"), turn=1)

        assert ledger.should_terminate() == "gave_up"

    def test_blocked_reason_set_is_blocked(self) -> None:
        ledger = ConversationLedger.from_scenario(_scenario())
        ledger.apply(TurnJudgeVerdict(responsiveness="refused"), turn=0)

        assert ledger.should_terminate() == "blocked"

    def test_user_satisfied_wins_over_blocked(self) -> None:
        """A completed transfer that fulfills the need counts toward coverage."""
        ledger = ConversationLedger.from_scenario(_scenario())
        ledger.apply(
            TurnJudgeVerdict(
                coverage=[SubGoalCoverage(item_number=1, coverage="complete")],
                responsiveness="transferred",
            ),
            turn=0,
        )

        assert ledger.should_terminate() == "user_satisfied"

    def test_user_satisfied_wins_over_gave_up(self) -> None:
        ledger = ConversationLedger.from_scenario(_scenario())
        ledger.patience_remaining = 0
        ledger.apply(
            TurnJudgeVerdict(
                coverage=[SubGoalCoverage(item_number=1, coverage="complete")],
                responsiveness="answered",
            ),
            turn=0,
        )

        assert ledger.should_terminate() == "user_satisfied"
