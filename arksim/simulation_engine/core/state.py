# SPDX-License-Identifier: Apache-2.0
"""Per-conversation state for the Simulation Engine V2 turn loop.

A ``ConversationLedger`` is created per conversation (never stored on the
shared ``Simulator``, which runs many conversations concurrently) and
threaded through the turn loop. The turn judge never sees the scenario's
ground-truth ``knowledge`` - only whether a reply addresses a sub-goal
(``coverage``), never whether it's factually correct; correctness and
agent-claim classification are deferred to the post-simulation evaluator.

``apply`` folds one turn's judgment into the ledger; ``should_terminate``
implements code-side termination, so ending a conversation never depends
on the LLM.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel

from arksim.scenario import (
    BehaviorParams,
    GoalChecklistItem,
    Scenario,
)

logger = logging.getLogger(__name__)

TerminationReason = Literal["user_satisfied", "gave_up", "blocked", "max_turns"]

CoverageLevel = Literal["none", "partial", "complete"]
_COVERAGE_ORDER: dict[CoverageLevel, int] = {"none": 0, "partial": 1, "complete": 2}

Responsiveness = Literal[
    "answered",
    "asked_clarification",
    "deflected",
    "boilerplate",
    "refused",
    "unable",
    "transferred",
]
# These three responsiveness values terminate a conversation as `blocked`,
# recorded verbatim as ConversationLedger.blocked_reason.
_BLOCKING_RESPONSIVENESS: frozenset[Responsiveness] = frozenset(
    {"refused", "unable", "transferred"}
)


# ── Turn judge output schema ──


class SubGoalCoverage(BaseModel):
    """Judge's coverage call for one pending checklist item."""

    # 1-based position in the pending-items list shown to the judge.
    item_number: int
    coverage: CoverageLevel = "none"


class TurnJudgeVerdict(BaseModel):
    """Structured output of one turn-judge call."""

    coverage: list[SubGoalCoverage] = []
    responsiveness: Responsiveness = "deflected"
    contradicted: bool = False
    contradiction_reason: str = ""


# ── Ledger ──


class ChecklistItemState(BaseModel):
    """Judge-tracked coverage of one goal checklist item."""

    item: GoalChecklistItem
    coverage: CoverageLevel = "none"
    coverage_updated_at_turn: int | None = None


class TurnRecord(BaseModel):
    """Full snapshot of ledger state after folding in one turn's judgment.

    A complete copy, not a diff - trivial to read at any turn without
    replaying history. ``checklist`` is deep-copied: the live ledger keeps
    mutating the same objects, so a shallow reference would let a later
    turn retroactively change what an earlier record shows. ``turn`` is
    explicit, not list position, since a failed judge call skips ``apply``
    (and this append) entirely, leaving gaps.

    No ``blocked_reason`` field: a blocking verdict always ends the
    conversation on the turn it fires, so it always equals this same
    record's ``responsiveness`` - nothing to add by repeating it.
    """

    turn: int
    responsiveness: Responsiveness
    contradicted: bool = False
    contradiction_reason: str = ""
    checklist: list[ChecklistItemState] = []
    patience_remaining: int


class ConversationLedger(BaseModel):
    """Mutable per-conversation judge state, persisted with the output."""

    checklist: list[ChecklistItemState] = []
    patience_initial: int = BehaviorParams().patience
    patience_remaining: int = BehaviorParams().patience
    termination_reason: TerminationReason | None = None
    blocked_reason: Responsiveness | None = None
    turns: list[TurnRecord] = []

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> ConversationLedger:
        """Seed a ledger from a compiled scenario's behavior fields."""
        patience = (
            scenario.behavior.patience
            if scenario.behavior is not None
            else BehaviorParams().patience
        )
        return cls(
            checklist=[
                ChecklistItemState(item=item) for item in scenario.goal_checklist
            ],
            patience_initial=patience,
            patience_remaining=patience,
        )

    def pending_checklist(self) -> list[ChecklistItemState]:
        return [state for state in self.checklist if state.coverage != "complete"]

    def apply(self, verdict: TurnJudgeVerdict, turn: int) -> None:
        """Fold one turn's judgment into the ledger.

        Item numbers reference the pending-checklist list as it was when the
        judge was called, i.e. before this update. Out-of-range numbers are
        ignored with a warning. Also appends a ``TurnRecord`` to ``turns`` -
        see that class for what is and isn't captured.
        """
        pending = self.pending_checklist()
        transitioned = False
        for update in verdict.coverage:
            if not 1 <= update.item_number <= len(pending):
                logger.warning(
                    "Turn judge coverage item_number %d out of range (1..%d)",
                    update.item_number,
                    len(pending),
                )
                continue
            state = pending[update.item_number - 1]
            if _COVERAGE_ORDER[update.coverage] > _COVERAGE_ORDER[state.coverage]:
                state.coverage = update.coverage
                state.coverage_updated_at_turn = turn
                transitioned = True

        if verdict.responsiveness in _BLOCKING_RESPONSIVENESS:
            # No patience change: the termination policy handles this turn
            # as `blocked`, not as an unproductive turn.
            self.blocked_reason = verdict.responsiveness
        else:
            if transitioned:
                delta = 1
            elif verdict.responsiveness == "asked_clarification":
                # Asking the customer for information the agent needs is
                # legitimate work, not an unproductive turn, so it costs
                # nothing however many times it happens.
                delta = 0
            else:
                delta = -1
            if verdict.contradicted:
                delta -= 1
            self.patience_remaining = max(
                0, min(self.patience_initial, self.patience_remaining + delta)
            )

        self.turns.append(
            TurnRecord(
                turn=turn,
                responsiveness=verdict.responsiveness,
                contradicted=verdict.contradicted,
                contradiction_reason=verdict.contradiction_reason,
                checklist=[item.model_copy(deep=True) for item in self.checklist],
                patience_remaining=self.patience_remaining,
            )
        )

    def should_terminate(self) -> TerminationReason | None:
        """Code-side termination; no LLM decides to stop.

        Checked in this order so a final reply that completes every
        mandatory sub-goal ends `user_satisfied` even if patience would
        also reach 0, or the same reply came from a completed transfer.
        """
        mandatory = [state for state in self.checklist if state.item.mandatory]
        if mandatory and all(state.coverage == "complete" for state in mandatory):
            return "user_satisfied"
        if self.blocked_reason is not None:
            return "blocked"
        if self.patience_remaining <= 0:
            return "gave_up"
        return None
