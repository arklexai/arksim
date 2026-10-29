# SPDX-License-Identifier: Apache-2.0
"""Turn judge for the Simulation Engine V2 loop.

One LLM call per turn, scoring the agent's latest reply on coverage,
responsiveness, and self-contradiction.

The call never sees the scenario's ground-truth ``knowledge``: coverage is
about whether a reply addressed a sub-goal, not whether it's factually
correct - that's deferred to the post-simulation evaluator.

A failed call skips the whole turn's ledger update; its fields are
load-bearing, so no update beats a guess.
"""

from __future__ import annotations

import logging
from typing import Any

from arksim.llms.chat import LLM
from arksim.llms.chat.base.usage import usage_tags
from arksim.scenario import Scenario
from arksim.simulation_engine.core.state import (
    ChecklistItemState,
    ConversationLedger,
    TurnJudgeVerdict,
)
from arksim.simulation_engine.utils.prompts import TURN_JUDGE_PROMPT
from arksim.simulation_engine.utils.utils import format_conversation_for_prompt

logger = logging.getLogger(__name__)


# ── Formatting helpers ──


def _format_pending(pending: list[ChecklistItemState]) -> str:
    if not pending:
        return "(none)"
    lines = []
    for i, state in enumerate(pending):
        lines.append(f"{i + 1}. {state.item.description}")
        lines.append(f"   Satisfied when: {state.item.satisfied_when}")
        lines.append(f"   Current coverage: {state.coverage}")
    return "\n".join(lines)


# ── The call ──


async def judge_turn(
    llm: LLM,
    scenario: Scenario,
    ledger: ConversationLedger,
    history: list[dict[str, Any]],
    agent_reply: str,
    turn: int,
) -> TurnJudgeVerdict | None:
    """Judge the agent's latest reply; return None if the call fails."""
    prompt = TURN_JUDGE_PROMPT.format(
        pending_checklist=_format_pending(ledger.pending_checklist()),
        conversation=format_conversation_for_prompt(history),
        agent_reply=agent_reply,
    )
    try:
        with usage_tags(component="turn_judge"):
            return await llm.call_async(
                [{"role": "system", "content": prompt}],
                schema=TurnJudgeVerdict,
            )
    except Exception as e:
        logger.warning(
            "Turn judge failed for scenario %s at turn %d (%s); "
            "skipping the ledger update for this turn",
            scenario.scenario_id,
            turn,
            e,
        )
        return None
