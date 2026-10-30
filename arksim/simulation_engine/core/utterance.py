# SPDX-License-Identifier: Apache-2.0
"""Utterance generation for the Simulation Engine V2 turn loop.

Two sequential steps (unlike the turn judge's parallel calls, these
depend on each other): draft, then check-and-rewrite.

1. Draft: conditioned on a per-turn directive (pending checklist
   description + mandatory flag, current patience) - never ground-truth
   knowledge, satisfied_when, or a ###STOP### token, since termination
   is code-only.
2. Check-and-rewrite: validates the draft against a fixed rubric,
   rewriting it if needed.

The directive is appended as an ephemeral message ahead of the real
history, never persisted - the same technique the old per-turn knowledge
injection used - so the system prompt stays fixed for the whole
conversation. The simulated user has no case facts of its own beyond its
profile, so it declines rather than inventing details when asked for
something it does not have.

A failed check-and-rewrite ships the raw draft unmodified (non-fatal,
matching the turn judge's pattern). A failed draft has no fallback and
propagates, aborting the conversation like any other unhandled failure.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from arksim.llms.chat import LLM
from arksim.llms.chat.base.usage import usage_tags
from arksim.scenario import Scenario
from arksim.simulation_engine.core.state import ChecklistItemState, ConversationLedger
from arksim.simulation_engine.utils.prompts import (
    CHECK_AND_REWRITE_PROMPT,
    UTTERANCE_DIRECTIVE_TEMPLATE,
)
from arksim.simulation_engine.utils.utils import format_conversation_for_prompt

logger = logging.getLogger(__name__)


class CheckedUtterance(BaseModel):
    """Structured output of the check-and-rewrite call."""

    utterance: str


def _format_pending_for_generator(pending: list[ChecklistItemState]) -> str:
    if not pending:
        return "(none)"
    return "\n".join(
        f"- ({'mandatory' if state.item.mandatory else 'optional'}) {state.item.description}"
        for state in pending
    )


def _format_settled(settled: list[ChecklistItemState]) -> str:
    """Finished wants, kept in view: a code or fact written on one is still held."""
    if not settled:
        return ""
    lines = "\n".join(f"- {state.item.description}" for state in settled)
    return (
        "\nAlready settled (do not ask for these again; what they say you hold "
        f"or know is still true):\n{lines}\n"
    )


def _render_directive(ledger: ConversationLedger) -> str:
    return UTTERANCE_DIRECTIVE_TEMPLATE.format(
        patience_remaining=ledger.patience_remaining,
        patience_initial=ledger.patience_initial,
        pending_checklist=_format_pending_for_generator(ledger.pending_checklist()),
        settled=_format_settled(
            [state for state in ledger.checklist if state.coverage == "complete"]
        ),
    )


async def _generate_draft(
    llm: LLM,
    ledger: ConversationLedger,
    history: list[dict[str, Any]],
    web_search: bool,
) -> str:
    directive = _render_directive(ledger)
    augmented = [*history, {"role": "user", "content": directive}]
    return await llm.call_async(augmented, web_search=web_search)


async def _check_and_rewrite(
    llm: LLM,
    scenario: Scenario,
    history: list[dict[str, Any]],
    draft: str,
    turn: int,
) -> str:
    prompt = CHECK_AND_REWRITE_PROMPT.format(
        conversation=format_conversation_for_prompt(history),
        draft=draft,
    )
    try:
        with usage_tags(component="check_and_rewrite"):
            result = await llm.call_async(
                [{"role": "system", "content": prompt}],
                schema=CheckedUtterance,
            )
        return result.utterance
    except Exception as e:
        logger.warning(
            "Check-and-rewrite failed for scenario %s at turn %d (%s); "
            "sending the unchecked draft",
            scenario.scenario_id,
            turn,
            e,
        )
        return draft


async def generate_utterance(
    llm: LLM,
    scenario: Scenario,
    ledger: ConversationLedger,
    history: list[dict[str, Any]],
    turn: int,
    web_search: bool = False,
) -> str:
    """Generate the simulated user's next message: draft, then check & rewrite."""
    draft = await _generate_draft(llm, ledger, history, web_search)
    return await _check_and_rewrite(llm, scenario, history, draft, turn)
