# SPDX-License-Identifier: Apache-2.0
"""Tests for utterance generation: the per-turn directive, the ephemeral
injection, and the check-and-rewrite pass.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from arksim.scenario import (
    BehaviorParams,
    GoalChecklistItem,
    KnowledgeItem,
    Scenario,
)
from arksim.simulation_engine.core.state import ConversationLedger
from arksim.simulation_engine.core.utterance import (
    CheckedUtterance,
    generate_utterance,
)

_HISTORY: list[dict[str, Any]] = [
    {"role": "system", "content": "fixed system prompt"},
    {"role": "assistant", "content": "How often should I service it?"},
    {"role": "user", "content": "Every 6 months."},
]


def _scenario() -> Scenario:
    return Scenario(
        scenario_id="s1",
        user_id="u1",
        goal="learn the maintenance interval",
        agent_context="ctx",
        user_profile="profile",
        knowledge=[KnowledgeItem(id="kc_01", content="Interval is 6 months")],
        behavior=BehaviorParams(patience=2),
        goal_checklist=[
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
    )


def _mock_llm(
    draft: str | Exception, checked: CheckedUtterance | Exception
) -> MagicMock:
    """A schema-aware LLM stub: draft calls have no schema, check calls do."""

    async def fake_call_async(
        messages: list[dict[str, Any]], schema: type | None = None, **kwargs: object
    ) -> str | CheckedUtterance:
        if schema is None:
            if isinstance(draft, Exception):
                raise draft
            return draft
        if schema is CheckedUtterance:
            if isinstance(checked, Exception):
                raise checked
            return checked
        raise AssertionError(f"unexpected schema {schema}")

    llm = MagicMock()
    llm.call_async = AsyncMock(side_effect=fake_call_async)
    return llm


@pytest.mark.asyncio
async def test_directive_never_sees_knowledge_or_satisfied_when_or_triggers() -> None:
    llm = _mock_llm("draft", CheckedUtterance(utterance="draft"))
    scenario = _scenario()
    ledger = ConversationLedger.from_scenario(scenario)

    await generate_utterance(llm, scenario, ledger, _HISTORY, turn=0)

    draft_call = next(
        c for c in llm.call_async.call_args_list if c.kwargs.get("schema") is None
    )
    directive = draft_call.args[0][-1]["content"]
    assert "Interval is 6 months" not in directive
    assert "kc_01" not in directive
    assert "The agent states an interval" not in directive
    assert "###STOP###" not in directive


@pytest.mark.asyncio
async def test_directive_shows_pending_items_with_mandatory_flag() -> None:
    llm = _mock_llm("draft", CheckedUtterance(utterance="draft"))
    scenario = _scenario()
    ledger = ConversationLedger.from_scenario(scenario)

    await generate_utterance(llm, scenario, ledger, _HISTORY, turn=0)

    directive = next(
        c for c in llm.call_async.call_args_list if c.kwargs.get("schema") is None
    ).args[0][-1]["content"]
    assert "(mandatory) Learn the interval" in directive
    assert "(optional) Learn the price" in directive
    assert "Patience remaining: 2/2" in directive


@pytest.mark.asyncio
async def test_directive_is_ephemeral_not_persisted() -> None:
    """The directive is appended only for the LLM call, never to history."""
    llm = _mock_llm("draft", CheckedUtterance(utterance="draft"))
    scenario = _scenario()
    ledger = ConversationLedger.from_scenario(scenario)
    history_before = [dict(m) for m in _HISTORY]

    await generate_utterance(llm, scenario, ledger, _HISTORY, turn=0)

    assert history_before == _HISTORY
    draft_call = next(
        c for c in llm.call_async.call_args_list if c.kwargs.get("schema") is None
    )
    augmented = draft_call.args[0]
    assert len(augmented) == len(_HISTORY) + 1
    assert augmented[:-1] == _HISTORY


@pytest.mark.asyncio
async def test_directive_reflects_current_ledger_state() -> None:
    """Two calls with different ledger states produce different directives -
    there is no cached/stale directive; it's rendered fresh each call."""
    llm = _mock_llm("draft", CheckedUtterance(utterance="draft"))
    scenario = _scenario()
    ledger = ConversationLedger.from_scenario(scenario)

    await generate_utterance(llm, scenario, ledger, _HISTORY, turn=0)
    first_directive = llm.call_async.call_args_list[0].args[0][-1]["content"]

    ledger.patience_remaining = 1
    ledger.checklist[0].coverage = "complete"
    await generate_utterance(llm, scenario, ledger, _HISTORY, turn=1)
    second_directive = llm.call_async.call_args_list[-2].args[0][-1]["content"]

    assert first_directive != second_directive
    assert "Patience remaining: 2/2" in first_directive
    assert "Patience remaining: 1/2" in second_directive
    # The satisfied sub-goal leaves the pending list for the settled one.
    pending, _, settled = second_directive.partition("Already settled")
    assert "(mandatory) Learn the interval" in first_directive
    assert "Learn the interval" not in pending
    assert "- Learn the interval" in settled
    assert "Already settled" not in first_directive


@pytest.mark.asyncio
async def test_a_code_on_a_settled_want_stays_in_view() -> None:
    """A code is written on the want it serves. Dropping the want once it was
    met left the user without the code when the agent asked for it again."""
    llm = _mock_llm("draft", CheckedUtterance(utterance="draft"))
    scenario = _scenario()
    scenario.goal_checklist[
        0
    ].description = "Check the status of ORD-1001. Your code is 123456."
    ledger = ConversationLedger.from_scenario(scenario)
    ledger.checklist[0].coverage = "complete"

    await generate_utterance(llm, scenario, ledger, _HISTORY, turn=1)

    directive = llm.call_async.call_args_list[0].args[0][-1]["content"]
    assert "Your code is 123456." in directive.partition("Already settled")[2]


@pytest.mark.asyncio
async def test_check_and_rewrite_sees_real_history_not_the_directive() -> None:
    llm = _mock_llm("draft text", CheckedUtterance(utterance="draft text"))
    scenario = _scenario()
    ledger = ConversationLedger.from_scenario(scenario)

    await generate_utterance(llm, scenario, ledger, _HISTORY, turn=0)

    check_call = next(
        c
        for c in llm.call_async.call_args_list
        if c.kwargs.get("schema") is CheckedUtterance
    )
    prompt = check_call.args[0][0]["content"]
    assert "CUSTOMER: How often should I service it?" in prompt
    assert "AGENT: Every 6 months." in prompt
    assert "draft text" in prompt
    assert "Patience remaining" not in prompt
    assert "mandatory" not in prompt


@pytest.mark.asyncio
async def test_check_and_rewrite_can_change_the_final_utterance() -> None:
    llm = _mock_llm(
        "um, idk, whatever", CheckedUtterance(utterance="Could you clarify?")
    )
    scenario = _scenario()
    ledger = ConversationLedger.from_scenario(scenario)

    result = await generate_utterance(llm, scenario, ledger, _HISTORY, turn=0)

    assert result == "Could you clarify?"


@pytest.mark.asyncio
async def test_check_and_rewrite_failure_ships_the_raw_draft() -> None:
    llm = _mock_llm("raw draft", RuntimeError("boom"))
    scenario = _scenario()
    ledger = ConversationLedger.from_scenario(scenario)

    result = await generate_utterance(llm, scenario, ledger, _HISTORY, turn=0)

    assert result == "raw draft"


@pytest.mark.asyncio
async def test_draft_failure_propagates() -> None:
    """Unlike check-and-rewrite, a failed draft has no sensible fallback."""
    llm = _mock_llm(RuntimeError("boom"), CheckedUtterance(utterance="unused"))
    scenario = _scenario()
    ledger = ConversationLedger.from_scenario(scenario)

    with pytest.raises(RuntimeError):
        await generate_utterance(llm, scenario, ledger, _HISTORY, turn=0)


def _draft_call(llm: MagicMock) -> object:
    return next(
        c for c in llm.call_async.call_args_list if c.kwargs.get("schema") is None
    )


def _check_call(llm: MagicMock) -> object:
    return next(
        c
        for c in llm.call_async.call_args_list
        if c.kwargs.get("schema") is CheckedUtterance
    )


@pytest.mark.asyncio
async def test_no_web_search_tool_unless_asked() -> None:
    """Searching is a run-level policy (SimulationInput.web_search, on by
    default); this function spends nothing unless told to."""
    llm = _mock_llm("draft", CheckedUtterance(utterance="draft"))
    scenario = _scenario()
    ledger = ConversationLedger.from_scenario(scenario)

    await generate_utterance(llm, scenario, ledger, _HISTORY, turn=0)

    assert _draft_call(llm).kwargs.get("web_search") is False


@pytest.mark.asyncio
async def test_web_search_tool_reaches_the_draft_call() -> None:
    llm = _mock_llm("draft", CheckedUtterance(utterance="draft"))
    scenario = _scenario()
    ledger = ConversationLedger.from_scenario(scenario)

    await generate_utterance(llm, scenario, ledger, _HISTORY, turn=0, web_search=True)

    assert _draft_call(llm).kwargs["web_search"] is True


@pytest.mark.asyncio
async def test_the_directive_says_nothing_about_searching() -> None:
    """Permission belongs in the system prompt, which the model reads once,
    not repeated into every turn's ephemeral state block."""
    llm = _mock_llm("draft", CheckedUtterance(utterance="draft"))
    scenario = _scenario()
    ledger = ConversationLedger.from_scenario(scenario)

    await generate_utterance(llm, scenario, ledger, _HISTORY, turn=0, web_search=True)

    directive = _draft_call(llm).args[0][-1]["content"]
    assert "Patience remaining: 2/2" in directive
    assert "check" not in directive.lower()
    assert "search" not in directive.lower()


@pytest.mark.asyncio
async def test_check_and_rewrite_never_gets_web_search() -> None:
    """It is a style pass over existing text, and it asks for structured
    output, which Gemini refuses to combine with a tool."""
    llm = _mock_llm("draft", CheckedUtterance(utterance="draft"))
    scenario = _scenario()
    ledger = ConversationLedger.from_scenario(scenario)

    await generate_utterance(llm, scenario, ledger, _HISTORY, turn=0, web_search=True)

    assert not _check_call(llm).kwargs.get("web_search")
