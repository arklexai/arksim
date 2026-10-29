# SPDX-License-Identifier: Apache-2.0
"""Tests for the V2 turn judge: one LLM call per turn."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from arksim.scenario import (
    GoalChecklistItem,
    KnowledgeItem,
    Scenario,
)
from arksim.simulation_engine.core.state import (
    ConversationLedger,
    SubGoalCoverage,
    TurnJudgeVerdict,
)
from arksim.simulation_engine.core.turn_judge import judge_turn


def _scenario() -> Scenario:
    return Scenario(
        scenario_id="s1",
        user_id="u1",
        goal="learn the maintenance interval",
        agent_context="ctx",
        user_profile="profile",
        # Ground-truth knowledge must never reach the judge prompt.
        knowledge=[KnowledgeItem(id="kc_01", content="Interval is 6 months")],
        goal_checklist=[
            GoalChecklistItem(
                description="Learn the interval",
                satisfied_when="The agent states an interval",
            ),
        ],
    )


_HISTORY = [
    {"role": "system", "content": "instructions"},
    {"role": "assistant", "content": "How often should I service it?"},
    {"role": "user", "content": "Every 6 months."},
]


def _mock_llm(result: TurnJudgeVerdict | Exception) -> MagicMock:
    async def fake_call_async(
        messages: list[dict[str, Any]], schema: type | None = None, **kwargs: object
    ) -> TurnJudgeVerdict:
        assert schema is TurnJudgeVerdict
        if isinstance(result, Exception):
            raise result
        return result

    llm = MagicMock()
    llm.call_async = AsyncMock(side_effect=fake_call_async)
    return llm


@pytest.mark.asyncio
async def test_returns_the_verdict_from_a_single_call() -> None:
    verdict = TurnJudgeVerdict(
        coverage=[SubGoalCoverage(item_number=1, coverage="complete")],
        responsiveness="answered",
    )
    llm = _mock_llm(verdict)
    scenario = _scenario()
    ledger = ConversationLedger.from_scenario(scenario)

    result = await judge_turn(
        llm, scenario, ledger, _HISTORY, agent_reply="Every 6 months.", turn=0
    )

    assert result == verdict
    assert llm.call_async.await_count == 1


@pytest.mark.asyncio
async def test_prompt_carries_the_checklist_but_never_ground_truth() -> None:
    llm = _mock_llm(TurnJudgeVerdict())
    scenario = _scenario()
    ledger = ConversationLedger.from_scenario(scenario)

    await judge_turn(llm, scenario, ledger, _HISTORY, agent_reply="reply", turn=0)

    prompt = llm.call_async.call_args.args[0][0]["content"]
    assert "Interval is 6 months" not in prompt
    assert "kc_01" not in prompt
    assert "1. Learn the interval" in prompt
    assert "Satisfied when: The agent states an interval" in prompt
    assert "CUSTOMER: How often should I service it?" in prompt
    assert "AGENT: Every 6 months." in prompt


@pytest.mark.asyncio
async def test_failure_returns_none() -> None:
    llm = _mock_llm(RuntimeError("boom"))
    scenario = _scenario()
    ledger = ConversationLedger.from_scenario(scenario)

    result = await judge_turn(
        llm, scenario, ledger, _HISTORY, agent_reply="reply", turn=0
    )

    assert result is None
