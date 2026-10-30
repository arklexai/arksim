# SPDX-License-Identifier: Apache-2.0
"""Tests for the Simulation Engine V2 turn loop in run_single_conversation.

There is a single loop now - no separate v1/legacy path. `compiled` in the
scenario helper below toggles whether the scenario has compiled fields
(behavior/goal_checklist), not which code path runs.

Utterance generation (draft + check & rewrite) is mocked as a whole via
`generate_utterance` - its own internals (directive rendering, the
ephemeral injection, the rewrite rubric) are covered by
test_utterance_generation.py, not here.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arksim.scenario import (
    BehaviorParams,
    GoalChecklistItem,
    KnowledgeItem,
    Scenario,
)
from arksim.simulation_engine.core.state import SubGoalCoverage, TurnJudgeVerdict
from arksim.simulation_engine.simulator import Simulator


def _scenario(compiled: bool = True, patience: int = 3) -> Scenario:
    return Scenario(
        scenario_id="s1",
        user_id="u1",
        goal="learn the maintenance interval",
        agent_context="ctx",
        user_profile="profile",
        knowledge=[KnowledgeItem(id="kc_01", content="Interval is 6 months")],
        behavior=BehaviorParams(patience=patience) if compiled else None,
        goal_checklist=(
            [
                GoalChecklistItem(
                    description="Learn the interval",
                    satisfied_when="The agent states an interval",
                ),
            ]
            if compiled
            else []
        ),
    )


def _make_simulator() -> Simulator:
    simulator_params = MagicMock()
    simulator_params.simulated_user_prompt_template = None
    # Set explicitly: this one reaches the rendered prompt and the output file,
    # so a MagicMock here would land in both.
    simulator_params.web_search = False
    return Simulator(
        agent_config=MagicMock(),
        simulator_params=simulator_params,
        llm=MagicMock(),
        trace_receiver=None,
    )


def _mock_agent(replies: list[str]) -> AsyncMock:
    agent = AsyncMock()
    agent.get_chat_id = AsyncMock(return_value="conv-1")
    agent.execute = AsyncMock(side_effect=replies)
    agent.close = AsyncMock()
    return agent


def _mock_generate(outputs: list[str]) -> AsyncMock:
    return AsyncMock(side_effect=outputs)


@pytest.mark.asyncio
async def test_user_satisfied_terminates_in_code() -> None:
    sim = _make_simulator()
    agent = _mock_agent(["Every 6 months.", "unused"])
    judge = AsyncMock(
        return_value=TurnJudgeVerdict(
            coverage=[SubGoalCoverage(item_number=1, coverage="complete")],
            responsiveness="answered",
        )
    )

    with (
        patch("arksim.simulation_engine.simulator.create_agent", return_value=agent),
        patch("arksim.simulation_engine.simulator.judge_turn", judge),
        patch(
            "arksim.simulation_engine.simulator.generate_utterance",
            _mock_generate(["how often?", "unused"]),
        ),
    ):
        state = await sim.run_single_conversation(_scenario(), max_turns=5)

    assert state is not None
    assert agent.execute.call_count == 1
    assert state.ledger is not None
    assert state.ledger.termination_reason == "user_satisfied"
    assert state.ledger.checklist[0].coverage == "complete"


@pytest.mark.asyncio
async def test_gave_up_terminates_on_zero_patience() -> None:
    sim = _make_simulator()
    agent = _mock_agent(["deflect", "deflect", "deflect", "deflect"])
    judge = AsyncMock(return_value=TurnJudgeVerdict(responsiveness="deflected"))

    with (
        patch("arksim.simulation_engine.simulator.create_agent", return_value=agent),
        patch("arksim.simulation_engine.simulator.judge_turn", judge),
        patch(
            "arksim.simulation_engine.simulator.generate_utterance",
            _mock_generate(["q1", "q2", "q3", "q4"]),
        ),
    ):
        state = await sim.run_single_conversation(_scenario(patience=2), max_turns=5)

    assert state is not None
    assert agent.execute.call_count == 2
    assert state.ledger is not None
    assert state.ledger.termination_reason == "gave_up"
    assert state.ledger.patience_remaining == 0


@pytest.mark.asyncio
async def test_refusal_terminates_as_blocked() -> None:
    sim = _make_simulator()
    agent = _mock_agent(["I can't help with that.", "unused"])
    judge = AsyncMock(return_value=TurnJudgeVerdict(responsiveness="refused"))

    with (
        patch("arksim.simulation_engine.simulator.create_agent", return_value=agent),
        patch("arksim.simulation_engine.simulator.judge_turn", judge),
        patch(
            "arksim.simulation_engine.simulator.generate_utterance",
            _mock_generate(["q1", "unused"]),
        ),
    ):
        state = await sim.run_single_conversation(_scenario(), max_turns=5)

    assert state is not None
    assert agent.execute.call_count == 1
    assert state.ledger is not None
    assert state.ledger.termination_reason == "blocked"
    assert state.ledger.blocked_reason == "refused"
    # Refused/transferred turns cost no patience of their own.
    assert state.ledger.patience_remaining == 3


@pytest.mark.asyncio
async def test_system_prompt_is_fixed_across_turns() -> None:
    """The system prompt never changes as the conversation progresses.

    Dynamic state (patience, checklist) now rides on the per-turn directive
    inside generate_utterance, which is mocked away here - so the loop-level
    contract to verify is simply that the persisted system prompt is
    untouched and contains none of it.
    """
    sim = _make_simulator()
    agent = _mock_agent(["Which store do you shop at?", "Every 6 months.", "bye"])
    judge = AsyncMock(
        side_effect=[
            TurnJudgeVerdict(responsiveness="asked_clarification"),
            TurnJudgeVerdict(
                coverage=[SubGoalCoverage(item_number=1, coverage="complete")],
                responsiveness="answered",
            ),
        ]
    )

    with (
        patch("arksim.simulation_engine.simulator.create_agent", return_value=agent),
        patch("arksim.simulation_engine.simulator.judge_turn", judge),
        patch(
            "arksim.simulation_engine.simulator.generate_utterance",
            _mock_generate(["q1", "at store #4128", "q3"]),
        ),
    ):
        state = await sim.run_single_conversation(_scenario(), max_turns=5)

    assert state is not None
    system_prompt = state.conversation_history[0]["content"]
    # user_profile stays in the fixed prompt for persona and communication
    # style. Goal and agent_context remain excluded.
    assert "profile" in system_prompt
    assert "learn the maintenance interval" not in system_prompt
    assert "ctx" not in system_prompt
    assert "Interval is 6 months" not in system_prompt


@pytest.mark.asyncio
async def test_v2_ignores_a_stray_stop_token() -> None:
    """Per the design doc, v2 termination is code-only: no ###STOP### token.

    Even if the generator emits it, the loop sends it to the agent like any
    other turn instead of self-terminating (unlike the old v1 loop).
    """
    sim = _make_simulator()
    agent = _mock_agent(["ok"])
    judge = AsyncMock(return_value=TurnJudgeVerdict(responsiveness="deflected"))

    with (
        patch("arksim.simulation_engine.simulator.create_agent", return_value=agent),
        patch("arksim.simulation_engine.simulator.judge_turn", judge),
        patch(
            "arksim.simulation_engine.simulator.generate_utterance",
            _mock_generate(["###STOP###"]),
        ),
    ):
        state = await sim.run_single_conversation(_scenario(), max_turns=1)

    assert state is not None
    assert agent.execute.call_count == 1
    assert state.ledger is not None
    assert state.ledger.termination_reason == "max_turns"


@pytest.mark.asyncio
async def test_max_turns_sets_reason() -> None:
    sim = _make_simulator()
    agent = _mock_agent(["a1", "a2"])
    judge = AsyncMock(return_value=TurnJudgeVerdict(responsiveness="deflected"))

    with (
        patch("arksim.simulation_engine.simulator.create_agent", return_value=agent),
        patch("arksim.simulation_engine.simulator.judge_turn", judge),
        patch(
            "arksim.simulation_engine.simulator.generate_utterance",
            _mock_generate(["q1", "q2"]),
        ),
    ):
        state = await sim.run_single_conversation(_scenario(patience=5), max_turns=2)

    assert state is not None
    assert state.ledger is not None
    assert state.ledger.termination_reason == "max_turns"


@pytest.mark.asyncio
async def test_judge_failure_keeps_loop_running() -> None:
    """A None verdict skips the ledger update; the loop still terminates."""
    sim = _make_simulator()
    agent = _mock_agent(["a1", "a2"])
    judge = AsyncMock(return_value=None)

    with (
        patch("arksim.simulation_engine.simulator.create_agent", return_value=agent),
        patch("arksim.simulation_engine.simulator.judge_turn", judge),
        patch(
            "arksim.simulation_engine.simulator.generate_utterance",
            _mock_generate(["q1", "q2"]),
        ),
    ):
        state = await sim.run_single_conversation(_scenario(), max_turns=2)

    assert state is not None
    assert agent.execute.call_count == 2
    assert state.ledger is not None
    assert state.ledger.termination_reason == "max_turns"
    assert state.ledger.patience_remaining == 3


@pytest.mark.asyncio
async def test_uncompiled_scenario_runs_the_same_loop() -> None:
    """An empty goal_checklist just has no mandatory sub-goal to satisfy.

    There is no separate legacy path: the judge still runs every turn, and
    the conversation ends via ``gave_up``/``max_turns`` since it can never
    reach ``user_satisfied`` with an empty checklist.
    """
    sim = _make_simulator()
    agent = _mock_agent(["a1", "a2"])
    judge = AsyncMock(return_value=TurnJudgeVerdict(responsiveness="deflected"))

    with (
        patch("arksim.simulation_engine.simulator.create_agent", return_value=agent),
        patch("arksim.simulation_engine.simulator.judge_turn", judge),
        patch(
            "arksim.simulation_engine.simulator.generate_utterance",
            _mock_generate(["q1", "q2"]),
        ),
    ):
        state = await sim.run_single_conversation(
            _scenario(compiled=False), max_turns=2
        )

    assert state is not None
    judge.assert_awaited()
    assert state.ledger is not None
    assert state.ledger.checklist == []
    assert state.ledger.termination_reason == "max_turns"


@pytest.mark.asyncio
async def test_output_conversation_carries_ledger() -> None:
    sim = _make_simulator()
    agent = _mock_agent(["Every 6 months."])
    judge = AsyncMock(
        return_value=TurnJudgeVerdict(
            coverage=[SubGoalCoverage(item_number=1, coverage="complete")],
            responsiveness="answered",
        )
    )

    with (
        patch("arksim.simulation_engine.simulator.create_agent", return_value=agent),
        patch("arksim.simulation_engine.simulator.judge_turn", judge),
        patch(
            "arksim.simulation_engine.simulator.generate_utterance",
            _mock_generate(["how often?"]),
        ),
    ):
        state = await sim.run_single_conversation(_scenario(), max_turns=5)

    assert state is not None
    conversation = sim.to_conversation_output(state)
    assert conversation.ledger is not None
    assert conversation.ledger.termination_reason == "user_satisfied"
    variables = conversation.simulated_user_prompt.variables
    # Fixed system prompt renders user_profile and web_search - agent_context
    # and goal never reach it (goal only reaches the generator decomposed into
    # the per-turn directive's pending_checklist).
    assert variables == {"scenario.user_profile": "profile", "web_search": False}
