# SPDX-License-Identifier: Apache-2.0
"""Tests that Simulator.simulate respects num_convos_per_scenario."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from arksim.config.core.agent import AgentConfig
from arksim.scenario.entities import KnowledgeItem, Scenario, Scenarios
from arksim.simulation_engine.entities import ConversationState, SimulationParams
from arksim.simulation_engine.simulator import Simulator


def _make_scenarios(n: int) -> Scenarios:
    """Build a Scenarios object with *n* minimal scenarios."""
    return Scenarios(
        schema_version="v1",
        scenarios=[
            Scenario(
                scenario_id=f"sc-{i}",
                user_id=f"user-{i}",
                goal=f"goal-{i}",
                agent_context="context",
                knowledge=[KnowledgeItem(content="k")],
                user_profile="profile",
                origin={},
            )
            for i in range(n)
        ],
    )


def _make_agent_config() -> AgentConfig:
    return AgentConfig(
        agent_type="chat_completions",
        agent_name="test-agent",
        api_config={
            "endpoint": "http://localhost/v1/chat/completions",
            "headers": {},
            "body": {"model": "test", "messages": []},
        },
    )


@pytest.fixture()
def _patch_conversation(monkeypatch: pytest.MonkeyPatch) -> Callable[[], int]:
    """Patch _run_single_conversation to return a dummy ConversationState."""
    call_count = 0

    async def fake_run(
        self: Simulator,
        profile: str,
        goal: str,
        knowledge: list[KnowledgeItem],
        agent_context: str,
        max_turns: int,
        scenario_id: str = "",
        on_turn_complete: Callable[[], None] | None = None,
        on_turn_display: Callable[[str, str, str, int], None] | None = None,
    ) -> ConversationState:
        nonlocal call_count
        call_count += 1
        if on_turn_complete:
            on_turn_complete()
        return ConversationState(
            conversation_id=f"conv-{call_count}",
            scenario_id=scenario_id,
            conversation_history=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            simulated_user_prompt_template="tpl",
            simulated_user_profile=profile,
            user_goal=goal,
            knowledge=["k"],
            agent_context=agent_context,
        )

    monkeypatch.setattr(Simulator, "_run_single_conversation", fake_run)

    def get_count() -> int:
        return call_count

    return get_count


class TestNumConvosPerScenario:
    """Verify the simulator spawns the correct number of conversations."""

    @pytest.mark.asyncio
    async def test_single_scenario_multiple_convos(
        self, _patch_conversation: Callable[[], int]
    ) -> None:
        scenarios = _make_scenarios(1)
        params = SimulationParams(num_convos_per_scenario=4, max_turns=1)
        sim = Simulator(_make_agent_config(), params, MagicMock())

        result = await sim.simulate(scenarios)

        assert len(result.conversations) == 4
        assert _patch_conversation() == 4
        assert all(c.scenario_id == "sc-0" for c in result.conversations)

    @pytest.mark.asyncio
    async def test_multiple_scenarios_multiple_convos(
        self, _patch_conversation: Callable[[], int]
    ) -> None:
        scenarios = _make_scenarios(3)
        params = SimulationParams(num_convos_per_scenario=2, max_turns=1)
        sim = Simulator(_make_agent_config(), params, MagicMock())

        result = await sim.simulate(scenarios)

        assert len(result.conversations) == 6
        assert _patch_conversation() == 6
        for sc in scenarios.scenarios:
            count = sum(
                1 for c in result.conversations if c.scenario_id == sc.scenario_id
            )
            assert count == 2, f"Expected 2 convos for {sc.scenario_id}, got {count}"

    @pytest.mark.asyncio
    async def test_one_convo_per_scenario(
        self, _patch_conversation: Callable[[], int]
    ) -> None:
        scenarios = _make_scenarios(2)
        params = SimulationParams(num_convos_per_scenario=1, max_turns=1)
        sim = Simulator(_make_agent_config(), params, MagicMock())

        result = await sim.simulate(scenarios)

        assert len(result.conversations) == 2
        assert _patch_conversation() == 2
