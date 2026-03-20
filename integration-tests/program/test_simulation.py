# SPDX-License-Identifier: Apache-2.0

"""Integration tests for arksim simulation with real API calls."""

from __future__ import annotations

import asyncio
import json
import os
import uuid

import pytest

from .conftest import OPENAI_MODEL, requires_openai

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300), requires_openai]


class TestSimulateWithChatCompletions:
    """Test simulation using chat_completions agent (OpenAI)."""

    def test_simulate_with_chat_completions_agent(
        self,
        minimal_scenarios: str,
        agent_config_openai: dict,
        tmp_output_dir: str,
    ) -> None:
        from arksim.simulation_engine import (
            Simulation,
            SimulationInput,
            run_simulation,
        )

        sim_output_path = os.path.join(tmp_output_dir, "simulation", "simulation.json")
        simulation_input = SimulationInput.model_validate(
            {
                "agent_config": agent_config_openai,
                "scenario_file_path": minimal_scenarios,
                "num_conversations_per_scenario": 1,
                "max_turns": 5,
                "num_workers": 20,
                "provider": "openai",
                "model": OPENAI_MODEL,
                "output_file_path": sim_output_path,
            }
        )

        simulation = asyncio.run(run_simulation(simulation_input))

        assert isinstance(simulation, Simulation)
        assert simulation.schema_version == "v1.1"
        assert simulation.simulation_id
        assert simulation.generated_at

        assert len(simulation.conversations) == 3

        for convo in simulation.conversations:
            history = convo.conversation_history
            assert len(history) >= 2, "Should have at least 1 turn (user + assistant)"
            assert len(history) <= 10, "Should not exceed 5 turns (10 messages)"

            for i, msg in enumerate(history):
                expected = "simulated_user" if i % 2 == 0 else "assistant"
                assert msg.role == expected, (
                    f"Message {i} expected role '{expected}', got '{msg.role}'"
                )
                assert msg.content, "Message content should not be empty"

        # Verify output file was saved correctly
        assert os.path.isfile(sim_output_path)
        with open(sim_output_path) as f:
            data = json.load(f)

        loaded = Simulation.model_validate(data)
        assert len(loaded.conversations) == 3
        assert loaded.schema_version == "v1.1"


class TestSimulateWithCustomAgent:
    """Test simulation using a custom BaseAgent subclass."""

    def test_simulate_with_custom_agent(
        self,
        minimal_scenarios: str,
        tmp_output_dir: str,
    ) -> None:
        from arksim.config import AgentConfig, CustomConfig
        from arksim.llms.chat import LLM
        from arksim.scenario import Scenarios
        from arksim.simulation_engine import (
            Simulation,
            SimulationParams,
            Simulator,
        )
        from arksim.simulation_engine.agent.base import BaseAgent

        class SimpleTestAgent(BaseAgent):
            """A simple agent that echoes using OpenAI."""

            def __init__(self, agent_config: dict) -> None:
                super().__init__(agent_config)
                self._chat_id = str(uuid.uuid4())
                self._llm = LLM(model=OPENAI_MODEL, provider="openai")

            async def get_chat_id(self) -> str:
                return self._chat_id

            async def execute(self, user_query: str, **kwargs: object) -> str:
                response = await self._llm.call_async(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful insurance "
                                "assistant. Be concise. Never "
                                "exceed 80 words."
                            ),
                        },
                        {"role": "user", "content": user_query},
                    ]
                )
                return response

        scenarios = Scenarios.load(minimal_scenarios)
        agent_config = AgentConfig(
            agent_type="custom",
            agent_name="SimpleTestAgent",
            custom_config=CustomConfig(agent_class=SimpleTestAgent),
        )
        llm = LLM(model=OPENAI_MODEL, provider="openai")

        sim_params = SimulationParams(
            num_convos_per_scenario=1,
            max_turns=5,
            num_workers=20,
            output_file_path=os.path.join(
                tmp_output_dir, "simulation", "simulation.json"
            ),
        )

        simulator = Simulator(
            agent_config=agent_config,
            simulator_params=sim_params,
            llm=llm,
        )
        simulation = asyncio.run(simulator.simulate(scenarios))

        assert isinstance(simulation, Simulation)
        assert len(simulation.conversations) == 3

        for convo in simulation.conversations:
            assert len(convo.conversation_history) >= 2
            for i, msg in enumerate(convo.conversation_history):
                expected = "simulated_user" if i % 2 == 0 else "assistant"
                assert msg.role == expected
                assert msg.content
