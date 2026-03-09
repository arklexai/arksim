# SPDX-License-Identifier: Apache-2.0
"""OpenAI Agents SDK integration for ArkSim.

Install: pip install openai-agents
Auth:    export OPENAI_API_KEY="<your-key>"
"""

from __future__ import annotations

import uuid

from agents import Agent, Runner

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent


class OpenAIAgentsSDKAgent(BaseAgent):
    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._chat_id = str(uuid.uuid4())
        self._agent = Agent(
            name="assistant",
            instructions="You are a helpful assistant.",
        )

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        result = await Runner.run(self._agent, input=user_query)
        return result.final_output
