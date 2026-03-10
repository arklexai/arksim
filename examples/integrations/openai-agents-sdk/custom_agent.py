# SPDX-License-Identifier: Apache-2.0
"""OpenAI Agents SDK integration for ArkSim.

Install: pip install openai-agents
Auth:    export OPENAI_API_KEY="<your-key>"
"""

from __future__ import annotations

import uuid

from agents import Agent, Runner, RunResult

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
        self._last_result: RunResult | None = None

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        if self._last_result is not None:
            input_list = self._last_result.to_input_list() + [
                {"role": "user", "content": user_query}
            ]
        else:
            input_list = [{"role": "user", "content": user_query}]
        self._last_result = await Runner.run(self._agent, input=input_list)
        return self._last_result.final_output
