# SPDX-License-Identifier: Apache-2.0
"""Pydantic AI integration for ArkSim.

Install: pip install pydantic-ai
Auth:    export OPENAI_API_KEY="<your-key>"
"""

from __future__ import annotations

import uuid

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent


class PydanticAIAgent(BaseAgent):
    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._chat_id = str(uuid.uuid4())
        self._agent = Agent(
            "openai:gpt-4o",
            system_prompt="You are a helpful assistant.",
        )
        self._history: list[ModelMessage] = []

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        result = await self._agent.run(
            user_query,
            message_history=self._history,
        )
        self._history = result.all_messages()
        return result.output
