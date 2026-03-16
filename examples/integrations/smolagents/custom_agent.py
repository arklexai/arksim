# SPDX-License-Identifier: Apache-2.0
"""Smolagents (Hugging Face) integration for ArkSim.

Install: pip install smolagents
Auth:    export OPENAI_API_KEY="<your-key>"
"""

from __future__ import annotations

import asyncio
import os
import uuid

from smolagents import CodeAgent, OpenAIServerModel

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent


class SmolagentsAgent(BaseAgent):
    """Smolagents agent wrapper.

    Uses reset=False on run() to maintain conversation history
    across turns internally.
    """

    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._chat_id = str(uuid.uuid4())
        model = OpenAIServerModel(
            model_id="gpt-4o",
            api_base="https://api.openai.com/v1",
            api_key=os.environ["OPENAI_API_KEY"],
        )
        self._agent = CodeAgent(tools=[], model=model)
        self._first_turn = True

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        if self._first_turn:
            result = await asyncio.to_thread(self._agent.run, user_query)
            self._first_turn = False
        else:
            result = await asyncio.to_thread(self._agent.run, user_query, reset=False)
        return str(result)
