# SPDX-License-Identifier: Apache-2.0
"""LlamaIndex integration for ArkSim.

Install: pip install llama-index llama-index-llms-openai
Auth:    export OPENAI_API_KEY="<your-key>"
"""

from __future__ import annotations

import uuid

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent


class LlamaIndexAgent(BaseAgent):
    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._chat_id = str(uuid.uuid4())
        llm = OpenAI(model="gpt-5.1")
        self._memory = ChatMemoryBuffer.from_defaults()
        self._agent = FunctionAgent(
            tools=[],
            llm=llm,
            system_prompt="You are a helpful assistant.",
        )

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        response = await self._agent.run(user_query, memory=self._memory)
        return str(response)
