# SPDX-License-Identifier: Apache-2.0
"""Strands Agents integration for arksim.

Install:
    pip install 'arksim[strands]'
    pip install strands-agents[openai]
Auth:
    export OPENAI_API_KEY="<your-key>"

Wires arksim's Strands tracing adapter into a Strands ``Agent`` with
two mock tools (lookup_order, book_table). Running
``arksim simulate-evaluate`` produces a simulation.json whose
``tool_calls`` field is populated by the captured invocations.

By default Strands routes through AWS Bedrock. This example uses the
``OpenAIModel`` provider so the wiring matches the rest of the
integration examples (single ``OPENAI_API_KEY`` env var).
"""

from __future__ import annotations

import uuid

from strands import Agent
from strands.models.openai import OpenAIModel
from tools import book_table, lookup_order

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.tracing.integrations.strands import ArksimStrandsHookProvider


class StrandsAgent(BaseAgent):
    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._chat_id = str(uuid.uuid4())
        self._agent = Agent(
            model=OpenAIModel(model_id="gpt-5.1"),
            tools=[lookup_order, book_table],
            system_prompt="You are a helpful assistant.",
            hooks=[ArksimStrandsHookProvider()],
        )

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        result = await self._agent.invoke_async(user_query)
        return str(result)
