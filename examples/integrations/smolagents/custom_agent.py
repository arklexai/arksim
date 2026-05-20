# SPDX-License-Identifier: Apache-2.0
"""Smolagents (Hugging Face) integration for arksim.

Install:
    pip install 'arksim[smolagents]'
Auth:
    export OPENAI_API_KEY="<your-key>"

Wires arksim's Smolagents tracing adapter into a ``CodeAgent`` with two
mock tools (lookup_order, book_table). ``ArksimSmolagentsCallback`` is
registered via ``step_callbacks=[...]`` and emits one ``ToolCall`` per
``ActionStep``. Running ``arksim simulate-evaluate`` produces a
simulation.json whose ``tool_calls`` field is populated by the captured
invocations.
"""

from __future__ import annotations

import asyncio
import os
import uuid

from smolagents import CodeAgent, OpenAIServerModel
from tools import book_table, lookup_order

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.tracing.integrations.smolagents import ArksimSmolagentsCallback

_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.1")


class SmolagentsAgent(BaseAgent):
    """Smolagents agent wrapper.

    Uses ``reset=False`` on ``run()`` after the first turn to maintain
    conversation history across turns internally.
    """

    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._chat_id = str(uuid.uuid4())
        self._callback = ArksimSmolagentsCallback()
        model = OpenAIServerModel(
            model_id=_MODEL,
            api_base="https://api.openai.com/v1",
            api_key=os.environ["OPENAI_API_KEY"],
        )
        self._agent = CodeAgent(
            tools=[lookup_order, book_table],
            model=model,
            step_callbacks=[self._callback],
        )
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
