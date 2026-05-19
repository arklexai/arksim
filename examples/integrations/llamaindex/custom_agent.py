# SPDX-License-Identifier: Apache-2.0
"""LlamaIndex integration for arksim.

Install:
    pip install 'arksim[llamaindex]'
    pip install llama-index-core llama-index-llms-openai
Auth:
    export OPENAI_API_KEY="<your-key>"

Wires arksim's LlamaIndex tracing adapter into a ``FunctionAgent`` with
two mock tools (lookup_order, book_table). Unlike the other integrations
that register a callback on the agent, LlamaIndex emits tool-call events
through the workflow stream, so we use
``ArksimLlamaIndexObserver.consume_stream(handler)`` to forward those
events into arksim. Running ``arksim simulate-evaluate`` produces a
simulation.json whose ``tool_calls`` field is populated by the captured
invocations.
"""

from __future__ import annotations

import uuid

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from tools import book_table, lookup_order

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.tracing.integrations.llamaindex import ArksimLlamaIndexObserver


class LlamaIndexAgent(BaseAgent):
    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._chat_id = str(uuid.uuid4())
        self._observer = ArksimLlamaIndexObserver()
        self._workflow = FunctionAgent(
            tools=[lookup_order, book_table],
            llm=OpenAI(model="gpt-5.1"),
            system_prompt=(
                "You are a helpful assistant with access to two tools: "
                "lookup_order(order_id) and book_table(party_size, time). "
                "Use them when relevant to answer the user."
            ),
        )

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        handler = self._workflow.run(user_msg=user_query)
        await self._observer.consume_stream(handler)
        result = await handler
        return str(result)
