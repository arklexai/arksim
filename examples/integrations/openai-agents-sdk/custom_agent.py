# SPDX-License-Identifier: Apache-2.0
"""OpenAI Agents SDK integration for arksim.

Install:
    pip install 'arksim[otel]'
    pip install openai-agents
Auth:
    export OPENAI_API_KEY="<your-key>"

Wires two mock tools (``lookup_order``, ``book_table``) into an OpenAI
Agents SDK ``Agent`` and registers ``ArksimTracingProcessor`` so every
``FunctionSpanData`` the SDK emits lands as a ``ToolCall`` on the right
turn in ``results/simulation/simulation.json``.
"""

from __future__ import annotations

import os
import uuid

from agents import Agent, Runner, RunResult
from agents.tracing import add_trace_processor
from tools import book_table, lookup_order

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.tracing.openai import ArksimTracingProcessor

# Register once at module load; the simulator caches modules by file path
# so this runs exactly once regardless of how many conversations start.
add_trace_processor(ArksimTracingProcessor())


class OpenAIAgentsSDKAgent(BaseAgent):
    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._chat_id = str(uuid.uuid4())
        self._agent = Agent(
            name="assistant",
            instructions=(
                "You are a helpful assistant with access to two tools: "
                "lookup_order(order_id) and book_table(party_size, time). "
                "Call them when relevant to answer the user."
            ),
            model=os.environ.get("OPENAI_MODEL", "gpt-5.1"),
            tools=[lookup_order, book_table],
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
