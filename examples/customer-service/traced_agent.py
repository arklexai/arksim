# SPDX-License-Identifier: Apache-2.0
"""Traced variant of the customer-service example agent.

Captures tool calls via arksim's ``ArksimTracingProcessor`` and the
OpenAI Agents SDK's ``TracingProcessor`` interface. Compare with
``custom_agent.py`` which returns tool calls in ``AgentResponse``.

The agent registers the processor once at module load. The simulator
sets routing context via ``contextvars``, so no per-turn wrapping is
needed. The module loader caches modules by file path, so this
registration runs exactly once regardless of conversation count.

Install: pip install openai-agents
Auth:    export OPENAI_API_KEY="<your-key>"
"""

from __future__ import annotations

import uuid

from agents import Agent, Runner, RunResult
from agents.tracing import add_trace_processor

# Import shared tools and DB setup from the standard agent
from custom_agent import (
    _init_db,
    cancel_order,
    get_order,
    lookup_customer,
    search_products,
    send_verification_code,
    verify_customer,
)

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.tracing.openai import ArksimTracingProcessor

# Register once at module load. The module loader caches by file path,
# so this runs exactly once even when the simulator creates multiple
# agent instances for different conversations.
add_trace_processor(ArksimTracingProcessor())


class TracedToolCallAgent(BaseAgent):
    """Agent that captures tool calls via ArksimTracingProcessor.

    Tool calls are captured automatically by the processor's
    ``on_span_end`` callback. The simulator injects routing context
    (conversation_id, turn_id, receiver) via ``contextvars``.
    """

    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        _init_db()
        self._chat_id = str(uuid.uuid4())
        self._agent = Agent(
            name="assistant",
            instructions=(
                "You are a customer service assistant for an online store. "
                "You have access to tools to look up customers, check orders, "
                "search products, and cancel orders. Use them to help the user. "
                "Always confirm destructive actions like cancellations before proceeding."
            ),
            tools=[
                lookup_customer,
                get_order,
                search_products,
                cancel_order,
                send_verification_code,
                verify_customer,
            ],
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

    async def close(self) -> None:
        pass
