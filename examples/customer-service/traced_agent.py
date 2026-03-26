# SPDX-License-Identifier: Apache-2.0
"""Traced variant of the customer-service example agent.

Uses arksim's ``ArksimTracingProcessor`` to capture tool calls via the
OpenAI Agents SDK's ``TracingProcessor`` interface. Compare with
``custom_agent.py`` which returns tool calls in ``AgentResponse``.

Install: pip install openai-agents
Auth:    export OPENAI_API_KEY="<your-key>"
"""

from __future__ import annotations

import threading
import uuid

from agents import Agent, Runner, RunResult
from agents.tracing import set_trace_processors
from agents.tracing import trace as sdk_trace

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
from arksim.tracing import ArksimTracingProcessor

# Shared singleton processor
_processor_lock = threading.Lock()
_shared_processor: ArksimTracingProcessor | None = None


def _get_shared_processor() -> ArksimTracingProcessor:
    """Get or create the shared tracing processor singleton."""
    global _shared_processor
    with _processor_lock:
        if _shared_processor is None:
            _shared_processor = ArksimTracingProcessor()
            set_trace_processors([_shared_processor])
        return _shared_processor


class TracedToolCallAgent(BaseAgent):
    """Agent that captures tool calls via ArksimTracingProcessor.

    When ``metadata["trace_receiver"]`` is provided (same-process),
    tool calls are injected directly into the receiver's buffer.
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
        self._processor = _get_shared_processor()

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        metadata = kwargs.get("metadata", {})
        turn_id = metadata.get("turn_id", 0)
        chat_id = metadata.get("chat_id", self._chat_id)
        receiver = metadata.get("trace_receiver")

        if self._last_result is not None:
            input_list = self._last_result.to_input_list() + [
                {"role": "user", "content": user_query}
            ]
        else:
            input_list = [{"role": "user", "content": user_query}]

        with sdk_trace(workflow_name="agent_turn", group_id=chat_id) as t:
            self._processor.register_context(t.trace_id, chat_id, turn_id, receiver)
            self._last_result = await Runner.run(self._agent, input=input_list)

        return self._last_result.final_output

    async def close(self) -> None:
        pass
