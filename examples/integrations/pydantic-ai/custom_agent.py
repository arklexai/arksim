# SPDX-License-Identifier: Apache-2.0
"""Pydantic AI integration for arksim.

Install:
    pip install pydantic-ai
    pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
Auth:
    export OPENAI_API_KEY="<your-key>"

Wires two mock tools (``lookup_order``, ``book_table``) into a Pydantic AI
``Agent`` and exports OTel spans to arksim's OTLP/HTTP trace receiver.
Pydantic AI emits ``gen_ai.tool.*`` spans natively for every tool call
when ``instrument=True``; a custom span processor stamps
``arksim.conversation_id`` and ``arksim.turn_id`` on every span so the
receiver can route tool calls to the right turn.
"""

from __future__ import annotations

import uuid

from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic_ai import Agent, InstrumentationSettings
from pydantic_ai.messages import ModelMessage
from tools import book_table, lookup_order

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.tracing.context import trace_turn_id

_OTLP_ENDPOINT = "http://127.0.0.1:4318/v1/traces"


class _ArksimRoutingProcessor(SpanProcessor):
    """Stamp ``arksim.turn_id`` on every span from the current contextvar."""

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        turn_id = trace_turn_id.get()
        if turn_id is not None:
            span.set_attribute("arksim.turn_id", turn_id)

    def on_end(self, span: Span) -> None:
        return

    def shutdown(self) -> None:
        return

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class PydanticAIAgent(BaseAgent):
    """Pydantic AI agent wrapper with tool calls captured via OTel.

    Maintains conversation history via the ``message_history`` parameter
    and ``all_messages()`` to accumulate context across turns.
    """

    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._chat_id = str(uuid.uuid4())

        resource = Resource.create(
            {
                "service.name": "arksim-pydantic-ai-example",
                "arksim.conversation_id": self._chat_id,
            }
        )
        self._provider = TracerProvider(resource=resource)
        self._provider.add_span_processor(_ArksimRoutingProcessor())
        self._provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=_OTLP_ENDPOINT))
        )

        self._agent = Agent(
            "openai:gpt-4o",
            system_prompt=(
                "You are a helpful assistant with access to two tools: "
                "lookup_order(order_id) and book_table(party_size, time). "
                "Call them when relevant to answer the user."
            ),
            tools=[lookup_order, book_table],
            instrument=InstrumentationSettings(tracer_provider=self._provider),
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
        self._provider.force_flush()
        return result.output

    async def close(self) -> None:
        self._provider.shutdown()
