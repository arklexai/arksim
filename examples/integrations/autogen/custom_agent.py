# SPDX-License-Identifier: Apache-2.0
"""AutoGen integration for arksim.

Install:
    pip install autogen-agentchat autogen-ext[openai]
    pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
Auth:
    export OPENAI_API_KEY="<your-key>"

Wires two mock tools (``lookup_order``, ``book_table``) into an
``AssistantAgent`` and exports OTel spans to arksim's OTLP/HTTP trace
receiver. Each tool wraps its body in a ``gen_ai.tool.*`` span; a custom
span processor stamps ``arksim.conversation_id`` and ``arksim.turn_id``
on every span so the receiver can route tool calls to the right turn.
"""

from __future__ import annotations

import uuid

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from tools import book_table, lookup_order

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.tracing.context import trace_turn_id

_OTLP_ENDPOINT = "http://127.0.0.1:4318/v1/traces"


class _ArksimRoutingProcessor(SpanProcessor):
    """Stamp ``arksim.turn_id`` on every span from the current contextvar.

    ``arksim.conversation_id`` is set once on the ``Resource`` because each
    ``BaseAgent`` is bound to a single conversation. The turn id changes
    per execute() call and is read from arksim's contextvar.
    """

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


class AutoGenAgent(BaseAgent):
    """AutoGen ``AssistantAgent`` with tool calls captured via OTel.

    ``AssistantAgent`` maintains internal model context across ``on_messages``
    calls, so only the new user message is passed each turn.
    """

    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._chat_id = str(uuid.uuid4())

        resource = Resource.create(
            {
                "service.name": "arksim-autogen-example",
                "arksim.conversation_id": self._chat_id,
            }
        )
        self._provider = TracerProvider(resource=resource)
        self._provider.add_span_processor(_ArksimRoutingProcessor())
        self._provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=_OTLP_ENDPOINT))
        )
        # Set as global so tools.py picks it up via trace.get_tracer().
        trace.set_tracer_provider(self._provider)

        model_client = OpenAIChatCompletionClient(model="gpt-4o")
        self._agent = AssistantAgent(
            name="assistant",
            system_message=(
                "You are a helpful assistant with access to two tools: "
                "lookup_order(order_id) and book_table(party_size, time). "
                "Call them when relevant to answer the user."
            ),
            model_client=model_client,
            tools=[lookup_order, book_table],
        )

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        response = await self._agent.on_messages(
            [TextMessage(content=user_query, source="user")],
            cancellation_token=CancellationToken(),
        )
        self._provider.force_flush()
        return response.chat_message.content

    async def close(self) -> None:
        self._provider.shutdown()
