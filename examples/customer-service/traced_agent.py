# SPDX-License-Identifier: Apache-2.0
"""Traced variant of the customer-service example agent.

Uses the OpenAI Agents SDK's ``TracingProcessor`` interface to capture
tool calls as OTel spans automatically, instead of returning them in
``AgentResponse``. This follows the same integration pattern as
Braintrust and OpenInference.

    Agent executes tools -> SDK fires TracingProcessor -> OTel spans pushed
    -> arksim captures -> evaluator scores

Install: pip install openai-agents opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
Auth:    export OPENAI_API_KEY="<your-key>"
"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from typing import Any

from agents import Agent, Runner, RunResult
from agents.tracing import Span, Trace, TracingProcessor, set_trace_processors
from agents.tracing.span_data import FunctionSpanData

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
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent

# Default trace receiver endpoint (matches arksim's default port)
_DEFAULT_TRACE_URL = "http://127.0.0.1:4318/v1/traces"

# Global lock for thread-safe processor registration
_processor_lock = threading.Lock()
_shared_processor: ArksimTracingProcessor | None = None


class ArksimTracingProcessor(TracingProcessor):
    """Bridges the OpenAI Agents SDK tracing to arksim's OTel trace receiver.

    Implements the SDK's ``TracingProcessor`` interface. When the SDK
    completes a function span (tool call), this processor creates an
    OTel span with ``gen_ai.tool.*`` attributes and exports it via
    ``OTLPSpanExporter`` to arksim's receiver.

    Handles concurrent conversations by mapping SDK trace_ids to
    (conversation_id, turn_id) context. Each ``Runner.run()`` call
    creates a unique trace_id, which we capture in ``on_trace_start``
    and look up in ``on_span_end`` to route spans correctly.
    """

    def __init__(self, endpoint: str = _DEFAULT_TRACE_URL) -> None:
        self._endpoint = endpoint
        self._provider = TracerProvider(
            resource=Resource.create({"service.name": "traced-tool-call-agent"})
        )
        # BatchSpanProcessor exports in a background thread, which is
        # essential when the trace receiver runs in the same process.
        # SimpleSpanProcessor would block the event loop on export.
        self._provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=endpoint),
                max_export_batch_size=1,
                schedule_delay_millis=100,
            )
        )
        self._tracer = self._provider.get_tracer("traced-tool-call-agent")
        # Maps SDK trace_id -> (conversation_id, turn_id)
        self._trace_contexts: dict[str, tuple[str, int]] = {}
        self._lock = threading.Lock()

    def register_context(
        self, trace_id: str, conversation_id: str, turn_id: int
    ) -> None:
        """Register routing context for an upcoming Runner.run() trace."""
        with self._lock:
            self._trace_contexts[trace_id] = (conversation_id, turn_id)

    def unregister_context(self, trace_id: str) -> None:
        """Clean up context after a trace completes."""
        with self._lock:
            self._trace_contexts.pop(trace_id, None)

    def on_trace_start(self, _trace: Trace) -> None:
        pass

    def on_trace_end(self, trace: Trace) -> None:
        self.unregister_context(trace.trace_id)

    def on_span_start(self, _span: Span[Any]) -> None:
        pass

    def on_span_end(self, span: Span[Any]) -> None:
        """Convert completed function spans to OTel spans with gen_ai attributes."""
        if not isinstance(span.span_data, FunctionSpanData):
            return

        # Look up conversation context from the span's parent trace
        with self._lock:
            ctx = self._trace_contexts.get(span.trace_id)
        if ctx is None:
            return

        conversation_id, turn_id = ctx
        data = span.span_data
        with self._tracer.start_as_current_span(
            f"execute_tool {data.name}"
        ) as otel_span:
            otel_span.set_attribute("arksim.conversation_id", conversation_id)
            otel_span.set_attribute("arksim.turn_id", turn_id)
            otel_span.set_attribute("gen_ai.tool.name", data.name)
            otel_span.set_attribute("gen_ai.tool.call.id", span.span_id)
            if data.input:
                otel_span.set_attribute("gen_ai.tool.call.arguments", data.input)
            if data.output is not None:
                output = data.output
                if not isinstance(output, str):
                    output = json.dumps(output)
                otel_span.set_attribute("gen_ai.tool.call.result", output)
            if span.error is not None:
                otel_span.set_attribute("error", True)
                otel_span.set_attribute("error.message", span.error.message)

    def shutdown(self) -> None:
        self._provider.shutdown()

    def force_flush(self) -> None:
        self._provider.force_flush()


def _get_shared_processor() -> ArksimTracingProcessor:
    """Get or create the shared tracing processor singleton."""
    global _shared_processor
    with _processor_lock:
        if _shared_processor is None:
            _shared_processor = ArksimTracingProcessor()
            set_trace_processors([_shared_processor])
        return _shared_processor


class TracedToolCallAgent(BaseAgent):
    """Agent that pushes tool calls as OTel spans via TracingProcessor."""

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

        if self._last_result is not None:
            input_list = self._last_result.to_input_list() + [
                {"role": "user", "content": user_query}
            ]
        else:
            input_list = [{"role": "user", "content": user_query}]

        # Register context before running so on_span_end can route spans.
        # Runner.run creates a Trace with a unique trace_id; we pre-register
        # it by running with a known trace wrapper.
        from agents.tracing import trace as sdk_trace

        with sdk_trace(
            workflow_name="agent_turn",
            group_id=chat_id,
        ) as t:
            self._processor.register_context(t.trace_id, chat_id, turn_id)
            self._last_result = await Runner.run(self._agent, input=input_list)

        # Force flush in a thread to avoid blocking the event loop.
        # The OTel exporter uses synchronous HTTP, and the trace receiver
        # runs on the same event loop, so flushing on the loop would deadlock.
        await asyncio.to_thread(self._processor.force_flush)

        return self._last_result.final_output

    async def close(self) -> None:
        pass
