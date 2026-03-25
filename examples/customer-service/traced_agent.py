# SPDX-License-Identifier: Apache-2.0
"""Traced variant of the customer-service example agent.

Uses the OpenAI Agents SDK's ``TracingProcessor`` interface to capture
tool calls automatically, instead of returning them in ``AgentResponse``.
This follows the same integration pattern as Braintrust and OpenInference.

When a ``TraceReceiver`` reference is available (same-process), tool calls
are injected directly into the receiver's buffer, bypassing HTTP entirely.
This avoids event loop contention and works at any parallelism level.
For agents in separate processes, use the HTTP export path instead.

Install: pip install openai-agents opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
Auth:    export OPENAI_API_KEY="<your-key>"
"""

from __future__ import annotations

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

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.simulation_engine.tool_types import ToolCall

# Global lock for thread-safe processor registration
_processor_lock = threading.Lock()
_shared_processor: ArksimTracingProcessor | None = None


class ArksimTracingProcessor(TracingProcessor):
    """Bridges the OpenAI Agents SDK tracing to arksim's trace receiver.

    Implements the SDK's ``TracingProcessor`` interface. When the SDK
    completes a function span (tool call), this processor either:

    - **In-process**: injects ``ToolCall`` directly into the receiver's
      buffer via ``submit_tool_calls()`` (no HTTP, no serialization)
    - **Cross-process**: creates OTel spans and exports via HTTP (requires
      opentelemetry-sdk + exporter, not used in this example)

    Handles concurrent conversations by mapping SDK trace_ids to
    (conversation_id, turn_id) context.
    """

    def __init__(self) -> None:
        # Maps SDK trace_id -> (conversation_id, turn_id, receiver_ref)
        self._trace_contexts: dict[str, tuple[str, int, object]] = {}
        self._lock = threading.Lock()

    def register_context(
        self,
        trace_id: str,
        conversation_id: str,
        turn_id: int,
        receiver: object = None,
    ) -> None:
        """Register routing context for an upcoming Runner.run() trace."""
        with self._lock:
            self._trace_contexts[trace_id] = (conversation_id, turn_id, receiver)

    def on_trace_start(self, _trace: Trace) -> None:
        pass

    def on_trace_end(self, trace: Trace) -> None:
        with self._lock:
            self._trace_contexts.pop(trace.trace_id, None)

    def on_span_start(self, _span: Span[Any]) -> None:
        pass

    def on_span_end(self, span: Span[Any]) -> None:
        """Convert completed function spans to ToolCall and submit to receiver."""
        if not isinstance(span.span_data, FunctionSpanData):
            return

        with self._lock:
            ctx = self._trace_contexts.get(span.trace_id)
        if ctx is None:
            return

        conversation_id, turn_id, receiver = ctx
        data = span.span_data

        # Parse arguments from JSON string
        arguments: dict[str, Any] = {}
        if data.input:
            try:
                parsed = json.loads(data.input)
                if isinstance(parsed, dict):
                    arguments = parsed
            except (json.JSONDecodeError, TypeError):
                pass

        # Build result string
        result = None
        if data.output is not None:
            result = (
                data.output if isinstance(data.output, str) else json.dumps(data.output)
            )

        # Build error string (SpanError is a TypedDict)
        error = None
        if span.error is not None:
            error = span.error["message"]

        tc = ToolCall(
            id=span.span_id,
            name=data.name,
            arguments=arguments,
            result=result,
            error=error,
        )

        # In-process: inject directly into receiver buffer
        if receiver is not None:
            receiver.submit_tool_calls(conversation_id, turn_id, [tc])

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass


def _get_shared_processor() -> ArksimTracingProcessor:
    """Get or create the shared tracing processor singleton."""
    global _shared_processor
    with _processor_lock:
        if _shared_processor is None:
            _shared_processor = ArksimTracingProcessor()
            set_trace_processors([_shared_processor])
        return _shared_processor


class TracedToolCallAgent(BaseAgent):
    """Agent that captures tool calls via TracingProcessor.

    When ``metadata["trace_receiver"]`` is provided (same-process),
    tool calls are injected directly into the receiver's buffer.
    No OTel SDK or HTTP is needed for this path.
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

        # Wrap Runner.run in a trace so we can map trace_id -> context.
        from agents.tracing import trace as sdk_trace

        with sdk_trace(workflow_name="agent_turn", group_id=chat_id) as t:
            self._processor.register_context(t.trace_id, chat_id, turn_id, receiver)
            self._last_result = await Runner.run(self._agent, input=input_list)

        return self._last_result.final_output

    async def close(self) -> None:
        pass
