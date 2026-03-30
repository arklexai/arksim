# SPDX-License-Identifier: Apache-2.0
"""OpenAI Agents SDK integration for arksim's trace receiver.

Provides ``ArksimTracingProcessor``, a ``TracingProcessor`` implementation
that captures tool calls from the OpenAI Agents SDK and injects them into
arksim's trace receiver.

When used with arksim's simulator, routing context is set automatically
via ``contextvars``. Register the processor once and run your agent
normally::

    from arksim.tracing import ArksimTracingProcessor

    processor = ArksimTracingProcessor()
    processor.ensure_registered()

For standalone use (outside arksim's simulator), use the ``.trace()``
context manager to provide routing context explicitly::

    processor = ArksimTracingProcessor()

    async with processor.trace(conversation_id=chat_id, turn_id=turn_id, receiver=receiver):
        result = await Runner.run(agent, input=input_list)

Requires: ``pip install openai-agents``
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arksim.tracing.receiver import TraceReceiver

from arksim.simulation_engine.tool_types import ToolCall
from arksim.tracing.context import (
    trace_conversation_id,
    trace_receiver_ref,
    trace_turn_id,
)

logger = logging.getLogger(__name__)

try:
    from agents.tracing import Span, Trace, TracingProcessor
    from agents.tracing.span_data import FunctionSpanData

    _HAS_AGENTS_SDK = True
    _Base: type = TracingProcessor
except ImportError:
    _HAS_AGENTS_SDK = False
    _Base = object


class ArksimTracingProcessor(_Base):  # type: ignore[misc]
    """Bridges the OpenAI Agents SDK tracing to arksim's trace receiver.

    Implements the SDK's ``TracingProcessor`` interface. When the SDK
    completes a function span (tool call), this processor converts it
    to a ``ToolCall`` and injects it directly into the receiver's buffer
    via ``submit_tool_calls()`` (no HTTP, no serialization).

    Routing context is resolved in this order:
    1. Explicit context from ``.trace()`` context manager (standalone use)
    2. ``contextvars`` set by the simulator (automatic, zero setup)

    Raises ``ImportError`` at instantiation if ``openai-agents`` is not
    installed.
    """

    def __init__(self) -> None:
        if not _HAS_AGENTS_SDK:
            raise ImportError(
                "ArksimTracingProcessor requires the OpenAI Agents SDK. "
                "Install with: pip install openai-agents"
            )
        self._registered = False
        self._lock = threading.Lock()
        # Explicit context from .trace() (for standalone use outside simulator)
        self._trace_contexts: dict[str, tuple[str, int, TraceReceiver | None]] = {}

    @asynccontextmanager
    async def trace(
        self,
        conversation_id: str,
        turn_id: int,
        receiver: TraceReceiver | None = None,
    ) -> AsyncIterator[None]:
        """Context manager for standalone use (outside arksim's simulator).

        When using arksim's simulator, this is not needed. The simulator
        sets routing context automatically via ``contextvars``.

        For standalone use, this handles SDK trace context and routing::

            async with processor.trace(conversation_id=cid, turn_id=tid, receiver=recv):
                result = await Runner.run(agent, input=input_list)

        Args:
            conversation_id: Conversation ID for routing traces.
            turn_id: Turn index within the conversation.
            receiver: Receiver to submit tool calls to.
        """
        from agents.tracing import trace as sdk_trace

        self.ensure_registered()

        with sdk_trace(workflow_name="agent_turn", group_id=conversation_id) as t:
            with self._lock:
                self._trace_contexts[t.trace_id] = (
                    conversation_id,
                    turn_id,
                    receiver,
                )
            yield

        if receiver is not None:
            receiver.signal_turn_complete(conversation_id, turn_id)

    def ensure_registered(self) -> None:
        """Register this processor with the SDK if not already registered.

        Safe to call multiple times and from multiple module copies
        (arksim's dynamic module loader creates fresh copies per agent).
        Checks the SDK's processor list to avoid duplicates, with a
        fallback to an instance flag if SDK internals change.
        """
        from agents.tracing import add_trace_processor, get_trace_provider

        with self._lock:
            try:
                processors = get_trace_provider()._multi_processor._processors
                if any(isinstance(p, ArksimTracingProcessor) for p in processors):
                    return
            except AttributeError:
                if self._registered:
                    return
            add_trace_processor(self)
            self._registered = True

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

        # Resolve routing: explicit .trace() context first, then contextvars
        conversation_id: str | None = None
        turn_id: int | None = None
        receiver: TraceReceiver | None = None

        with self._lock:
            ctx = self._trace_contexts.get(span.trace_id)
        if ctx is not None:
            conversation_id, turn_id, receiver = ctx
        else:
            conversation_id = trace_conversation_id.get()
            turn_id = trace_turn_id.get()
            receiver = trace_receiver_ref.get()

        if conversation_id is None or turn_id is None:
            return

        data = span.span_data

        # Parse arguments from JSON string
        arguments: dict[str, Any] = {}
        if data.input:
            try:
                parsed = json.loads(data.input)
                if isinstance(parsed, dict):
                    arguments = parsed
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Malformed JSON in tool call arguments for %s", data.name
                )

        # Build result string
        result = None
        if data.output is not None:
            result = (
                data.output if isinstance(data.output, str) else json.dumps(data.output)
            )

        # Build error string (SpanError is a TypedDict)
        error = None
        if span.error is not None:
            error = span.error.get("message", "unknown error")

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
        else:
            logger.debug(
                "Tool call %s captured but no receiver registered for trace %s",
                tc.name,
                span.trace_id,
            )

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass
