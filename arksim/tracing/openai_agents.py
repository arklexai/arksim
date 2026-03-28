# SPDX-License-Identifier: Apache-2.0
"""OpenAI Agents SDK integration for arksim's trace receiver.

Provides ``ArksimTracingProcessor``, a ``TracingProcessor`` implementation
that captures tool calls from the OpenAI Agents SDK and injects them into
arksim's trace receiver. This follows the same pattern as Braintrust's
``BraintrustTracingProcessor`` and LangWatch's ``OpenAIAgentsInstrumentor``.

Usage::

    from arksim.tracing import ArksimTracingProcessor

    processor = ArksimTracingProcessor(receiver=receiver)

    async with processor.trace(conversation_id=chat_id, turn_id=turn_id):
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

    Handles concurrent conversations by mapping SDK trace_ids to
    ``(conversation_id, turn_id, receiver_ref)`` context.

    Raises ``ImportError`` at instantiation if ``openai-agents`` is not
    installed.
    """

    def __init__(self, receiver: TraceReceiver | None = None) -> None:
        if not _HAS_AGENTS_SDK:
            raise ImportError(
                "ArksimTracingProcessor requires the OpenAI Agents SDK. "
                "Install with: pip install openai-agents"
            )
        self._receiver = receiver
        self._registered = False
        # Maps SDK trace_id -> (conversation_id, turn_id, receiver_ref)
        self._trace_contexts: dict[str, tuple[str, int, TraceReceiver | None]] = {}
        self._lock = threading.Lock()

    def register_context(
        self,
        trace_id: str,
        conversation_id: str,
        turn_id: int,
        receiver: TraceReceiver | None = None,
    ) -> None:
        """Register routing context for an upcoming ``Runner.run()`` trace."""
        with self._lock:
            self._trace_contexts[trace_id] = (conversation_id, turn_id, receiver)

    @asynccontextmanager
    async def trace(
        self,
        conversation_id: str,
        turn_id: int,
        receiver: TraceReceiver | None = None,
    ) -> AsyncIterator[None]:
        """Context manager that wraps an agent turn for trace capture.

        Handles processor registration, SDK trace context, and
        conversation routing in a single call::

            async with processor.trace(conversation_id=chat_id, turn_id=turn_id):
                result = await Runner.run(agent, input=input_list)

        On first use, calls ``set_trace_processors([self])`` which replaces
        the SDK's processor list. This follows the same convention as
        Braintrust and LangWatch integrations. If you need to combine with
        other processors, call ``set_trace_processors`` manually instead
        of using this context manager.

        Args:
            conversation_id: Conversation ID for routing traces.
            turn_id: Turn index within the conversation.
            receiver: Optional receiver override. Falls back to the
                receiver passed at ``__init__``.
        """
        from agents.tracing import set_trace_processors
        from agents.tracing import trace as sdk_trace

        with self._lock:
            if not self._registered:
                set_trace_processors([self])
                self._registered = True

        resolved_receiver = receiver if receiver is not None else self._receiver

        with sdk_trace(workflow_name="agent_turn", group_id=conversation_id) as t:
            self.register_context(
                t.trace_id, conversation_id, turn_id, resolved_receiver
            )
            yield

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
