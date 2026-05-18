# SPDX-License-Identifier: Apache-2.0
"""OpenAI Agents SDK integration for arksim.

This module provides two complementary ways to surface tool calls made
by agents built on the OpenAI Agents SDK:

* ``ArksimTracingProcessor`` - a ``TracingProcessor`` that captures tool
  calls via SDK tracing hooks and injects them into arksim's trace
  receiver (zero per-turn wrapping).
* ``extract_tool_calls`` - a helper for explicit capture from a
  ``RunResult``, used when your agent returns ``AgentResponse`` directly.

Both require ``pip install openai-agents``.

---

``ArksimTracingProcessor``
--------------------------

A ``TracingProcessor`` implementation that captures tool calls from the
OpenAI Agents SDK and injects them into arksim's trace receiver.

When used with ``arksim simulate``, the simulator sets routing context
via ``contextvars`` automatically. The agent registers the processor once
at module load::

    from agents.tracing import add_trace_processor
    from arksim.tracing.openai import ArksimTracingProcessor

    add_trace_processor(ArksimTracingProcessor())

For standalone use (outside arksim's simulator), register with the SDK
directly and use ``.trace()`` for routing context::

    processor = ArksimTracingProcessor()
    add_trace_processor(processor)

    async with processor.trace(conversation_id=cid, turn_id=tid, receiver=recv):
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

from arksim.simulation_engine.tool_types import ToolCall, ToolCallSource
from arksim.tracing.context import (
    trace_conversation_id,
    trace_receiver_ref,
    trace_turn_id,
)
from arksim.tracing.span_converter import _parse_arguments

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

        arguments = _parse_arguments(data.input, span_name=data.name)

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
            source=ToolCallSource.OPENAI_AGENTS,
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


def extract_tool_calls(result: object) -> list[ToolCall]:
    """Extract tool calls from an OpenAI Agents SDK RunResult.

    Pairs ``ToolCallItem`` entries with their ``ToolCallOutputItem`` outputs,
    returning arksim ``ToolCall`` objects in invocation order.

    Use this in custom agents that call ``Runner.run()`` and need to return
    tool calls via ``AgentResponse``::

        from arksim.tracing.openai import extract_tool_calls

        result = await Runner.run(agent, input=messages)
        return AgentResponse(
            content=result.final_output,
            tool_calls=extract_tool_calls(result),
        )

    Requires: ``pip install openai-agents``

    Args:
        result: A ``RunResult`` from ``Runner.run()``. Typed as ``object``
            to avoid a hard dependency on the ``openai-agents`` package.

    Returns:
        A list of ToolCall objects extracted from the result.
    """
    from agents.items import ToolCallItem, ToolCallOutputItem
    from openai.types.responses import ResponseFunctionToolCall

    # First pass: collect outputs keyed by call_id.
    outputs: dict[str, str] = {}
    for item in result.new_items:  # type: ignore[attr-defined]
        if isinstance(item, ToolCallOutputItem):
            # raw_item is a dict in older SDK versions and a typed object
            # in newer ones; handle both for forward compatibility.
            raw = item.raw_item
            if isinstance(raw, dict):
                call_id = raw.get("call_id", "")
                output = raw.get("output", "")
            else:
                call_id = getattr(raw, "call_id", "")  # noqa: B009
                output = getattr(raw, "output", "")  # noqa: B009
            if isinstance(output, list):
                output = json.dumps(output)
            outputs[call_id] = str(output)

    # Second pass: build tool calls in invocation order.
    tool_calls: list[ToolCall] = []
    for item in result.new_items:  # type: ignore[attr-defined]
        if not isinstance(item, ToolCallItem):
            continue
        raw = item.raw_item
        if not isinstance(raw, ResponseFunctionToolCall):
            continue
        call_id = raw.call_id
        tool_calls.append(
            ToolCall(
                id=call_id,
                name=raw.name,
                arguments=json.loads(raw.arguments) if raw.arguments else {},
                result=outputs.get(call_id),
                source=ToolCallSource.OPENAI_AGENTS,
            )
        )

    return tool_calls
