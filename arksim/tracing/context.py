# SPDX-License-Identifier: Apache-2.0
"""Context variables for passing trace routing state through the call stack.

The simulator sets these before each ``agent.execute()`` call so that
tracing processors (e.g. ``ArksimTracingProcessor``) can read routing
context without the agent needing to pass it explicitly.

These are ``contextvars`` (not thread-locals), so they propagate correctly
across ``asyncio.create_task`` boundaries and isolate concurrent
conversations running under different async tasks.
"""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arksim.tracing.receiver import TraceReceiver

# Routing context set by the simulator before agent.execute()
trace_conversation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "trace_conversation_id", default=None
)
trace_turn_id: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "trace_turn_id", default=None
)
trace_receiver_ref: contextvars.ContextVar[TraceReceiver | None] = (
    contextvars.ContextVar("trace_receiver_ref", default=None)
)


def _set_trace_context(
    conversation_id: str,
    turn_id: int,
    receiver: TraceReceiver,
) -> None:
    """Set routing context before agent.execute(). Called by the simulator."""
    trace_conversation_id.set(conversation_id)
    trace_turn_id.set(turn_id)
    trace_receiver_ref.set(receiver)


def _clear_trace_context() -> None:
    """Clear routing context after agent.execute(). Called by the simulator."""
    trace_conversation_id.set(None)
    trace_turn_id.set(None)
    trace_receiver_ref.set(None)
