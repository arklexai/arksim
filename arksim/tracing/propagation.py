# SPDX-License-Identifier: Apache-2.0
"""W3C Trace Context propagation for cross-process trace correlation.

Generates traceparent headers, provides an httpx event hook for
automatic injection, and a convenience factory for traced HTTP clients.

The hook reads from the ``trace_traceparent`` contextvar, which the
simulator sets before each ``agent.execute()`` call for cross-process
agents.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import httpx

from arksim.tracing.context import trace_traceparent

if TYPE_CHECKING:
    from arksim.tracing.receiver import TraceReceiver


def generate_traceparent(
    receiver: TraceReceiver, conversation_id: str, turn_id: int
) -> str:
    """Generate a W3C traceparent header and register the mapping.

    Uses os.urandom for trace_id (128-bit) and parent_id (64-bit).
    The parent_id is synthetic since arksim is a trace context
    initiator, not a participant. tracestate is intentionally omitted.
    trace-flags is always 01 (sampled) because arksim requires all
    spans for evaluation.
    """
    trace_id = os.urandom(16).hex()
    parent_id = os.urandom(8).hex()
    receiver.register_trace_id(trace_id, conversation_id, turn_id)
    return f"00-{trace_id}-{parent_id}-01"


async def inject_trace_context(request: httpx.Request) -> None:
    """httpx event hook: injects traceparent from contextvar.

    Async because httpx 0.28+ awaits event hooks on AsyncClient.

    No-ops when contextvar is None (e.g. during A2A client init,
    agent card resolution, or other non-execute HTTP calls).
    """
    tp = trace_traceparent.get()
    if tp is not None:
        request.headers["traceparent"] = tp


def create_traced_client(**kwargs: Any) -> httpx.AsyncClient:  # noqa: ANN401
    """Create an httpx client with automatic trace context injection.

    Appends ``inject_trace_context`` to existing event_hooks if
    provided in kwargs (does not replace).
    """
    event_hooks = kwargs.pop("event_hooks", {})
    request_hooks = list(event_hooks.get("request", []))
    request_hooks.append(inject_trace_context)
    event_hooks["request"] = request_hooks
    return httpx.AsyncClient(event_hooks=event_hooks, **kwargs)
