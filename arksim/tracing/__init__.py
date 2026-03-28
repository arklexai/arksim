# SPDX-License-Identifier: Apache-2.0
"""OTel-compatible trace receiver for capturing tool calls from agent spans."""

from __future__ import annotations

from arksim.tracing.config import TraceReceiverConfig
from arksim.tracing.receiver import TraceReceiver
from arksim.tracing.span_converter import spans_to_tool_calls

__all__ = [
    "ArksimTracingProcessor",
    "TraceReceiver",
    "TraceReceiverConfig",
    # Public utility: converts raw OTLP span dicts to ToolCall objects.
    # Useful for custom receivers, testing, and processing spans outside
    # the built-in TraceReceiver pipeline.
    "spans_to_tool_calls",
]


def __getattr__(name: str) -> object:
    """Lazy import for optional dependencies."""
    if name == "ArksimTracingProcessor":
        from arksim.tracing.openai_agents import ArksimTracingProcessor

        return ArksimTracingProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
