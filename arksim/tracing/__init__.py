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
    "extract_tool_calls",
    "spans_to_tool_calls",
]


def __getattr__(name: str) -> object:
    """Lazy import for optional dependencies."""
    if name == "ArksimTracingProcessor":
        from arksim.tracing.openai import ArksimTracingProcessor

        return ArksimTracingProcessor
    if name == "extract_tool_calls":
        from arksim.tracing.openai import extract_tool_calls

        return extract_tool_calls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
