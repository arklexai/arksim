# SPDX-License-Identifier: Apache-2.0
"""OTel-compatible trace receiver for capturing tool calls from agent spans."""

from __future__ import annotations

from arksim.tracing.config import TraceReceiverConfig
from arksim.tracing.receiver import TraceReceiver
from arksim.tracing.span_converter import spans_to_tool_calls

__all__ = ["TraceReceiver", "TraceReceiverConfig", "spans_to_tool_calls"]
