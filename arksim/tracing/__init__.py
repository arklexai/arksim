# SPDX-License-Identifier: Apache-2.0
"""OTel-compatible trace receiver for capturing tool calls from agent spans."""

from __future__ import annotations

from .config import TraceReceiverConfig
from .receiver import TraceReceiver
from .span_converter import spans_to_tool_calls

__all__ = ["TraceReceiver", "TraceReceiverConfig", "spans_to_tool_calls"]
