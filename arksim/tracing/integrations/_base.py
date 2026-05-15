# SPDX-License-Identifier: Apache-2.0
"""Shared submission logic for SDK-specific tracing adapters."""

from __future__ import annotations

import logging

from arksim.simulation_engine.tool_types import ToolCall
from arksim.tracing.context import (
    trace_conversation_id,
    trace_receiver_ref,
    trace_turn_id,
)

logger = logging.getLogger(__name__)


class BaseTracingAdapter:
    """Shared submission logic for SDK-specific tool-call adapters.

    Subclasses translate SDK events into ToolCall instances and call _submit.
    Routing context (conversation_id, turn_id, receiver) comes from contextvars
    set by the simulator. The adapter holds no per-conversation state.
    """

    def _submit(self, tool_call: ToolCall) -> None:
        conversation_id = trace_conversation_id.get()
        turn_id = trace_turn_id.get()
        if conversation_id is None or turn_id is None:
            return
        receiver = trace_receiver_ref.get()
        if receiver is None:
            logger.debug(
                "Tracing adapter has routing ids but no receiver; "
                "tool call %r dropped.",
                tool_call.name,
            )
            return
        receiver.submit_tool_calls(conversation_id, turn_id, [tool_call])
