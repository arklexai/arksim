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

    Subclass contract for ToolCall construction:

    * Pass ``result=str(tool_output) if tool_output is not None else None``
      so empty-string and falsy results (``False``, ``0``, ``""``) are
      preserved as their string form rather than coerced to ``None``.
    * Pass ``error=str(error_value) if error_value is not None else None``
      on the error path. Use the same ``str(...)`` coercion when the SDK
      types the field as ``Any``, since ``ToolCall.error`` is strict
      ``str | None`` (Pydantic raises ``ValidationError`` on non-string
      inputs).
    * Downstream consumers should treat ``result=""`` as "tool returned
      an empty string" and ``result=None`` as "no result captured".

    Observability never breaks the observed: ``_submit`` catches and logs
    any exception raised by the receiver, so a misbehaving receiver
    cannot propagate back into the SDK callback that invoked the adapter.
    """

    def _submit(self, tool_call: ToolCall) -> None:
        conversation_id = trace_conversation_id.get()
        turn_id = trace_turn_id.get()
        if conversation_id is None or turn_id is None:
            logger.debug(
                "Tracing adapter has no routing context; tool call %r dropped.",
                tool_call.name,
            )
            return
        receiver = trace_receiver_ref.get()
        if receiver is None:
            logger.debug(
                "Tracing adapter has routing ids but no receiver; "
                "tool call %r dropped.",
                tool_call.name,
            )
            return
        try:
            receiver.submit_tool_calls(conversation_id, turn_id, [tool_call])
        except Exception:
            # Observability must never break the observed. Swallow the
            # receiver's exception, log it with full routing context, and
            # let the SDK callback that invoked us return normally.
            logger.exception(
                "Tracing receiver raised while submitting tool call "
                "(conversation_id=%s, turn_id=%s, tool=%r); dropped.",
                conversation_id,
                turn_id,
                tool_call.name,
            )
