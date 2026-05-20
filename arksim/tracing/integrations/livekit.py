# SPDX-License-Identifier: Apache-2.0
"""LiveKit Agents tracing adapter (source=ToolCallSource.LIVEKIT).

Verified against livekit-agents 1.5.9. Subscribes to the
``function_tools_executed`` event on ``AgentSession``; each fired
``FunctionToolsExecutedEvent`` is a *batch* carrying parallel
``FunctionCall`` / ``FunctionCallOutput`` pairs, and the adapter emits
one ``ToolCall`` per pair.

Notes on the event shape:

* ``FunctionCall.arguments`` is a JSON-encoded string, normalized via
  ``parse_tool_arguments``.
* ``FunctionCallOutput`` may be ``None`` for a given call - LiveKit
  emits ``None`` when a tool raises ``StopResponse`` (no value should
  be sent back to the LLM). Those pairs produce a ``ToolCall`` with
  both ``result`` and ``error`` unset.
* When a tool raises, LiveKit produces a ``FunctionCallOutput`` with
  ``is_error=True`` and the error message in ``output``. We forward
  that string verbatim into ``ToolCall.error`` - LiveKit has already
  formatted it (typically as the exception's string form), and the
  exception type is not exposed on the event.
* ``ToolCall.id`` is populated from ``FunctionCall.call_id`` so arksim
  trace consumers can correlate emissions with LiveKit tool-use records.

Usage:

    from livekit.agents.voice import AgentSession
    from arksim.tracing.integrations.livekit import ArksimLiveKitHandler

    handler = ArksimLiveKitHandler()
    session = AgentSession(...)
    handler.attach_to(session)
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from livekit.agents.voice.events import FunctionToolsExecutedEvent
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ArksimLiveKitHandler requires the 'livekit-agents' extra. "
        "Install with: pip install 'arksim[livekit-agents]'"
    ) from exc

from arksim.simulation_engine.tool_types import ToolCall, ToolCallSource
from arksim.tracing.integrations._args import parse_tool_arguments
from arksim.tracing.integrations._base import BaseTracingAdapter

logger = logging.getLogger(__name__)

EVENT_NAME = "function_tools_executed"


class ArksimLiveKitHandler(BaseTracingAdapter):
    """LiveKit ``AgentSession`` event handler for tool-call capture.

    Construct one per simulator agent and call ``attach_to(session)``.
    Stateless across emissions.
    """

    def on_function_tools_executed(self, event: FunctionToolsExecutedEvent) -> None:
        for fn_call, fn_output in event.zipped():
            call_id = fn_call.call_id
            name = fn_call.name
            arguments = parse_tool_arguments(fn_call.arguments)

            if fn_output is None:
                self._submit(
                    ToolCall(
                        id=call_id,
                        name=name,
                        arguments=arguments,
                        source=ToolCallSource.LIVEKIT,
                    )
                )
                continue

            output = fn_output.output
            if fn_output.is_error:
                self._submit(
                    ToolCall(
                        id=call_id,
                        name=name,
                        arguments=arguments,
                        error=str(output) if output is not None else None,
                        source=ToolCallSource.LIVEKIT,
                    )
                )
                continue

            self._submit(
                ToolCall(
                    id=call_id,
                    name=name,
                    arguments=arguments,
                    result=str(output) if output is not None else None,
                    source=ToolCallSource.LIVEKIT,
                )
            )

    def attach_to(
        self,
        session: Any,  # noqa: ANN401  (livekit AgentSession)
    ) -> None:
        """Subscribe to ``function_tools_executed`` on the given session."""
        session.on(EVENT_NAME, self.on_function_tools_executed)
