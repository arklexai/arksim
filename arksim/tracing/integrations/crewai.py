# SPDX-License-Identifier: Apache-2.0
"""CrewAI tracing adapter (source=ToolCallSource.CREWAI).

Verified against crewai 1.6.1. Subscribes to ``ToolUsageFinishedEvent``
and ``ToolUsageErrorEvent`` on the CrewAI event bus; each emission becomes
one ``ToolCall`` with the captured tool name, arguments (normalized via
``parse_tool_arguments``), and result-or-error.

Deviations from the original plan template:

* Import path is ``crewai.events`` in crewai 1.6+ (was
  ``crewai.utilities.events`` in older releases).
* ``ToolUsageFinishedEvent`` carries the tool result in the ``output``
  field, not ``result``.
* The event model has no ``event_id``/correlation id, so ``ToolCall.id``
  is left empty - a single completion-style event is self-contained and
  needs no correlation across emissions.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from crewai.events import (
        BaseEventListener,
        ToolUsageErrorEvent,
        ToolUsageFinishedEvent,
    )
    from crewai.events.event_bus import CrewAIEventsBus
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ArksimCrewEventListener requires the 'crewai' extra. "
        "Install with: pip install 'arksim[crewai]'"
    ) from exc

from arksim.simulation_engine.tool_types import ToolCall, ToolCallSource
from arksim.tracing.integrations._args import parse_tool_arguments
from arksim.tracing.integrations._base import BaseTracingAdapter

logger = logging.getLogger(__name__)


class ArksimCrewEventListener(BaseTracingAdapter, BaseEventListener):
    """CrewAI event listener capturing tool calls into arksim.

    Construct one per simulator agent: ``ArksimCrewEventListener()``. The
    base class registers handlers eagerly against the global
    ``crewai_event_bus`` in ``__init__``, so simply instantiating the
    listener is enough; no explicit ``Crew(event_listeners=[...])`` wiring
    is required. The adapter is stateless across emissions.
    """

    def setup_listeners(self, crewai_event_bus: CrewAIEventsBus) -> None:
        # Note: CrewAI's event bus dispatches handlers via
        # ``ThreadPoolExecutor.submit(ctx.run, ...)`` after
        # ``contextvars.copy_context()``, which preserves arksim's routing
        # context (trace_conversation_id, trace_turn_id,
        # trace_receiver_ref) across the thread boundary. The adapter's
        # correctness depends on that copy. If CrewAI ever drops the
        # ``copy_context()`` step (or switches to a non-context-preserving
        # dispatch path), every CrewAI tool call will start dropping
        # silently because the contextvars will read as None on the worker
        # thread. The cross-adapter contract test would catch that
        # regression.
        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def _on_tool_finished(
            source: Any,  # noqa: ANN401, ARG001  (signature fixed by crewai bus)
            event: ToolUsageFinishedEvent,
        ) -> None:
            output = event.output
            self._submit(
                ToolCall(
                    id="",
                    name=event.tool_name,
                    arguments=parse_tool_arguments(event.tool_args),
                    result=str(output) if output is not None else None,
                    source=ToolCallSource.CREWAI,
                )
            )

        @crewai_event_bus.on(ToolUsageErrorEvent)
        def _on_tool_error(
            source: Any,  # noqa: ANN401, ARG001  (signature fixed by crewai bus)
            event: ToolUsageErrorEvent,
        ) -> None:
            error = event.error
            if isinstance(error, BaseException):
                error_str = f"{type(error).__name__}: {error}"
            else:
                error_str = str(error)
            self._submit(
                ToolCall(
                    id="",
                    name=event.tool_name,
                    arguments=parse_tool_arguments(event.tool_args),
                    error=error_str,
                    source=ToolCallSource.CREWAI,
                )
            )
