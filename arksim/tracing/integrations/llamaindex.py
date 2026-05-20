# SPDX-License-Identifier: Apache-2.0
"""LlamaIndex tracing adapter (source=ToolCallSource.LLAMAINDEX).

Verified against llama-index-core 0.14.15. LlamaIndex's modern
``AgentWorkflow`` (``FunctionAgent``, ``ReActAgent``, ``CodeActAgent``)
emits tool calls as workflow stream events, not through the
``llama_index_instrumentation`` dispatcher. Specifically,
``base_agent.call_tool`` writes ``ToolCall`` and ``ToolCallResult``
events into ``ctx.write_event_to_stream``; consumers read them via
``handler.stream_events()``.

Deviations from the original plan template:

* The plan suggested ``class ArksimLlamaIndexObserver(BaseTracingAdapter,
  BaseEventHandler)`` to receive events via the instrumentation
  dispatcher. We do NOT subclass ``BaseEventHandler`` because the
  instrumentation dispatcher does not receive ``ToolCall`` /
  ``ToolCallResult`` events for ``AgentWorkflow`` in 0.14.x. The
  legacy ``AgentToolCallEvent`` class is defined but no longer emitted
  anywhere in core.
* The adapter exposes a plain ``observe(event)`` entry point and an
  ``async consume_stream(handler)`` convenience method. Users either
  forward stream events in their existing loop or hand a workflow
  handler to the adapter.

Event shapes:

* ``ToolCall``: ``tool_name`` (str), ``tool_kwargs`` (dict),
  ``tool_id`` (str) - emitted before the tool runs.
* ``ToolCallResult``: same three fields plus ``tool_output``
  (``ToolOutput`` with ``content`` property and ``exception`` accessor)
  and ``is_error`` via ``tool_output.is_error``.

Correlation is by ``tool_id``. ``PendingToolCalls`` holds in-flight
calls and sweeps anything older than 60s, so a workflow that crashes
between ``ToolCall`` and ``ToolCallResult`` cannot leak memory.

Usage:

    from llama_index.core.agent.workflow import FunctionAgent
    from arksim.tracing.integrations.llamaindex import ArksimLlamaIndexObserver

    observer = ArksimLlamaIndexObserver()
    workflow = FunctionAgent(tools=..., llm=...)
    handler = workflow.run(user_msg="...")
    async for event in handler.stream_events():
        observer.observe(event)
    final = await handler

Or, if you do not need to inspect stream events yourself:

    handler = workflow.run(user_msg="...")
    await observer.consume_stream(handler)
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from llama_index.core.agent.workflow import ToolCall as LIToolCall
    from llama_index.core.agent.workflow import ToolCallResult as LIToolCallResult
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ArksimLlamaIndexObserver requires the 'llamaindex' extra. "
        "Install with: pip install 'arksim[llamaindex]'"
    ) from exc

from arksim.simulation_engine.tool_types import ToolCall, ToolCallSource
from arksim.tracing.integrations._args import parse_tool_arguments
from arksim.tracing.integrations._base import BaseTracingAdapter
from arksim.tracing.integrations._pending import PendingToolCalls

logger = logging.getLogger(__name__)


class ArksimLlamaIndexObserver(BaseTracingAdapter):
    """Observe LlamaIndex ``AgentWorkflow`` tool-call stream events.

    Construct one per simulator agent. The observer maintains its own
    pending-call map keyed by ``tool_id``; do not share across
    concurrent conversations (use one per simulator agent).
    """

    def __init__(self) -> None:
        super().__init__()
        self._pending = PendingToolCalls()

    def observe(self, event: Any) -> None:  # noqa: ANN401  (workflow Event union)
        """Forward a workflow stream event to the adapter.

        Non-tool events are ignored. Call this for every event yielded
        by ``handler.stream_events()``.
        """
        if isinstance(event, LIToolCallResult):
            self._on_result(event)
            return
        if isinstance(event, LIToolCall):
            self._on_start(event)

    async def consume_stream(self, handler: Any) -> None:  # noqa: ANN401
        """Iterate a workflow handler's stream and forward all events.

        Convenience wrapper for users who do not need to inspect events
        themselves. Does not await the handler's final result; callers
        are expected to ``await handler`` afterwards if they need it.
        """
        async for event in handler.stream_events():
            self.observe(event)

    def _on_start(self, event: LIToolCall) -> None:
        self._pending.add(
            event.tool_id,
            {
                "name": event.tool_name,
                "arguments": parse_tool_arguments(event.tool_kwargs),
            },
        )

    def _on_result(self, event: LIToolCallResult) -> None:
        payload = self._pending.pop(event.tool_id)
        if payload is None:
            logger.debug("unmatched ToolCallResult for tool_id=%s", event.tool_id)
            return

        tool_output = event.tool_output
        content = tool_output.content
        if tool_output.is_error:
            exception = tool_output.exception
            if exception is not None:
                error_str: str | None = f"{type(exception).__name__}: {exception}"
            else:
                error_str = str(content) if content is not None else None
            self._submit(
                ToolCall(
                    id=event.tool_id,
                    name=payload["name"],
                    arguments=payload["arguments"],
                    error=error_str,
                    source=ToolCallSource.LLAMAINDEX,
                )
            )
            return

        self._submit(
            ToolCall(
                id=event.tool_id,
                name=payload["name"],
                arguments=payload["arguments"],
                result=str(content) if content is not None else None,
                source=ToolCallSource.LLAMAINDEX,
            )
        )
