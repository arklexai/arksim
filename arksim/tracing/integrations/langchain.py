# SPDX-License-Identifier: Apache-2.0
"""LangChain / LangGraph tracing adapter (source=ToolCallSource.LANGCHAIN).

Inherits from ``AsyncCallbackHandler`` (which itself subclasses
``BaseCallbackHandler``) so the same instance satisfies both sync and
async dispatch paths: ``chain.invoke()`` and ``chain.ainvoke()`` both
accept it via ``callbacks=[...]``. LangGraph routes tool execution
through the same LangChain callback bus, so this one adapter covers both
frameworks. We do NOT list ``BaseCallbackHandler`` separately in the bases
because that produces an MRO conflict (it is already an ancestor of
``AsyncCallbackHandler``).

Verified against langchain-core 1.3.0 callback signatures:

    on_tool_start(serialized, input_str, *, run_id, parent_run_id=None,
                  tags=None, metadata=None, inputs=None, **kwargs)
    on_tool_end(output, *, run_id, parent_run_id=None, **kwargs)
    on_tool_error(error, *, run_id, parent_run_id=None, **kwargs)

``AsyncCallbackHandler.on_tool_*`` are declared ``async def``. We override
each with a sync ``def`` to shadow both the sync and async base methods.
On the async path the callback manager dispatches via
``getattr(handler, "on_tool_start")`` and inspects ``iscoroutinefunction``;
finding a sync method, it runs it under ``copy_context().run(...)`` so
contextvars propagate, which is what ``BaseTracingAdapter._submit`` needs.
``run_inline = True`` keeps that execution on the event loop instead of a
thread pool because the work is pure in-memory bookkeeping.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

try:
    from langchain_core.callbacks import AsyncCallbackHandler
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ArksimLangChainHandler requires the 'langchain' extra. "
        "Install with: pip install 'arksim[langchain]'"
    ) from exc

from arksim.simulation_engine.tool_types import ToolCall, ToolCallSource
from arksim.tracing.integrations._args import parse_tool_arguments
from arksim.tracing.integrations._base import BaseTracingAdapter
from arksim.tracing.integrations._pending import PendingToolCalls

logger = logging.getLogger(__name__)


class ArksimLangChainHandler(BaseTracingAdapter, AsyncCallbackHandler):
    """LangChain callback handler that captures tool calls into arksim.

    Pass via ``callbacks=[ArksimLangChainHandler()]`` to ``chain.invoke()``
    or ``chain.ainvoke()``. Each instance maintains its own pending-call
    map; do not share across concurrent conversations (use one per
    simulator agent).
    """

    # Run inline on the event loop instead of in a thread executor;
    # all work is in-memory bookkeeping with no blocking I/O.
    run_inline = True

    def __init__(self) -> None:
        super().__init__()
        self._pending = PendingToolCalls()

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,  # noqa: ANN401  (signature fixed by LangChain protocol)
    ) -> None:
        name = serialized.get("name") or kwargs.get("name") or ""
        arguments = parse_tool_arguments(input_str)
        self._pending.add(str(run_id), {"name": name, "arguments": arguments})

    def on_tool_end(
        self,
        output: Any,  # noqa: ANN401  (signature fixed by LangChain protocol)
        *,
        run_id: UUID,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        payload = self._pending.pop(str(run_id))
        if payload is None:
            logger.debug("unmatched on_tool_end for run_id=%s", run_id)
            return
        self._submit(
            ToolCall(
                id=str(run_id),
                name=payload["name"],
                arguments=payload["arguments"],
                result=str(output) if output is not None else None,
                source=ToolCallSource.LANGCHAIN,
            )
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,  # noqa: ANN401  (signature fixed by LangChain protocol)
    ) -> None:
        payload = self._pending.pop(str(run_id))
        if payload is None:
            logger.debug("unmatched on_tool_error for run_id=%s", run_id)
            return
        self._submit(
            ToolCall(
                id=str(run_id),
                name=payload["name"],
                arguments=payload["arguments"],
                error=f"{type(error).__name__}: {error}",
                source=ToolCallSource.LANGCHAIN,
            )
        )
