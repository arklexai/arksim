# SPDX-License-Identifier: Apache-2.0
"""Strands Agents tracing adapter (source=ToolCallSource.STRANDS).

Verified against strands-agents 1.33.0. Registers a callback on
``AfterToolCallEvent`` via the ``HookProvider`` protocol; each event
becomes one ``ToolCall`` with success/error branching driven by the
event's ``exception`` field.

Deviations from the original plan template:

* The event's ``tool_use`` is a ``ToolUse`` TypedDict with shape
  ``{"name": str, "toolUseId": str, "input": Any}``. The tool arguments
  live in ``tool_use["input"]`` (not in ``tool_use`` directly), so we
  pass that field to ``parse_tool_arguments``.
* The tool name is sourced from ``tool_use["name"]`` rather than
  ``event.selected_tool.tool_name``; the former is always present, while
  ``selected_tool`` can be ``None`` when tool lookup fails.
* Success/error branching uses the dedicated ``event.exception`` field
  (``Exception | None``). ``event.result`` is always a ``ToolResult``
  TypedDict, never a union with ``Exception``.
* The ``ToolCall.id`` is populated from ``tool_use["toolUseId"]`` so
  arksim trace consumers can correlate emissions with Strands tool-use
  records.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from strands.hooks import AfterToolCallEvent, HookProvider, HookRegistry
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ArksimStrandsHookProvider requires the 'strands' extra. "
        "Install with: pip install 'arksim[strands]'"
    ) from exc

from arksim.simulation_engine.tool_types import ToolCall, ToolCallSource
from arksim.tracing.integrations._args import parse_tool_arguments
from arksim.tracing.integrations._base import BaseTracingAdapter

logger = logging.getLogger(__name__)


class ArksimStrandsHookProvider(BaseTracingAdapter, HookProvider):
    """Strands HookProvider capturing tool calls into arksim.

    Pass via ``Agent(hooks=[ArksimStrandsHookProvider()])``. Stateless
    across emissions.
    """

    def register_hooks(
        self,
        registry: HookRegistry,
        **kwargs: Any,  # noqa: ANN401  (signature fixed by Strands protocol)
    ) -> None:
        registry.add_callback(AfterToolCallEvent, self._on_after_tool_call)

    def _on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        tool_use = event.tool_use or {}
        tool_name = tool_use.get("name", "") or ""
        tool_use_id = tool_use.get("toolUseId", "") or ""
        arguments = parse_tool_arguments(tool_use.get("input"))

        exception = event.exception
        if exception is not None:
            self._submit(
                ToolCall(
                    id=tool_use_id,
                    name=tool_name,
                    arguments=arguments,
                    error=f"{type(exception).__name__}: {exception}",
                    source=ToolCallSource.STRANDS,
                )
            )
            return

        result = event.result
        self._submit(
            ToolCall(
                id=tool_use_id,
                name=tool_name,
                arguments=arguments,
                result=str(result) if result is not None else None,
                source=ToolCallSource.STRANDS,
            )
        )
