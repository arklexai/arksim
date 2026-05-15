# SPDX-License-Identifier: Apache-2.0
"""Google ADK tracing adapter (source=ToolCallSource.GOOGLE_ADK).

Verified against google-adk 1.26.0. Subclasses ``BasePlugin`` and
overrides ``after_tool_callback`` to emit one ``ToolCall`` after every
tool invocation. ADK calls plugin callbacks in registration order; this
adapter never returns a value, so the original tool result is preserved
for downstream plugins and the agent itself.

Use::

    plugin = ArksimADKPlugin()
    runner = Runner(app_name="my-app", agent=root_agent, plugins=[plugin])

``BasePlugin.__init__`` requires a ``name``; the adapter defaults to
``"arksim_tracing"`` which is fine for the common single-plugin case and
can be overridden if multiple instances are registered.

Deviations from the original plan template:

* ``BasePlugin.__init__`` is ``__init__(self, name: str)`` (not no-arg),
  so the adapter forwards a default ``name``.
* ``ToolContext`` resolves to ``google.adk.agents.context.Context`` in
  this release; the imported alias from ``google.adk.tools`` remains the
  public re-export and is what the SDK passes at runtime.
* ``after_tool_callback`` returns ``None`` (the SDK accepts
  ``Optional[dict]``; a non-``None`` return would short-circuit
  remaining plugins and replace the tool result, which observation-only
  tracing must never do).

InMemoryRunner may not invoke plugin callbacks (per adk-python issue
#4464). Production runs use the real ``Runner``. Unit tests call
``await plugin.after_tool_callback(...)`` directly with synthetic
keyword arguments.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from google.adk.plugins import BasePlugin
    from google.adk.tools import BaseTool, ToolContext
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ArksimADKPlugin requires the 'google-adk' extra. "
        "Install with: pip install 'arksim[google-adk]'"
    ) from exc

from arksim.simulation_engine.tool_types import ToolCall, ToolCallSource
from arksim.tracing.integrations._args import parse_tool_arguments
from arksim.tracing.integrations._base import BaseTracingAdapter

logger = logging.getLogger(__name__)


class ArksimADKPlugin(BaseTracingAdapter, BasePlugin):
    """Google ADK plugin capturing tool calls into arksim.

    Stateless across emissions; safe to share one instance across
    concurrent simulator agents because routing context comes from
    contextvars set by the simulator on each conversation.
    """

    def __init__(self, name: str = "arksim_tracing") -> None:
        super().__init__(name=name)

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        self._submit(
            ToolCall(
                id=str(getattr(tool_context, "invocation_id", "") or ""),
                name=getattr(tool, "name", "") or "",
                arguments=parse_tool_arguments(tool_args),
                result=str(result) if result is not None else None,
                source=ToolCallSource.GOOGLE_ADK,
            )
        )
        return None
