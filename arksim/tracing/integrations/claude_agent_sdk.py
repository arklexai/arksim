# SPDX-License-Identifier: Apache-2.0
"""Claude Agent SDK tracing adapter (source=ToolCallSource.CLAUDE_AGENT_SDK).

Verified against claude-agent-sdk 0.1.48. The SDK invokes hooks as
``async`` callables with the signature
``(input_data: PostToolUseHookInput, tool_use_id: str | None, context: HookContext)``
and expects an awaitable JSON-shaped return value (``{}`` is a valid
"no-op" response). Hooks are registered on ``ClaudeAgentOptions`` as a
dict keyed by event name, with each value a list of ``HookMatcher``
instances; ``HookMatcher(matcher=None, hooks=[callable])`` runs the
callable for every event of that kind.

Use::

    hooks = ArksimClaudeHooks()
    options = ClaudeAgentOptions(hooks=hooks.hooks_dict())

The ``input_data`` payload (``PostToolUseHookInput``) carries the tool
name, the tool input dict, the tool response, and a stable
``tool_use_id`` that arksim copies onto ``ToolCall.id`` for downstream
correlation. ``tool_response`` is typed ``Any``; we stringify so
``ToolCall.result`` is a printable string regardless of whether the tool
returned a scalar, a dict, or a structured Pydantic model.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from claude_agent_sdk import HookMatcher
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ArksimClaudeHooks requires the 'claude-agent' extra. "
        "Install with: pip install 'arksim[claude-agent]'"
    ) from exc

from arksim.simulation_engine.tool_types import ToolCall, ToolCallSource
from arksim.tracing.integrations._args import parse_tool_arguments
from arksim.tracing.integrations._base import BaseTracingAdapter

logger = logging.getLogger(__name__)


class ArksimClaudeHooks(BaseTracingAdapter):
    """Claude Agent SDK hooks producer capturing tool calls into arksim.

    Stateless across emissions; safe to share one instance across
    concurrent simulator agents because routing context comes from
    contextvars set by the simulator on each conversation.
    """

    async def post_tool_use(
        self,
        input_data: dict[str, Any],
        tool_use_id: str | None,
        context: Any,  # noqa: ANN401, ARG002  (signature fixed by SDK)
    ) -> dict[str, Any]:
        tool_name = input_data.get("tool_name", "") or ""
        tool_input = input_data.get("tool_input")
        tool_response = input_data.get("tool_response")
        self._submit(
            ToolCall(
                id=str(tool_use_id or ""),
                name=tool_name,
                arguments=parse_tool_arguments(tool_input),
                result=str(tool_response) if tool_response is not None else None,
                source=ToolCallSource.CLAUDE_AGENT_SDK,
            )
        )
        return {}

    def hooks_dict(self) -> dict[str, list[HookMatcher]]:
        """Return the hooks dict to pass to ``ClaudeAgentOptions(hooks=...)``."""
        return {"PostToolUse": [HookMatcher(hooks=[self.post_tool_use])]}
