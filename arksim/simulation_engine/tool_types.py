# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ToolCall(BaseModel):
    """A single tool/function call observed during a turn."""

    id: str
    name: str
    arguments: dict[str, Any] = {}
    result: str | None = None
    error: str | None = None


class AgentResponse(BaseModel):
    """Structured return from agent execution, carrying both text and tool calls."""

    content: str
    tool_calls: list[ToolCall] = []


__all__ = ["ToolCall", "AgentResponse"]
