# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolCallSource:
    """Constants for tool call provenance tracking."""

    A2A_PROTOCOL = "a2a_protocol"


class ToolCall(BaseModel):
    """A single tool/function call observed during a turn."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str | None = None
    error: str | None = None
    source: str | None = None


class AgentResponse(BaseModel):
    """Structured return from agent execution, carrying both text and tool calls."""

    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)


__all__ = ["AgentResponse", "ToolCall", "ToolCallSource"]
