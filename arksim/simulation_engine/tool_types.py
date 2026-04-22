# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolCallSource(str, Enum):
    """Provenance of a captured tool call.

    Identifies which arksim capture path produced a given ``ToolCall``,
    so evaluators and custom metrics can reason about data quality or
    filter by source. Inherits from ``str`` (rather than ``StrEnum``) to
    keep Python 3.10 compatibility; the ``str`` base is what makes values
    serialize to JSON as plain strings (e.g. ``"a2a_protocol"``).
    """

    #: Captured via arksim's A2A tool capture extension convention
    #: (extension URI declared in AgentCard + tool calls in Artifact
    #: metadata). Not produced by generic A2A transport participation.
    A2A_PROTOCOL = "a2a_protocol"

    #: Captured from an OpenAI Agents SDK RunResult (via
    #: ``ArksimTracingProcessor`` or the ``extract_tool_calls`` helper).
    OPENAI_AGENTS = "openai_agents"

    #: Captured from an OTLP / OpenInference span (via the trace receiver).
    OTEL_TRACE = "otel_trace"

    #: Captured from a Chat Completions connector response body (OpenAI,
    #: Anthropic, or Gemini formats). Result is not captured on this path:
    #: raw Chat Completions endpoints return tool results through follow-up
    #: ``role=tool`` messages sent by the client, and arksim does not supply
    #: the tool implementations needed to produce those messages.
    CHAT_COMPLETIONS = "chat_completions"


class A2AToolCaptureExtension:
    """A2A AgentExtension URI for arksim's tool call capture convention.

    A2A defines no native tool-call semantics (it defers to MCP for tool
    invocation). This extension is the convention arksim uses to surface
    tool execution data through A2A for evaluation purposes.

    The extension URI is declared in ``AgentCard.capabilities.extensions``
    by the remote agent. Tool call data is carried in
    ``Artifact.metadata[METADATA_KEY]`` as a list of tool call dicts
    matching the ``ToolCall`` schema. The artifact lists the URI in its
    ``extensions`` field to flag that this convention applies.

    Per the A2A extensions spec, breaking changes to this schema MUST
    bump the URI version (``/v1`` -> ``/v2``). ``METADATA_KEY`` is derived
    from ``URI``, so bumping the URI automatically changes the metadata
    key (which is also a wire-format change, as intended).
    """

    URI = "https://arksim.arklex.ai/a2a/tool-call-capture/v1"
    METADATA_KEY = f"{URI}/tool_calls"


class ToolCall(BaseModel):
    """A single tool/function call observed during a turn.

    Declares ``extra="ignore"`` explicitly so snapshots from future arksim
    versions that add new fields can still be loaded by older arksim
    without raising a ``ValidationError``.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str | None = None
    error: str | None = None
    source: ToolCallSource | None = None


class AgentResponse(BaseModel):
    """Structured return from agent execution, carrying both text and tool calls.

    Declares ``extra="ignore"`` for forward compatibility (matches
    ``ToolCall``): snapshots from future arksim versions that add new
    fields load on older arksim without raising ``ValidationError``.
    """

    model_config = ConfigDict(extra="ignore")

    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)


__all__ = ["A2AToolCaptureExtension", "AgentResponse", "ToolCall", "ToolCallSource"]
