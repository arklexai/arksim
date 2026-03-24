# SPDX-License-Identifier: Apache-2.0
"""Convert OTLP spans to ToolCall objects.

Supports two attribute conventions:
- OTel GenAI semconv: gen_ai.tool.name, gen_ai.tool.call.arguments, etc.
- OpenInference (Arize): tool.name, tool_call.function.arguments, etc.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from arksim.simulation_engine.tool_types import ToolCall
from arksim.tracing._attrs import first_attr

logger = logging.getLogger(__name__)


def _parse_arguments(raw: str | None, span_name: str = "") -> dict[str, Any]:
    """Parse a JSON string into a dict, returning empty dict on failure."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            logger.warning(
                "Tool call arguments for span %r parsed as %s, expected dict: %s",
                span_name,
                type(parsed).__name__,
                raw[:200],
            )
            return {}
        return parsed
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            "Malformed JSON in tool call arguments for span %r: %s",
            span_name,
            raw[:200],
        )
        return {}


def span_to_tool_call(span: dict[str, Any]) -> ToolCall | None:
    """Convert a single OTLP span dict to a ToolCall, or None if not a tool span."""
    attrs = span.get("attributes", [])

    # Extract tool name: OTel GenAI > OpenInference > span name prefix.
    # Only convert spans that have a tool-specific attribute or the
    # "execute_tool " span name prefix. This filters out non-tool spans
    # (HTTP clients, DB queries, parent spans) that may be routed here
    # when using resource-level conversation_id with automatic instrumentation.
    name = first_attr(attrs, "gen_ai.tool.name", "tool.name")
    if not name:
        span_name = span.get("name", "")
        if span_name.startswith("execute_tool "):
            name = span_name[len("execute_tool ") :]
        else:
            return None

    # Extract arguments
    raw_args = first_attr(
        attrs,
        "gen_ai.tool.call.arguments",
        "tool_call.function.arguments",
        "tool.parameters",
    )
    arguments = _parse_arguments(raw_args, span_name=span.get("name", ""))

    # Extract result: OTel GenAI > OpenInference output.value
    result = first_attr(attrs, "gen_ai.tool.call.result", "output.value")

    # Extract tool call ID
    tool_id = first_attr(attrs, "gen_ai.tool.call.id", "tool_call.id", "tool.id")
    if not tool_id:
        tool_id = span.get("spanId", span.get("span_id", ""))

    # Extract error from span status
    error = None
    status = span.get("status", {})
    if status.get("code") == 2 or status.get("code") == "STATUS_CODE_ERROR":
        error = status.get("message", "unknown error")

    return ToolCall(
        id=tool_id,
        name=name,
        arguments=arguments,
        result=result,
        error=error,
    )


def spans_to_tool_calls(spans: list[dict[str, Any]]) -> list[ToolCall]:
    """Convert a list of OTLP span dicts to ToolCall objects.

    Spans that cannot be parsed as tool calls are skipped with a warning.
    """
    tool_calls: list[ToolCall] = []
    for span in spans:
        tc = span_to_tool_call(span)
        if tc is not None:
            tool_calls.append(tc)
    return tool_calls
