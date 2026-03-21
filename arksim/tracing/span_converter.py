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

logger = logging.getLogger(__name__)


def _get_attr(attrs: list[dict[str, Any]], key: str) -> str | None:
    """Extract a string attribute value from OTLP attribute list.

    OTLP attribute values are typed (stringValue, intValue, boolValue, etc.).
    Handles both JSON-style (``stringValue``) and protobuf-converted
    (``string_value``) field names. We check each with ``is not None`` to
    avoid dropping falsy values like empty strings, ``0``, or ``False``.
    """
    for attr in attrs:
        if attr.get("key") == key:
            value = attr.get("value", {})
            str_val = value.get("stringValue", value.get("string_value"))
            if str_val is not None:
                return str(str_val)
            int_val = value.get("intValue", value.get("int_value"))
            if int_val is not None:
                return str(int_val)
            bool_val = value.get("boolValue", value.get("bool_value"))
            if bool_val is not None:
                return str(bool_val)
    return None


def _first_attr(attrs: list[dict[str, Any]], *keys: str) -> str | None:
    """Return the first matching attribute from the given keys."""
    for key in keys:
        val = _get_attr(attrs, key)
        if val is not None:
            return val
    return None


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    """Parse a JSON string into a dict, returning empty dict on failure."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def span_to_tool_call(span: dict[str, Any]) -> ToolCall | None:
    """Convert a single OTLP span dict to a ToolCall, or None if not a tool span."""
    attrs = span.get("attributes", [])

    # Extract tool name: OTel GenAI > OpenInference > span name fallback
    name = _first_attr(attrs, "gen_ai.tool.name", "tool.name")
    if not name:
        # Fall back to span name (e.g. "execute_tool search_flights")
        span_name = span.get("name", "")
        if span_name.startswith("execute_tool "):
            name = span_name[len("execute_tool ") :]
        elif span_name:
            name = span_name
        else:
            return None

    # Extract arguments
    raw_args = _first_attr(
        attrs,
        "gen_ai.tool.call.arguments",
        "tool_call.function.arguments",
        "tool.parameters",
    )
    arguments = _parse_arguments(raw_args)

    # Extract result: OTel GenAI > OpenInference output.value
    result = _first_attr(attrs, "gen_ai.tool.call.result", "output.value")

    # Extract tool call ID
    tool_id = _first_attr(attrs, "gen_ai.tool.call.id", "tool_call.id", "tool.id")
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
