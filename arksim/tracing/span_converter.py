# SPDX-License-Identifier: Apache-2.0
"""Convert OTLP spans to ToolCall objects.

Tool execution spans are recognized from these conventions:

- OTel GenAI semconv: ``gen_ai.operation.name == execute_tool`` with
  ``gen_ai.tool.*``, or an ``execute_tool <name>`` span name.
- OpenInference (Arize): ``openinference.span.kind == TOOL``, ``tool.name``,
  ``input.value`` / ``output.value``. ``tool.parameters`` is the tool's schema,
  not the call's arguments. For LangChain, a single-argument tool's
  ``input.value`` is the bare value; see ``proposed_arguments``.
- LangSmith OTel: ``langsmith.span.kind == tool`` with ``gen_ai.prompt`` /
  ``gen_ai.completion`` payloads (sent as bytes).
- Google ADK: ``gcp.vertex.agent.tool_call_args`` / ``tool_response``.

A span that declares another kind (chat, LLM, chain, retriever) is never a tool
call, so model-proposed calls and tool definitions on LLM spans are ignored.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from arksim.simulation_engine.tool_types import ToolCall, ToolCallSource
from arksim.tracing._attrs import first_attr, get_attr

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


ARGUMENT_KEYS = (
    "gen_ai.tool.call.arguments",
    "gcp.vertex.agent.tool_call_args",
    "tool_call.function.arguments",
    "input.value",
    "gen_ai.prompt",
)
RESULT_KEYS = (
    "gen_ai.tool.call.result",
    "gcp.vertex.agent.tool_response",
    "output.value",
    "gen_ai.completion",
)
ID_KEYS = ("gen_ai.tool.call.id", "tool_call.id", "tool.id")
NAME_KEYS = ("gen_ai.tool.name", "tool.name")
# OpenInference records the calls a model proposed on its LLM spans, both as
# the model's output and as the next request's input.
PROPOSED_ID_KEY = re.compile(
    r"llm\.(?:input|output)_messages\.\d+\.message\.tool_calls\.\d+\.tool_call\.id"
)
PROPOSED_OUTPUT_KEY = re.compile(r"llm\.output_messages\.\d+\.message\.tool_calls\.")
MODEL_OPERATIONS = ("chat", "text_completion", "generate_content")
# How a recorded model output names a proposed call: OpenAI and LangChain
# ``tool_calls``, Gemini ``function_call``, and typed content parts (GenAI
# semconv ``tool_call``, Anthropic ``tool_use``, OpenAI Responses ``function_call``).
CALL_KEYS = ("tool_calls", "tool_call", "function_call")
CALL_TYPES = ("tool_call", "tool_use", "function_call")


def _json(raw: str | None) -> object:
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return None


def _tool_message_call_id(raw: str | None) -> str | None:
    """The call id in a serialized LangChain ToolMessage (OpenInference output)."""
    message = _json(raw)
    if not isinstance(message, dict) or message.get("type") != "tool":
        return None
    data = message.get("data")
    call_id = data.get("tool_call_id") if isinstance(data, dict) else None
    return call_id if isinstance(call_id, str) and call_id else None


def proposed_arguments(spans: list[dict[str, Any]]) -> dict[str, str]:
    """Arguments the model proposed for each tool call id, as raw JSON text.

    LangChain passes a single-argument tool its bare value and OpenInference
    records only that, so the executed span loses the argument name; the
    model's proposed call for the same id still has it.
    """
    proposed: dict[str, str] = {}
    for span in spans:
        attrs = span.get("attributes") or []
        for attr in attrs:
            key = attr.get("key") or ""
            if not PROPOSED_ID_KEY.fullmatch(key):
                continue
            call_id = get_attr(attrs, key)
            arguments = get_attr(attrs, key[: -len("id")] + "function.arguments")
            if call_id and arguments is not None:
                proposed[call_id] = arguments
    return proposed


def _has_call(value: object) -> bool:
    if isinstance(value, dict):
        if value.get("type") in CALL_TYPES:
            return True
        return any(
            (key in CALL_KEYS and bool(item)) or _has_call(item)
            for key, item in value.items()
        )
    return isinstance(value, list) and any(_has_call(item) for item in value)


def _model_output(span: dict[str, Any]) -> str | None:
    """``proposed``, ``none``, or ``hidden`` (output not recorded) for a model
    call span; None for any other span."""
    attrs = span.get("attributes") or []
    if (first_attr(attrs, "openinference.span.kind") or "").upper() == "LLM":
        keys = [attr.get("key") or "" for attr in attrs]
        if not any(key.startswith("llm.output_messages.") for key in keys):
            return "hidden"
        return "proposed" if any(map(PROPOSED_OUTPUT_KEY.match, keys)) else "none"
    if (first_attr(attrs, "langsmith.span.kind") or "").lower() == "llm":
        raw = first_attr(attrs, "gen_ai.completion")
    elif span.get("name") == "call_llm":  # ADK; its tool spans carry llm_response too
        raw = first_attr(attrs, "gcp.vertex.agent.llm_response")
    elif first_attr(attrs, "gen_ai.operation.name") in MODEL_OPERATIONS:
        raw = first_attr(attrs, "gen_ai.output.messages")
    else:
        return None
    output = _json(raw)
    if not output:  # absent, not JSON, or emptied by content capture being off
        return "hidden"
    return "proposed" if _has_call(output) else "none"


def proves_no_tool_calls(spans: list[dict[str, Any]]) -> bool:
    """Whether the model spans show every response and none proposed a tool call.

    A tool runs because a model proposed it, so a turn whose model outputs are
    all recorded, none proposing a call, made no tool call even though no tool
    span says so. An unrecorded output is covered by a recorded one on its
    parent or child span (ADK wraps the provider's ``generate_content`` span in
    ``call_llm``).
    """
    outputs: dict[object, str] = {}
    parents: dict[object, object] = {}
    for span in spans:
        output = _model_output(span)
        if output is not None:
            span_id = span.get("spanId", span.get("span_id"))
            outputs[span_id] = output
            parents[span_id] = span.get("parentSpanId", span.get("parent_span_id"))
    recorded = {s for s, output in outputs.items() if output != "hidden"}
    if not recorded or "proposed" in outputs.values():
        return False
    return all(
        output != "hidden"
        or parents[s] in recorded
        or any(parents[r] == s for r in recorded)
        for s, output in outputs.items()
    )


@dataclass
class ToolSpanFields:
    """Raw fields of a recognized tool span; values are unparsed attribute text."""

    name: str
    call_id: str | None
    arguments: str | None
    result: str | None
    error: str | None


def _is_tool_span(name: str, attrs: list[dict[str, Any]]) -> bool:
    op = first_attr(attrs, "gen_ai.operation.name")
    kind = first_attr(attrs, "openinference.span.kind") or ""
    ls_kind = first_attr(attrs, "langsmith.span.kind") or ""
    if op == "execute_tool" or kind.upper() == "TOOL" or ls_kind.lower() == "tool":
        return True
    if op or kind or ls_kind:
        return False  # declared as chat / LLM / chain / retriever etc.
    if name.startswith("execute_tool "):
        return True
    if (
        name == "tool_call"
        and first_attr(attrs, "gcp.vertex.agent.tool_call_args") is not None
    ):
        return True
    return first_attr(attrs, *NAME_KEYS) is not None


def _tool_name(name: str, attrs: list[dict[str, Any]]) -> str | None:
    tool = first_attr(attrs, *NAME_KEYS)
    if tool:
        return tool
    if name.startswith("execute_tool "):
        return name[len("execute_tool ") :] or None
    return name or None


def _span_error(span: dict[str, Any], attrs: list[dict[str, Any]]) -> str | None:
    status = span.get("status") or {}
    for event in span.get("events") or []:
        if event.get("name") == "exception":
            ev_attrs = event.get("attributes") or []
            message = first_attr(ev_attrs, "exception.message") or ""
            kind = first_attr(ev_attrs, "exception.type")
            text = f"{kind}: {message}" if kind else message
            return text or status.get("message") or "unknown error"
    if status.get("code") in (2, "2", "STATUS_CODE_ERROR"):
        return (
            status.get("message") or first_attr(attrs, "error.type") or "unknown error"
        )
    return None


def tool_span_fields(
    span: dict[str, Any], proposed: dict[str, str] | None = None
) -> ToolSpanFields | None:
    """Recognize a tool execution span and read its raw fields, or None.

    ``proposed`` (from ``proposed_arguments`` over the same turn's spans)
    replaces arguments that are not a JSON object.
    """
    attrs = span.get("attributes") or []
    name = span.get("name") or ""
    tool = _tool_name(name, attrs) if _is_tool_span(name, attrs) else None
    if not tool or tool == "(merged tools)":  # ADK wrapper around parallel calls
        return None
    result = first_attr(attrs, *RESULT_KEYS)
    call_id = first_attr(attrs, *ID_KEYS) or _tool_message_call_id(result)
    arguments = first_attr(attrs, *ARGUMENT_KEYS)
    if proposed and call_id in proposed and not isinstance(_json(arguments), dict):
        arguments = proposed[call_id]
    return ToolSpanFields(
        name=tool,
        call_id=call_id,
        arguments=arguments,
        result=result,
        error=_span_error(span, attrs),
    )


def span_to_tool_call(
    span: dict[str, Any], proposed: dict[str, str] | None = None
) -> ToolCall | None:
    """Convert a single OTLP span dict to a ToolCall, or None if not a tool span."""
    fields = tool_span_fields(span, proposed)
    if fields is None:
        return None
    return ToolCall(
        id=fields.call_id or span.get("spanId", span.get("span_id", "")),
        name=fields.name,
        arguments=_parse_arguments(fields.arguments, span_name=span.get("name", "")),
        result=fields.result,
        error=fields.error,
        source=ToolCallSource.OTEL_TRACE,
    )


def spans_to_tool_calls(spans: list[dict[str, Any]]) -> list[ToolCall]:
    """Convert a list of OTLP span dicts to ToolCall objects.

    Spans without recognized tool attributes are silently skipped.
    """
    proposed = proposed_arguments(spans)
    tool_calls: list[ToolCall] = []
    for span in spans:
        tc = span_to_tool_call(span, proposed)
        if tc is not None:
            tool_calls.append(tc)
    return tool_calls
