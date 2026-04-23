# SPDX-License-Identifier: Apache-2.0
"""Response parsers for Chat Completions API responses.

Auto-detects provider format and normalizes into AgentResponse.
This is the single boundary where raw API responses enter arksim's
type system.

Supported formats:
- OpenAI Chat Completions (choices[].message)
- Anthropic Messages API (content[] blocks)
- Google Gemini (candidates[].content.parts)

To add a new format: add a parse function and a detection branch in
parse_response(). If your endpoint returns a non-standard format,
wrap it in OpenAI format, use the custom connector, or use OTel.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from arksim.simulation_engine.tool_types import AgentResponse, ToolCall, ToolCallSource

logger = logging.getLogger(__name__)


def _coerce_openai_arguments(raw: object) -> dict[str, Any]:
    """Normalize OpenAI tool call ``arguments`` into a dict.

    Spec-compliant OpenAI returns a JSON string, but LiteLLM, vLLM, and
    some Azure routers return a dict. None and other types fall back
    to an empty dict with a debug log.
    """
    if isinstance(raw, dict):
        return raw
    if raw is None:
        logger.debug("Tool call arguments is None; using empty dict")
        return {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug(
                "OpenAI tool call arguments failed JSON decode; using empty dict"
            )
            return {}
        if isinstance(parsed, dict):
            return parsed
        logger.debug(
            "OpenAI tool call arguments JSON decoded to %s, not dict; using empty dict",
            type(parsed).__name__,
        )
        return {}
    logger.debug("Tool call arguments has unexpected type %s", type(raw).__name__)
    return {}


def _extract_openai_tool_calls(msg: dict[str, Any]) -> list[ToolCall]:
    """Extract tool calls from an OpenAI chat message dict.

    Skips entries missing a string ``function.name``. Each captured call
    carries ``source=ToolCallSource.CHAT_COMPLETIONS``. ``result`` is left
    as ``None`` (not available on this path — see spec).

    The entry-level ``type`` field (e.g. ``"function"``) is intentionally
    ignored. OpenAI may add new ``type`` values (code interpreter, computer
    use) in the future; those will be silently skipped until explicitly
    modeled.
    """
    raw_calls = msg.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        return []
    tool_calls: list[ToolCall] = []
    for entry in raw_calls:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            logger.debug(
                "Skipping OpenAI tool call: missing or non-string function.name"
            )
            continue
        call_id = entry.get("id")
        tool_calls.append(
            ToolCall(
                # Empty string matches A2A convention for missing id. OpenAI spec
                # requires a string id; we fall back defensively.
                id=call_id if isinstance(call_id, str) else "",
                name=name,
                arguments=_coerce_openai_arguments(fn.get("arguments")),
                result=None,
                source=ToolCallSource.CHAT_COMPLETIONS,
            )
        )
    return tool_calls


def parse_response(result: dict[str, Any]) -> AgentResponse:
    """Auto-detect provider format and parse into AgentResponse.

    Detection precedence (first match wins):
    1. "choices" key -> OpenAI Chat Completions
    2. "content" as list -> Anthropic Messages API
    3. "candidates" key -> Google Gemini
    """
    if "choices" in result:
        return parse_openai(result)
    if "content" in result and isinstance(result["content"], list):
        return parse_anthropic(result)
    if "candidates" in result:
        return parse_gemini(result)
    raise ValueError(
        "Unsupported response format. Supported: OpenAI (choices), "
        "Anthropic (content list), Gemini (candidates). "
        f"Keys present: {list(result.keys())}. "
        "Wrap your endpoint in OpenAI format, use the custom connector, "
        "or use OTel trace correlation."
    )


def parse_openai(result: dict[str, Any]) -> AgentResponse:
    """Parse OpenAI Chat Completions format."""
    choices = result.get("choices", [])
    if not choices:
        raise ValueError("API response has empty 'choices'")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError(
            "API response 'choices[0]' must be a dict, got "
            f"{type(first_choice).__name__}"
        )

    msg = first_choice.get("message") or first_choice.get("delta") or {}
    raw_content = msg.get("content")
    content = raw_content if isinstance(raw_content, str) else ""

    return AgentResponse(
        content=content,
        tool_calls=_extract_openai_tool_calls(msg),
    )


def _extract_anthropic_tool_calls(blocks: list[Any]) -> list[ToolCall]:
    """Extract tool calls from an Anthropic ``content[]`` block list.

    Skips entries missing a string ``name``. ``input`` is already a dict
    in spec-compliant responses; non-dict values fall back to empty dict.

    Evaluation-irrelevant fields on ``tool_use`` blocks (``cache_control``,
    extended-thinking metadata) are intentionally not mapped.
    """
    tool_calls: list[ToolCall] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_use":
            continue
        name = block.get("name")
        if not isinstance(name, str) or not name:
            logger.debug(
                "Skipping Anthropic tool_use block: missing or non-string name"
            )
            continue
        raw_input = block.get("input")
        arguments = raw_input if isinstance(raw_input, dict) else {}
        block_id = block.get("id")
        tool_calls.append(
            ToolCall(
                # Empty string matches A2A convention for missing id.
                id=block_id if isinstance(block_id, str) else "",
                name=name,
                arguments=arguments,
                result=None,
                source=ToolCallSource.CHAT_COMPLETIONS,
            )
        )
    return tool_calls


def parse_anthropic(result: dict[str, Any]) -> AgentResponse:
    """Parse Anthropic Messages API format."""
    blocks = result.get("content", [])
    # Defensive for direct callers; parse_response already checks this.
    if not isinstance(blocks, list):
        blocks = []
    text_parts: list[str] = []

    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))

    return AgentResponse(
        content="".join(text_parts),
        tool_calls=_extract_anthropic_tool_calls(blocks),
    )


def _extract_gemini_tool_calls(parts: list[Any]) -> list[ToolCall]:
    """Extract tool calls from a Gemini ``content.parts[]`` list.

    Skips entries missing a string ``name``. Gemini has no per-call id
    field; id defaults to ``""`` (matches the A2A convention).
    """
    tool_calls: list[ToolCall] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        fn = part.get("functionCall")
        if not isinstance(fn, dict):
            # Debug hint for Gemini features we do not yet model
            # (executableCode, codeExecutionResult, etc).
            if any(k in part for k in ("executableCode", "codeExecutionResult")):
                logger.debug(
                    "Skipping unmodeled Gemini part type: %s", ",".join(part.keys())
                )
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            logger.debug(
                "Skipping Gemini functionCall part: missing or non-string name"
            )
            continue
        raw_args = fn.get("args")
        arguments = raw_args if isinstance(raw_args, dict) else {}
        tool_calls.append(
            ToolCall(
                # Gemini has no per-call id; empty string matches A2A convention.
                id="",
                name=name,
                arguments=arguments,
                result=None,
                source=ToolCallSource.CHAT_COMPLETIONS,
            )
        )
    return tool_calls


def parse_gemini(result: dict[str, Any]) -> AgentResponse:
    """Parse Google Gemini API format."""
    candidates = result.get("candidates", [])
    if not candidates:
        raise ValueError("API response has empty 'candidates'")

    content_obj = candidates[0].get("content", {})
    parts = content_obj.get("parts", []) if isinstance(content_obj, dict) else []
    if not isinstance(parts, list):
        parts = []

    text_parts: list[str] = []

    for part in parts:
        if not isinstance(part, dict):
            continue
        if "text" in part:
            text_parts.append(part["text"])

    return AgentResponse(
        content="".join(text_parts),
        tool_calls=_extract_gemini_tool_calls(parts),
    )


__all__ = ["parse_response", "parse_openai", "parse_anthropic", "parse_gemini"]
