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
import uuid
from typing import Any

from arksim.simulation_engine.tool_types import AgentResponse, ToolCall

logger = logging.getLogger(__name__)


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

    msg = choices[0].get("message") or choices[0].get("delta") or {}
    raw_content = msg.get("content")
    content: str = raw_content if isinstance(raw_content, str) else (raw_content or "")
    if not isinstance(content, str):
        content = str(content)

    tool_calls: list[ToolCall] = []
    for tc in msg.get("tool_calls") or []:
        func = tc.get("function", {})
        arguments: dict[str, Any] = {}
        raw_args = func.get("arguments")
        if raw_args:
            try:
                parsed = json.loads(raw_args)
                if isinstance(parsed, dict):
                    arguments = parsed
            except (json.JSONDecodeError, TypeError):
                logger.warning("Malformed tool call arguments: %s", raw_args[:200])

        tool_calls.append(
            ToolCall(
                id=tc.get("id") or str(uuid.uuid4()),
                name=func.get("name", ""),
                arguments=arguments,
                source="response_parse",
            )
        )

    return AgentResponse(content=content, tool_calls=tool_calls)


def parse_anthropic(result: dict[str, Any]) -> AgentResponse:
    """Parse Anthropic Messages API format."""
    blocks = result.get("content", [])
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.get("id") or str(uuid.uuid4()),
                    name=block.get("name", ""),
                    arguments=block.get("input", {}),
                    source="response_parse",
                )
            )

    return AgentResponse(content="".join(text_parts), tool_calls=tool_calls)


def parse_gemini(result: dict[str, Any]) -> AgentResponse:
    """Parse Google Gemini API format."""
    candidates = result.get("candidates", [])
    if not candidates:
        raise ValueError("API response has empty 'candidates'")

    content_obj = candidates[0].get("content", {})
    parts = content_obj.get("parts", []) if isinstance(content_obj, dict) else []

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for part in parts:
        if not isinstance(part, dict):
            continue
        if "text" in part:
            text_parts.append(part["text"])
        elif "functionCall" in part:
            fc = part["functionCall"]
            tool_calls.append(
                ToolCall(
                    id=str(uuid.uuid4()),
                    name=fc.get("name", ""),
                    arguments=fc.get("args", {}),
                    source="response_parse",
                )
            )

    return AgentResponse(content="".join(text_parts), tool_calls=tool_calls)


__all__ = ["parse_response", "parse_openai", "parse_anthropic", "parse_gemini"]
