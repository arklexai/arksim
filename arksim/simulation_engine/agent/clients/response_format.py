# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from enum import Enum
from typing import Any


class ResponseFormat(Enum):
    """Detected provider format of an API response."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


def detect_format(result: dict[str, Any]) -> ResponseFormat:
    """Detect provider format from a raw API response. Called once per response."""
    if "choices" in result:
        return ResponseFormat.OPENAI
    if "candidates" in result:
        return ResponseFormat.GEMINI
    if "content" in result and isinstance(result["content"], list):
        return ResponseFormat.ANTHROPIC
    raise ValueError(
        f"Unsupported response format. Keys present: {list(result.keys())}"
    )


def extract_tool_calls(
    fmt: ResponseFormat, result: dict[str, Any]
) -> list[dict[str, Any]] | None:
    """Extract tool calls from response, or ``None`` if text-only."""
    if fmt is ResponseFormat.OPENAI:
        msg = (result.get("choices") or [{}])[0].get("message", {})
        tc = msg.get("tool_calls")
        if tc:
            return tc

    elif fmt is ResponseFormat.ANTHROPIC:
        tool_use_blocks = [
            block
            for block in result["content"]
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
        if tool_use_blocks:
            return tool_use_blocks

    elif fmt is ResponseFormat.GEMINI:
        parts = ((result.get("candidates") or [{}])[0].get("content") or {}).get(
            "parts"
        ) or []
        fc_parts = [p for p in parts if isinstance(p, dict) and "functionCall" in p]
        if fc_parts:
            return fc_parts

    return None


def extract_content(fmt: ResponseFormat, result: dict[str, Any]) -> str:
    """Extract assistant text from API response."""
    if fmt is ResponseFormat.OPENAI:
        choices = result["choices"]
        if not choices:
            raise ValueError("API response has empty 'choices'")
        msg = choices[0].get("message") or choices[0].get("delta")
        if not msg:
            raise ValueError("API response choice has no 'message' or 'delta'")
        content = msg.get("content")
        if content is None and "delta" in choices[0]:
            content = choices[0]["delta"].get("content")
        if content is not None:
            return content if isinstance(content, str) else str(content)

    elif fmt is ResponseFormat.ANTHROPIC:
        parts = []
        for block in result["content"]:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        if parts:
            return "".join(parts)

    elif fmt is ResponseFormat.GEMINI:
        candidates = result["candidates"]
        if not candidates:
            raise ValueError("API response has empty 'candidates'")
        content = candidates[0].get("content")
        if content and isinstance(content, dict):
            parts = content.get("parts") or []
            if isinstance(parts, list):
                text_parts = [
                    p.get("text", "") if isinstance(p, dict) else "" for p in parts
                ]
                return "".join(text_parts)

    raise ValueError(
        "Unsupported response format: expected 'choices' (OpenAI), "
        "'content' list (Anthropic), or 'candidates' (Gemini). "
        f"Keys present: {list(result.keys())}"
    )


def build_assistant_tool_message(
    fmt: ResponseFormat, result: dict[str, Any]
) -> dict[str, Any]:
    """Build the assistant message to append for a tool-call turn."""
    if fmt is ResponseFormat.OPENAI:
        msg = (result.get("choices") or [{}])[0].get("message", {})
        return {
            "role": "assistant",
            "content": msg.get("content"),
            "tool_calls": msg["tool_calls"],
        }

    if fmt is ResponseFormat.GEMINI:
        content = (result.get("candidates") or [{}])[0].get("content", {})
        return {"role": "model", "parts": content.get("parts", [])}

    # Anthropic
    return {"role": "assistant", "content": result["content"]}


def build_tool_results(
    fmt: ResponseFormat,
    tool_calls: list[dict[str, Any]],
    tool_call_result: str,
) -> list[dict[str, Any]]:
    """Build synthetic tool-result messages for each tool call."""
    if fmt is ResponseFormat.ANTHROPIC:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": tool_call_result,
                    }
                    for tc in tool_calls
                ],
            }
        ]

    if fmt is ResponseFormat.GEMINI:
        return [
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": tc["functionCall"]["name"],
                            "response": {"result": tool_call_result},
                        }
                    }
                    for tc in tool_calls
                ],
            }
        ]

    # OpenAI: one tool message per call
    return [
        {
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": tool_call_result,
        }
        for tc in tool_calls
    ]
