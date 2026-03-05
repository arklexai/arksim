# SPDX-License-Identifier: Apache-2.0
"""Helpers for detecting and dispatching on LLM provider
response formats.

Every function takes a pre-detected ``ResponseFormat`` so
that format sniffing happens exactly once per API response
(in ``detect_format``).  All other functions trust the
``fmt`` argument and access provider-specific keys directly.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

__all__ = [
    "ResponseFormat",
    "build_assistant_tool_message",
    "build_tool_results",
    "detect_format",
    "extract_content",
    "extract_tool_calls",
]


class ResponseFormat(Enum):
    """Detected provider format of an API response."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


def detect_format(result: dict[str, Any]) -> ResponseFormat:
    """Detect provider format from a raw API response.

    Called once per response; the returned enum is then
    passed to every other helper in this module.

    Raises:
        ValueError: If the response does not match any
            known provider format.
    """
    if "choices" in result:
        return ResponseFormat.OPENAI
    if "candidates" in result:
        return ResponseFormat.GEMINI
    if "content" in result and isinstance(result["content"], list):
        return ResponseFormat.ANTHROPIC
    raise ValueError(
        f"Unsupported response format. Keys present: {list(result.keys())}"
    )


# ── Tool-call extraction ─────────────────────────────────


def extract_tool_calls(
    fmt: ResponseFormat, result: dict[str, Any]
) -> list[dict[str, Any]] | None:
    """Return tool-call objects from *result*, or ``None``
    if the response is text-only.
    """
    if fmt is ResponseFormat.OPENAI:
        msg = result["choices"][0].get("message", {})
        return msg.get("tool_calls") or None

    if fmt is ResponseFormat.ANTHROPIC:
        blocks = [
            b
            for b in result["content"]
            if isinstance(b, dict) and b.get("type") == "tool_use"
        ]
        return blocks or None

    # Gemini
    parts = (result["candidates"][0].get("content") or {}).get("parts") or []
    fc = [p for p in parts if isinstance(p, dict) and "functionCall" in p]
    return fc or None


# ── Text content extraction ──────────────────────────────


def _extract_openai_content(result: dict[str, Any]) -> str:
    choices = result["choices"]
    if not choices:
        raise ValueError("API response has empty 'choices'")
    choice = choices[0]
    msg = choice.get("message") or choice.get("delta")
    if not msg:
        raise ValueError("API response choice has no 'message' or 'delta'")
    content = msg.get("content")
    if content is None and "delta" in choice:
        content = choice["delta"].get("content")
    if content is not None:
        return content if isinstance(content, str) else str(content)
    raise ValueError("OpenAI response has no text content")


def _extract_anthropic_content(result: dict[str, Any]) -> str:
    texts = [
        b.get("text", "")
        for b in result["content"]
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    if texts:
        return "".join(texts)
    raise ValueError("Anthropic response has no text content blocks")


def _extract_gemini_content(result: dict[str, Any]) -> str:
    candidates = result["candidates"]
    if not candidates:
        raise ValueError("API response has empty 'candidates'")
    content = candidates[0].get("content")
    if content and isinstance(content, dict):
        parts = content.get("parts") or []
        text_parts = [p.get("text", "") if isinstance(p, dict) else "" for p in parts]
        joined = "".join(text_parts)
        if joined:
            return joined
    raise ValueError("Gemini response has no text content")


_CONTENT_EXTRACTORS = {
    ResponseFormat.OPENAI: _extract_openai_content,
    ResponseFormat.ANTHROPIC: _extract_anthropic_content,
    ResponseFormat.GEMINI: _extract_gemini_content,
}


def extract_content(fmt: ResponseFormat, result: dict[str, Any]) -> str:
    """Extract assistant text from an API response.

    Raises:
        ValueError: If the response contains no
            extractable text content.
    """
    return _CONTENT_EXTRACTORS[fmt](result)


# ── Conversation-history message builders ─────────────────


def build_assistant_tool_message(
    fmt: ResponseFormat, result: dict[str, Any]
) -> dict[str, Any]:
    """Build the assistant message to append when the model
    requested tool calls.
    """
    if fmt is ResponseFormat.OPENAI:
        msg = result["choices"][0]["message"]
        return {
            "role": "assistant",
            "content": msg.get("content"),
            "tool_calls": msg["tool_calls"],
        }

    if fmt is ResponseFormat.GEMINI:
        content = result["candidates"][0].get("content", {})
        return {
            "role": "model",
            "parts": content.get("parts", []),
        }

    # Anthropic
    return {"role": "assistant", "content": result["content"]}


def build_tool_results(
    fmt: ResponseFormat,
    tool_calls: list[dict[str, Any]],
    tool_call_result: str,
) -> list[dict[str, Any]]:
    """Build synthetic tool-result messages to feed back
    to the model after tool calls are resolved.
    """
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
                            "response": {
                                "result": tool_call_result,
                            },
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
