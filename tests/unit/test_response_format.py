# SPDX-License-Identifier: Apache-2.0
"""Tests for response format detection and extraction helpers."""

from __future__ import annotations

import pytest

from arksim.simulation_engine.agent.clients.response_format import (
    ResponseFormat,
    build_assistant_tool_message,
    build_tool_results,
    detect_format,
    extract_content,
    extract_tool_calls,
)

# ── Format detection ──


class TestDetectFormat:
    """Tests for detect_format auto-detection."""

    def test_openai(self) -> None:
        result = {"choices": [{"message": {"content": "hi"}}]}
        assert detect_format(result) is ResponseFormat.OPENAI

    def test_anthropic(self) -> None:
        result = {"content": [{"type": "text", "text": "hi"}]}
        assert detect_format(result) is ResponseFormat.ANTHROPIC

    def test_gemini(self) -> None:
        result = {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}
        assert detect_format(result) is ResponseFormat.GEMINI

    def test_unsupported_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported response format"):
            detect_format({"id": "xyz", "usage": {}})

    def test_empty_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported response format"):
            detect_format({})


# ── Content extraction ──


class TestExtractContentOpenAI:
    """Tests for OpenAI-style response format."""

    def test_message_content(self) -> None:
        result = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello, world!",
                    }
                }
            ]
        }
        assert extract_content(ResponseFormat.OPENAI, result) == "Hello, world!"

    def test_delta_content(self) -> None:
        result = {"choices": [{"delta": {"content": "Streamed "}}]}
        assert extract_content(ResponseFormat.OPENAI, result) == "Streamed "

    def test_empty_choices_raises(self) -> None:
        result = {"choices": []}
        with pytest.raises(ValueError, match="empty 'choices'"):
            extract_content(ResponseFormat.OPENAI, result)

    def test_choice_without_message_or_delta_raises(self) -> None:
        result = {"choices": [{"finish_reason": "stop"}]}
        with pytest.raises(ValueError, match="no 'message' or 'delta'"):
            extract_content(ResponseFormat.OPENAI, result)

    def test_null_content_raises(self) -> None:
        """Message exists but content is None (tool-call
        response passed to extract_content by mistake)."""
        result = {"choices": [{"message": {"role": "assistant", "content": None}}]}
        with pytest.raises(ValueError, match="no text content"):
            extract_content(ResponseFormat.OPENAI, result)


class TestExtractContentAnthropic:
    """Tests for Anthropic-style response format."""

    def test_single_text_block(self) -> None:
        result = {"content": [{"type": "text", "text": "Anthropic reply."}]}
        assert extract_content(ResponseFormat.ANTHROPIC, result) == "Anthropic reply."

    def test_multiple_text_blocks(self) -> None:
        result = {
            "content": [
                {"type": "text", "text": "Part one. "},
                {"type": "text", "text": "Part two."},
            ]
        }
        assert (
            extract_content(ResponseFormat.ANTHROPIC, result) == "Part one. Part two."
        )

    def test_ignores_non_text_blocks(self) -> None:
        result = {
            "content": [
                {"type": "image", "source": {}},
                {"type": "text", "text": "Only this."},
            ]
        }
        assert extract_content(ResponseFormat.ANTHROPIC, result) == "Only this."

    def test_no_text_blocks_raises(self) -> None:
        """All blocks are non-text (e.g. only tool_use)."""
        result = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "f",
                    "input": {},
                }
            ]
        }
        with pytest.raises(ValueError, match="no text content"):
            extract_content(ResponseFormat.ANTHROPIC, result)


class TestExtractContentGemini:
    """Tests for Gemini-style response format."""

    def test_single_part(self) -> None:
        result = {
            "candidates": [{"content": {"parts": [{"text": "Gemini response."}]}}]
        }
        assert extract_content(ResponseFormat.GEMINI, result) == "Gemini response."

    def test_multiple_parts(self) -> None:
        result = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "A"},
                            {"text": "B"},
                            {"text": "C"},
                        ]
                    }
                }
            ]
        }
        assert extract_content(ResponseFormat.GEMINI, result) == "ABC"

    def test_empty_candidates_raises(self) -> None:
        result = {"candidates": []}
        with pytest.raises(ValueError, match="empty 'candidates'"):
            extract_content(ResponseFormat.GEMINI, result)

    def test_no_text_parts_raises(self) -> None:
        """Parts contain only functionCall, no text."""
        result = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "f",
                                    "args": {},
                                }
                            }
                        ]
                    }
                }
            ]
        }
        with pytest.raises(ValueError, match="no text content"):
            extract_content(ResponseFormat.GEMINI, result)


# ── Tool-call detection ──


class TestExtractToolCalls:
    """Tests for extract_tool_calls."""

    def test_openai_tool_calls(self) -> None:
        result = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"NYC"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        tc = extract_tool_calls(ResponseFormat.OPENAI, result)
        assert tc is not None
        assert len(tc) == 1
        assert tc[0]["id"] == "call_1"

    def test_anthropic_tool_use(self) -> None:
        result = {
            "content": [
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "get_weather",
                    "input": {"city": "NYC"},
                },
            ]
        }
        tc = extract_tool_calls(ResponseFormat.ANTHROPIC, result)
        assert tc is not None
        assert len(tc) == 1
        assert tc[0]["id"] == "tu_1"

    def test_openai_text_only_returns_none(self) -> None:
        result = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Just text.",
                    }
                }
            ]
        }
        assert extract_tool_calls(ResponseFormat.OPENAI, result) is None

    def test_anthropic_text_only_returns_none(self) -> None:
        result = {"content": [{"type": "text", "text": "Just text."}]}
        assert extract_tool_calls(ResponseFormat.ANTHROPIC, result) is None

    def test_gemini_function_call(self) -> None:
        result = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"city": "NYC"},
                                }
                            }
                        ],
                    }
                }
            ]
        }
        tc = extract_tool_calls(ResponseFormat.GEMINI, result)
        assert tc is not None
        assert len(tc) == 1
        assert tc[0]["functionCall"]["name"] == "get_weather"

    def test_gemini_text_only_returns_none(self) -> None:
        result = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Just text."}],
                    }
                }
            ]
        }
        assert extract_tool_calls(ResponseFormat.GEMINI, result) is None


# ── Assistant tool message building ──


class TestBuildAssistantToolMessage:
    """Tests for build_assistant_tool_message."""

    def test_openai_format(self) -> None:
        result = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": "call_1", "type": "function"}],
                    }
                }
            ]
        }
        msg = build_assistant_tool_message(ResponseFormat.OPENAI, result)
        assert msg["role"] == "assistant"
        assert msg["tool_calls"] == [{"id": "call_1", "type": "function"}]

    def test_anthropic_format(self) -> None:
        result = {
            "content": [
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "f",
                    "input": {},
                },
            ]
        }
        msg = build_assistant_tool_message(ResponseFormat.ANTHROPIC, result)
        assert msg["role"] == "assistant"
        assert msg["content"] == result["content"]

    def test_gemini_format(self) -> None:
        result = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "f",
                                    "args": {},
                                }
                            }
                        ],
                    }
                }
            ]
        }
        msg = build_assistant_tool_message(ResponseFormat.GEMINI, result)
        assert msg["role"] == "model"
        assert "functionCall" in msg["parts"][0]


# ── Tool result building ──


class TestBuildToolResults:
    """Tests for build_tool_results."""

    def test_openai_format(self) -> None:
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            }
        ]
        results = build_tool_results(
            ResponseFormat.OPENAI,
            tool_calls,
            '{"status": "ok"}',
        )
        assert len(results) == 1
        assert results[0]["role"] == "tool"
        assert results[0]["tool_call_id"] == "call_1"
        assert results[0]["content"] == '{"status": "ok"}'

    def test_anthropic_format(self) -> None:
        tool_calls = [
            {
                "type": "tool_use",
                "id": "tu_1",
                "name": "f",
                "input": {},
            }
        ]
        results = build_tool_results(
            ResponseFormat.ANTHROPIC,
            tool_calls,
            '{"status": "ok"}',
        )
        assert len(results) == 1
        assert results[0]["role"] == "user"
        blocks = results[0]["content"]
        assert len(blocks) == 1
        assert blocks[0]["type"] == "tool_result"
        assert blocks[0]["tool_use_id"] == "tu_1"

    def test_gemini_format(self) -> None:
        tool_calls = [
            {
                "functionCall": {
                    "name": "get_weather",
                    "args": {"city": "NYC"},
                }
            }
        ]
        results = build_tool_results(
            ResponseFormat.GEMINI,
            tool_calls,
            '{"status": "ok"}',
        )
        assert len(results) == 1
        assert results[0]["role"] == "user"
        parts = results[0]["parts"]
        assert len(parts) == 1
        assert parts[0]["functionResponse"]["name"] == "get_weather"
        assert parts[0]["functionResponse"]["response"]["result"] == '{"status": "ok"}'
