# SPDX-License-Identifier: Apache-2.0
"""Tests for Chat Completions response parsers."""

from __future__ import annotations

import pytest

from arksim.simulation_engine.agent.response_parsers import (
    parse_anthropic,
    parse_gemini,
    parse_openai,
    parse_response,
)


class TestParseOpenAI:
    def test_text_only(self) -> None:
        result = {"choices": [{"message": {"role": "assistant", "content": "Hello!"}}]}
        response = parse_openai(result)
        assert response.content == "Hello!"
        assert response.tool_calls == []

    def test_tool_calls_only(self) -> None:
        """content=None with tool_calls: the crash that this PR fixes."""
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
                                    "arguments": '{"city": "NYC"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        response = parse_openai(result)
        assert response.content == ""
        assert len(response.tool_calls) == 1
        tc = response.tool_calls[0]
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "NYC"}
        assert tc.source == "response_parse"

    def test_text_and_tool_calls(self) -> None:
        result = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Let me check.",
                        "tool_calls": [
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q": "test"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        response = parse_openai(result)
        assert response.content == "Let me check."
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search"

    def test_empty_choices_raises(self) -> None:
        with pytest.raises(ValueError, match="empty 'choices'"):
            parse_openai({"choices": []})

    def test_malformed_arguments_logs_and_skips(self) -> None:
        """Invalid JSON in arguments logs a warning and leaves arguments empty."""
        result = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_bad",
                                "type": "function",
                                "function": {
                                    "name": "bad_tool",
                                    "arguments": "not-json",
                                },
                            }
                        ],
                    }
                }
            ]
        }
        response = parse_openai(result)
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].arguments == {}

    def test_tool_call_id_generated_when_missing(self) -> None:
        """Missing 'id' on a tool_call produces a UUID rather than raising."""
        result = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ]
        }
        response = parse_openai(result)
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id  # non-empty UUID string


class TestParseAnthropic:
    def test_text_blocks(self) -> None:
        result = {"content": [{"type": "text", "text": "Hello from Claude."}]}
        response = parse_anthropic(result)
        assert response.content == "Hello from Claude."
        assert response.tool_calls == []

    def test_tool_use_blocks(self) -> None:
        result = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "get_weather",
                    "input": {"city": "NYC"},
                }
            ]
        }
        response = parse_anthropic(result)
        assert response.content == ""
        assert len(response.tool_calls) == 1
        tc = response.tool_calls[0]
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "NYC"}
        assert tc.source == "response_parse"

    def test_mixed_text_and_tool_use(self) -> None:
        result = {
            "content": [
                {"type": "text", "text": "Let me check. "},
                {
                    "type": "tool_use",
                    "id": "toolu_02",
                    "name": "search",
                    "input": {"q": "test"},
                },
            ]
        }
        response = parse_anthropic(result)
        assert response.content == "Let me check. "
        assert len(response.tool_calls) == 1

    def test_non_dict_blocks_ignored(self) -> None:
        """Malformed non-dict entries in content list are silently skipped."""
        result = {"content": ["not-a-dict", {"type": "text", "text": "ok"}]}
        response = parse_anthropic(result)
        assert response.content == "ok"

    def test_unknown_block_types_ignored(self) -> None:
        result = {
            "content": [
                {"type": "image", "source": {}},
                {"type": "text", "text": "text only"},
            ]
        }
        response = parse_anthropic(result)
        assert response.content == "text only"
        assert response.tool_calls == []


class TestParseGemini:
    def test_text_parts(self) -> None:
        result = {
            "candidates": [{"content": {"parts": [{"text": "Hello from Gemini."}]}}]
        }
        response = parse_gemini(result)
        assert response.content == "Hello from Gemini."
        assert response.tool_calls == []

    def test_function_call_parts(self) -> None:
        result = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"city": "NYC"},
                                }
                            }
                        ]
                    }
                }
            ]
        }
        response = parse_gemini(result)
        assert response.content == ""
        assert len(response.tool_calls) == 1
        tc = response.tool_calls[0]
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "NYC"}
        assert tc.source == "response_parse"

    def test_empty_candidates_raises(self) -> None:
        with pytest.raises(ValueError, match="empty 'candidates'"):
            parse_gemini({"candidates": []})

    def test_multiple_text_parts_concatenated(self) -> None:
        result = {
            "candidates": [
                {"content": {"parts": [{"text": "Hello"}, {"text": " world"}]}}
            ]
        }
        response = parse_gemini(result)
        assert response.content == "Hello world"


class TestParseResponseDispatch:
    def test_dispatches_openai(self) -> None:
        result = {"choices": [{"message": {"content": "hi"}}]}
        response = parse_response(result)
        assert response.content == "hi"

    def test_dispatches_anthropic(self) -> None:
        result = {"content": [{"type": "text", "text": "hi"}]}
        response = parse_response(result)
        assert response.content == "hi"

    def test_dispatches_gemini(self) -> None:
        result = {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}
        response = parse_response(result)
        assert response.content == "hi"

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported response format"):
            parse_response({"data": "something"})

    def test_source_field_set(self) -> None:
        result = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ]
        }
        response = parse_response(result)
        assert response.tool_calls[0].source == "response_parse"

    def test_openai_takes_precedence_over_anthropic(self) -> None:
        """A response with both 'choices' and 'content' is parsed as OpenAI."""
        result = {
            "choices": [{"message": {"content": "from choices"}}],
            "content": [{"type": "text", "text": "from content"}],
        }
        response = parse_response(result)
        assert response.content == "from choices"
