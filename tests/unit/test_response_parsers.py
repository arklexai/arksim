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
from arksim.simulation_engine.tool_types import ToolCallSource


class TestParseOpenAI:
    def test_text_only(self) -> None:
        result = {"choices": [{"message": {"role": "assistant", "content": "Hello!"}}]}
        response = parse_openai(result)
        assert response.content == "Hello!"
        assert response.tool_calls == []

    def test_tool_calls_only(self) -> None:
        """content=None with tool_calls captured as-is; content falls back to ""."""
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
        assert tc.id == "call_1"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "NYC"}
        assert tc.result is None
        assert tc.source == ToolCallSource.CHAT_COMPLETIONS

    def test_text_and_tool_calls(self) -> None:
        """Both content and tool_calls captured."""
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
        assert response.tool_calls[0].arguments == {"q": "test"}

    def test_arguments_as_dict(self) -> None:
        """Some OpenAI-compatible routers (LiteLLM, vLLM) return arguments as a dict."""
        result = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_dict",
                                "type": "function",
                                "function": {
                                    "name": "f",
                                    "arguments": {"k": "v"},
                                },
                            }
                        ],
                    }
                }
            ]
        }
        response = parse_openai(result)
        assert response.tool_calls[0].arguments == {"k": "v"}

    def test_arguments_none(self) -> None:
        result = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_none",
                                "type": "function",
                                "function": {"name": "f", "arguments": None},
                            }
                        ],
                    }
                }
            ]
        }
        response = parse_openai(result)
        assert response.tool_calls[0].arguments == {}

    def test_arguments_malformed_json(self) -> None:
        """Malformed JSON string falls back to empty dict; call still captured."""
        result = {
            "choices": [
                {
                    "message": {
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
        assert response.tool_calls[0].name == "bad_tool"
        assert response.tool_calls[0].arguments == {}

    def test_multiple_tool_calls(self) -> None:
        result = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "a", "arguments": "{}"},
                            },
                            {
                                "id": "c2",
                                "type": "function",
                                "function": {"name": "b", "arguments": "{}"},
                            },
                        ],
                    }
                }
            ]
        }
        response = parse_openai(result)
        assert [tc.name for tc in response.tool_calls] == ["a", "b"]

    def test_missing_name_skipped(self) -> None:
        """Tool call entries without a name are skipped."""
        result = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"arguments": "{}"},
                            },
                            {
                                "id": "c2",
                                "type": "function",
                                "function": {"name": "ok", "arguments": "{}"},
                            },
                        ],
                    }
                }
            ]
        }
        response = parse_openai(result)
        assert [tc.name for tc in response.tool_calls] == ["ok"]

    def test_non_string_name_skipped(self) -> None:
        result = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": 42, "arguments": "{}"},
                            }
                        ],
                    }
                }
            ]
        }
        response = parse_openai(result)
        assert response.tool_calls == []

    def test_missing_id_defaults_to_empty_string(self) -> None:
        """Matches the A2A convention; OpenAI spec requires id but be defensive."""
        result = {
            "choices": [
                {
                    "message": {
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
        assert response.tool_calls[0].id == ""
        assert response.tool_calls[0].name == "f"

    def test_empty_tool_calls_list(self) -> None:
        result = {"choices": [{"message": {"content": "hi", "tool_calls": []}}]}
        response = parse_openai(result)
        assert response.content == "hi"
        assert response.tool_calls == []

    def test_empty_choices_raises(self) -> None:
        with pytest.raises(ValueError, match="empty 'choices'"):
            parse_openai({"choices": []})


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
        assert tc.id == "toolu_01"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "NYC"}
        assert tc.source == ToolCallSource.CHAT_COMPLETIONS

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
        assert response.tool_calls[0].name == "search"

    def test_multiple_tool_use_blocks(self) -> None:
        result = {
            "content": [
                {"type": "tool_use", "id": "t1", "name": "a", "input": {}},
                {"type": "tool_use", "id": "t2", "name": "b", "input": {}},
            ]
        }
        response = parse_anthropic(result)
        assert [tc.name for tc in response.tool_calls] == ["a", "b"]

    def test_tool_use_missing_name_skipped(self) -> None:
        result = {
            "content": [
                {"type": "tool_use", "id": "t1", "input": {}},
                {"type": "tool_use", "id": "t2", "name": "ok", "input": {}},
            ]
        }
        response = parse_anthropic(result)
        assert [tc.name for tc in response.tool_calls] == ["ok"]

    def test_tool_use_input_not_dict(self) -> None:
        """Defensive: if input isn't a dict, fall back to empty dict."""
        result = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "f",
                    "input": "not-a-dict",
                }
            ]
        }
        response = parse_anthropic(result)
        assert response.tool_calls[0].arguments == {}

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
        """Function call parts in response are not extracted; only text is."""
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
        assert response.tool_calls == []

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

    def test_tool_calls_extracted_via_dispatch(self) -> None:
        """parse_response delegates tool call extraction to the provider parser."""
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
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "f"

    def test_openai_takes_precedence_over_anthropic(self) -> None:
        """A response with both 'choices' and 'content' is parsed as OpenAI."""
        result = {
            "choices": [{"message": {"content": "from choices"}}],
            "content": [{"type": "text", "text": "from content"}],
        }
        response = parse_response(result)
        assert response.content == "from choices"
