# SPDX-License-Identifier: Apache-2.0
"""Tests for chat completions response parsing via parse_response."""

from __future__ import annotations

import pytest

from arksim.simulation_engine.agent.response_parsers import parse_response
from arksim.simulation_engine.tool_types import AgentResponse


class TestParseResponseOpenAI:
    """Tests for OpenAI-style response format."""

    def test_message_content(self) -> None:
        result = {
            "choices": [{"message": {"role": "assistant", "content": "Hello, world!"}}]
        }
        response = parse_response(result)
        assert isinstance(response, AgentResponse)
        assert response.content == "Hello, world!"

    def test_delta_content(self) -> None:
        result = {"choices": [{"delta": {"content": "Streamed "}}]}
        response = parse_response(result)
        assert response.content == "Streamed "

    def test_empty_choices_raises(self) -> None:
        result = {"choices": []}
        with pytest.raises(ValueError, match="empty 'choices'"):
            parse_response(result)

    def test_choice_without_message_or_delta_returns_empty(self) -> None:
        # A choice with no message/delta key yields empty content (no tool calls).
        result = {"choices": [{"finish_reason": "stop"}]}
        response = parse_response(result)
        assert response.content == ""
        assert response.tool_calls == []

    def test_tool_calls_parsed(self) -> None:
        result = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"query": "arksim"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        response = parse_response(result)
        assert response.content == ""
        assert len(response.tool_calls) == 1
        tc = response.tool_calls[0]
        assert tc.id == "call_abc"
        assert tc.name == "search"
        assert tc.arguments == {"query": "arksim"}


class TestParseResponseAnthropic:
    """Tests for Anthropic-style response format."""

    def test_single_text_block(self) -> None:
        result = {"content": [{"type": "text", "text": "Anthropic reply."}]}
        response = parse_response(result)
        assert isinstance(response, AgentResponse)
        assert response.content == "Anthropic reply."

    def test_multiple_text_blocks(self) -> None:
        result = {
            "content": [
                {"type": "text", "text": "Part one. "},
                {"type": "text", "text": "Part two."},
            ]
        }
        response = parse_response(result)
        assert response.content == "Part one. Part two."

    def test_ignores_non_text_blocks(self) -> None:
        result = {
            "content": [
                {"type": "image", "source": {}},
                {"type": "text", "text": "Only this."},
            ]
        }
        response = parse_response(result)
        assert response.content == "Only this."

    def test_tool_use_block_parsed(self) -> None:
        result = {
            "content": [
                {"type": "text", "text": "Using tool."},
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "calculator",
                    "input": {"expression": "2+2"},
                },
            ]
        }
        response = parse_response(result)
        assert response.content == "Using tool."
        assert len(response.tool_calls) == 1
        tc = response.tool_calls[0]
        assert tc.id == "toolu_01"
        assert tc.name == "calculator"
        assert tc.arguments == {"expression": "2+2"}


class TestParseResponseGoogle:
    """Tests for Google Gemini-style response format."""

    def test_single_part(self) -> None:
        result = {
            "candidates": [{"content": {"parts": [{"text": "Google response."}]}}]
        }
        response = parse_response(result)
        assert isinstance(response, AgentResponse)
        assert response.content == "Google response."

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
        response = parse_response(result)
        assert response.content == "ABC"

    def test_empty_candidates_raises(self) -> None:
        result = {"candidates": []}
        with pytest.raises(ValueError, match="empty 'candidates'"):
            parse_response(result)

    def test_function_call_part_parsed(self) -> None:
        result = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Calling tool."},
                            {
                                "functionCall": {
                                    "name": "weather",
                                    "args": {"city": "NYC"},
                                }
                            },
                        ]
                    }
                }
            ]
        }
        response = parse_response(result)
        assert response.content == "Calling tool."
        assert len(response.tool_calls) == 1
        tc = response.tool_calls[0]
        assert tc.name == "weather"
        assert tc.arguments == {"city": "NYC"}


class TestParseResponseUnsupported:
    """Tests for unsupported or invalid response formats."""

    def test_empty_dict_raises(self) -> None:
        result: dict = {}
        with pytest.raises(ValueError, match="Unsupported response format"):
            parse_response(result)

    def test_unknown_keys_in_error_message(self) -> None:
        result = {"id": "xyz", "usage": {}}
        with pytest.raises(ValueError, match="Keys present"):
            parse_response(result)
