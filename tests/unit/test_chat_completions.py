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


class TestChatCompletionsAgentExecute:
    """End-to-end: execute() flows captured tool calls into AgentResponse."""

    @pytest.mark.asyncio
    async def test_execute_populates_tool_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Full path: execute() -> parse_openai -> AgentResponse.tool_calls."""
        from arksim.config import AgentConfig, AgentType
        from arksim.simulation_engine.agent.clients.chat_completions import (
            ChatCompletionsAgent,
        )
        from arksim.simulation_engine.tool_types import ToolCallSource

        # AgentConfig's mode='before' validator calls ChatCompletionsConfig(**api_config),
        # so api_config must be a plain dict, not a pre-constructed model instance.
        config = AgentConfig(
            agent_name="test",
            agent_type=AgentType.CHAT_COMPLETIONS.value,
            api_config={
                "endpoint": "http://localhost:9999/v1/chat/completions",
                "headers": {"Authorization": "Bearer test"},
                "body": {
                    "model": "test-model",
                    "messages": [{"role": "system", "content": "you are a test"}],
                },
            },
        )
        agent = ChatCompletionsAgent(config)

        async def fake_post(payload: dict) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_x",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "SF"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

        monkeypatch.setattr(agent, "_post_request", fake_post)

        response = await agent.execute("what's the weather?")
        assert response.content == ""
        assert len(response.tool_calls) == 1
        tc = response.tool_calls[0]
        assert tc.id == "call_x"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "SF"}
        assert tc.source == ToolCallSource.CHAT_COMPLETIONS

        await agent.close()
