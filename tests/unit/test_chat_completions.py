# SPDX-License-Identifier: Apache-2.0
"""Tests for chat completions agent _extract_content and tool-call handling."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.clients.chat_completions import ChatCompletionsAgent


@pytest.fixture
def chat_completions_agent(
    valid_agent_config_chat_completions_new: dict, mock_env_vars: dict
) -> ChatCompletionsAgent:
    """ChatCompletionsAgent instance for testing _extract_content."""
    config = AgentConfig(**valid_agent_config_chat_completions_new)
    return ChatCompletionsAgent(config)


class TestExtractContentOpenAI:
    """Tests for OpenAI-style response format."""

    def test_message_content(
        self, chat_completions_agent: ChatCompletionsAgent
    ) -> None:
        result = {
            "choices": [{"message": {"role": "assistant", "content": "Hello, world!"}}]
        }
        assert chat_completions_agent._extract_content(result) == "Hello, world!"

    def test_delta_content(self, chat_completions_agent: ChatCompletionsAgent) -> None:
        result = {"choices": [{"delta": {"content": "Streamed "}}]}
        assert chat_completions_agent._extract_content(result) == "Streamed "

    def test_empty_choices_raises(
        self, chat_completions_agent: ChatCompletionsAgent
    ) -> None:
        result = {"choices": []}
        with pytest.raises(ValueError, match="empty 'choices'"):
            chat_completions_agent._extract_content(result)

    def test_choice_without_message_or_delta_raises(
        self, chat_completions_agent: ChatCompletionsAgent
    ) -> None:
        result = {"choices": [{"finish_reason": "stop"}]}
        with pytest.raises(ValueError, match="no 'message' or 'delta'"):
            chat_completions_agent._extract_content(result)


class TestExtractContentAnthropic:
    """Tests for Anthropic-style response format."""

    def test_single_text_block(
        self, chat_completions_agent: ChatCompletionsAgent
    ) -> None:
        result = {"content": [{"type": "text", "text": "Anthropic reply."}]}
        assert chat_completions_agent._extract_content(result) == "Anthropic reply."

    def test_multiple_text_blocks(
        self, chat_completions_agent: ChatCompletionsAgent
    ) -> None:
        result = {
            "content": [
                {"type": "text", "text": "Part one. "},
                {"type": "text", "text": "Part two."},
            ]
        }
        assert chat_completions_agent._extract_content(result) == "Part one. Part two."

    def test_ignores_non_text_blocks(
        self, chat_completions_agent: ChatCompletionsAgent
    ) -> None:
        result = {
            "content": [
                {"type": "image", "source": {}},
                {"type": "text", "text": "Only this."},
            ]
        }
        assert chat_completions_agent._extract_content(result) == "Only this."


class TestExtractContentGoogle:
    """Tests for Google-style response format."""

    def test_single_part(self, chat_completions_agent: ChatCompletionsAgent) -> None:
        result = {
            "candidates": [{"content": {"parts": [{"text": "Google response."}]}}]
        }
        assert chat_completions_agent._extract_content(result) == "Google response."

    def test_multiple_parts(self, chat_completions_agent: ChatCompletionsAgent) -> None:
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
        assert chat_completions_agent._extract_content(result) == "ABC"

    def test_empty_candidates_raises(
        self, chat_completions_agent: ChatCompletionsAgent
    ) -> None:
        result = {"candidates": []}
        with pytest.raises(ValueError, match="empty 'candidates'"):
            chat_completions_agent._extract_content(result)


class TestExtractContentUnsupported:
    """Tests for unsupported or invalid response format."""

    def test_empty_dict_raises(
        self, chat_completions_agent: ChatCompletionsAgent
    ) -> None:
        result = {}
        with pytest.raises(ValueError, match="Unsupported response format"):
            chat_completions_agent._extract_content(result)

    def test_unknown_keys_in_error_message(
        self, chat_completions_agent: ChatCompletionsAgent
    ) -> None:
        result = {"id": "xyz", "usage": {}}
        with pytest.raises(ValueError, match="Keys present"):
            chat_completions_agent._extract_content(result)


# ── Tool-call detection ──


class TestExtractToolCalls:
    """Tests for _extract_tool_calls auto-detection."""

    def test_openai_tool_calls(
        self, chat_completions_agent: ChatCompletionsAgent
    ) -> None:
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
        tc = chat_completions_agent._extract_tool_calls(result)
        assert tc is not None
        assert len(tc) == 1
        assert tc[0]["id"] == "call_1"

    def test_anthropic_tool_use(
        self, chat_completions_agent: ChatCompletionsAgent
    ) -> None:
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
        tc = chat_completions_agent._extract_tool_calls(result)
        assert tc is not None
        assert len(tc) == 1
        assert tc[0]["id"] == "tu_1"

    def test_no_tool_calls_returns_none(
        self, chat_completions_agent: ChatCompletionsAgent
    ) -> None:
        result = {
            "choices": [{"message": {"role": "assistant", "content": "Just text."}}]
        }
        assert chat_completions_agent._extract_tool_calls(result) is None

    def test_anthropic_text_only_returns_none(
        self, chat_completions_agent: ChatCompletionsAgent
    ) -> None:
        result = {"content": [{"type": "text", "text": "Just text."}]}
        assert chat_completions_agent._extract_tool_calls(result) is None


# ── Tool result building ──


class TestBuildToolResults:
    """Tests for _build_tool_results format handling."""

    def test_openai_format(self, chat_completions_agent: ChatCompletionsAgent) -> None:
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            }
        ]
        results = chat_completions_agent._build_tool_results(tool_calls)
        assert len(results) == 1
        assert results[0]["role"] == "tool"
        assert results[0]["tool_call_id"] == "call_1"
        assert results[0]["content"] == '{"status": "ok"}'

    def test_anthropic_format(
        self, chat_completions_agent: ChatCompletionsAgent
    ) -> None:
        tool_calls = [{"type": "tool_use", "id": "tu_1", "name": "f", "input": {}}]
        results = chat_completions_agent._build_tool_results(tool_calls)
        assert len(results) == 1
        assert results[0]["role"] == "user"
        blocks = results[0]["content"]
        assert len(blocks) == 1
        assert blocks[0]["type"] == "tool_result"
        assert blocks[0]["tool_use_id"] == "tu_1"


# ── Tool-call loop integration ──


class TestToolCallLoop:
    """Test that execute() loops on tool calls then returns text."""

    @pytest.mark.asyncio
    async def test_loop_then_text(
        self, chat_completions_agent: ChatCompletionsAgent
    ) -> None:
        tool_response = {
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
                                    "name": "lookup",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    }
                }
            ]
        }
        text_response = {
            "choices": [{"message": {"role": "assistant", "content": "Final answer."}}]
        }

        with patch.object(
            chat_completions_agent,
            "_post_request",
            new_callable=AsyncMock,
            side_effect=[tool_response, text_response],
        ):
            answer = await chat_completions_agent.execute("hello")

        assert answer == "Final answer."
        # History should contain: user, assistant (tool_calls), tool result, assistant (text)
        hist = chat_completions_agent.conversation_history
        assert hist[0]["role"] == "user"
        assert hist[1]["role"] == "assistant"
        assert "tool_calls" in hist[1]
        assert hist[2]["role"] == "tool"
        assert hist[3]["role"] == "assistant"
        assert hist[3]["content"] == "Final answer."


class TestToolCallMaxRounds:
    """Test that execute() raises after max rounds exceeded."""

    @pytest.mark.asyncio
    async def test_exceeds_max_rounds(
        self, chat_completions_agent: ChatCompletionsAgent
    ) -> None:
        chat_completions_agent.max_tool_call_rounds = 2
        tool_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ]
        }

        with (
            patch.object(
                chat_completions_agent,
                "_post_request",
                new_callable=AsyncMock,
                return_value=tool_response,
            ),
            pytest.raises(RuntimeError, match="exceeded 2 tool-call rounds"),
        ):
            await chat_completions_agent.execute("hello")


class TestToolCallAnthropicFormat:
    """Test tool-call loop with Anthropic-style tool_use responses."""

    @pytest.mark.asyncio
    async def test_anthropic_loop_then_text(
        self, chat_completions_agent: ChatCompletionsAgent
    ) -> None:
        tool_response = {
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
        text_response = {"content": [{"type": "text", "text": "The weather is sunny."}]}

        with patch.object(
            chat_completions_agent,
            "_post_request",
            new_callable=AsyncMock,
            side_effect=[tool_response, text_response],
        ):
            answer = await chat_completions_agent.execute("weather?")

        assert answer == "The weather is sunny."
        hist = chat_completions_agent.conversation_history
        # user, assistant (tool_use content), user (tool_result), assistant (text)
        assert hist[0]["role"] == "user"
        assert hist[1]["role"] == "assistant"
        assert hist[2]["role"] == "user"
        assert hist[2]["content"][0]["type"] == "tool_result"
        assert hist[3]["role"] == "assistant"
