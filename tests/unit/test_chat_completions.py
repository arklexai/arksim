# SPDX-License-Identifier: Apache-2.0
"""Integration tests for ChatCompletionsAgent.execute() tool-call loop."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.clients.chat_completions import ChatCompletionsAgent


@pytest.fixture
def chat_completions_agent(
    valid_agent_config_chat_completions_new: dict, mock_env_vars: dict
) -> ChatCompletionsAgent:
    """ChatCompletionsAgent instance for testing execute()."""
    config = AgentConfig(**valid_agent_config_chat_completions_new)
    return ChatCompletionsAgent(config)


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


class TestToolCallGeminiFormat:
    """Test tool-call loop with Gemini-style functionCall responses."""

    @pytest.mark.asyncio
    async def test_gemini_loop_then_text(
        self, chat_completions_agent: ChatCompletionsAgent
    ) -> None:
        tool_response = {
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
        text_response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "It is sunny in NYC."}],
                    }
                }
            ]
        }

        with patch.object(
            chat_completions_agent,
            "_post_request",
            new_callable=AsyncMock,
            side_effect=[tool_response, text_response],
        ):
            answer = await chat_completions_agent.execute("weather?")

        assert answer == "It is sunny in NYC."
        hist = chat_completions_agent.conversation_history
        # user, model (functionCall parts), user (functionResponse), assistant (text)
        assert hist[0]["role"] == "user"
        assert hist[1]["role"] == "model"
        assert "functionCall" in hist[1]["parts"][0]
        assert hist[2]["role"] == "user"
        assert "functionResponse" in hist[2]["parts"][0]
        assert hist[3]["role"] == "assistant"
