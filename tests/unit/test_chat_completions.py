# SPDX-License-Identifier: Apache-2.0
"""Tests for chat completions agent _extract_content."""

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


class TestExtractContentGemini:
    """Tests for Gemini-style response format."""

    def test_single_part(self, chat_completions_agent: ChatCompletionsAgent) -> None:
        result = {
            "candidates": [{"content": {"parts": [{"text": "Gemini response."}]}}]
        }
        assert chat_completions_agent._extract_content(result) == "Gemini response."

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
