# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.llms.chat.providers.openai (OpenAILLM constructor)."""

from __future__ import annotations

from unittest.mock import patch

from arksim.llms.chat.providers.openai import OpenAILLM


class TestOpenAILLMConstructor:
    def test_defaults_no_base_url_or_api_key(self) -> None:
        with (
            patch("arksim.llms.chat.providers.openai.OpenAI") as mock_sync,
            patch("arksim.llms.chat.providers.openai.AsyncOpenAI") as mock_async,
        ):
            OpenAILLM(model="gpt-4.1-mini")
            mock_sync.assert_called_once_with()
            mock_async.assert_called_once_with()

    def test_base_url_passthrough(self) -> None:
        with (
            patch("arksim.llms.chat.providers.openai.OpenAI") as mock_sync,
            patch("arksim.llms.chat.providers.openai.AsyncOpenAI") as mock_async,
        ):
            OpenAILLM(model="llama3.1", base_url="http://localhost:11434/v1")
            mock_sync.assert_called_once_with(base_url="http://localhost:11434/v1")
            mock_async.assert_called_once_with(base_url="http://localhost:11434/v1")

    def test_api_key_passthrough(self) -> None:
        with (
            patch("arksim.llms.chat.providers.openai.OpenAI") as mock_sync,
            patch("arksim.llms.chat.providers.openai.AsyncOpenAI") as mock_async,
        ):
            OpenAILLM(model="gpt-4.1-mini", api_key="sk-test-123")
            mock_sync.assert_called_once_with(api_key="sk-test-123")
            mock_async.assert_called_once_with(api_key="sk-test-123")

    def test_base_url_and_api_key_both_passthrough(self) -> None:
        with (
            patch("arksim.llms.chat.providers.openai.OpenAI") as mock_sync,
            patch("arksim.llms.chat.providers.openai.AsyncOpenAI") as mock_async,
        ):
            OpenAILLM(
                model="meta-llama/Llama-3.1-8B-Instruct",
                base_url="http://my-vllm:8000/v1",
                api_key="vllm-token",
            )
            mock_sync.assert_called_once_with(
                base_url="http://my-vllm:8000/v1", api_key="vllm-token"
            )
            mock_async.assert_called_once_with(
                base_url="http://my-vllm:8000/v1", api_key="vllm-token"
            )

    def test_explicit_none_treated_as_omitted(self) -> None:
        with (
            patch("arksim.llms.chat.providers.openai.OpenAI") as mock_sync,
            patch("arksim.llms.chat.providers.openai.AsyncOpenAI") as mock_async,
        ):
            OpenAILLM(model="gpt-4.1-mini", base_url=None, api_key=None)
            mock_sync.assert_called_once_with()
            mock_async.assert_called_once_with()

    def test_empty_string_treated_as_omitted(self) -> None:
        """YAML configs can produce empty strings; treat them like None.

        Avoids confusing httpx/auth errors when a user leaves
        `base_url:` or `api_key:` blank in their config.
        """
        with (
            patch("arksim.llms.chat.providers.openai.OpenAI") as mock_sync,
            patch("arksim.llms.chat.providers.openai.AsyncOpenAI") as mock_async,
        ):
            OpenAILLM(model="gpt-4.1-mini", base_url="", api_key="")
            mock_sync.assert_called_once_with()
            mock_async.assert_called_once_with()

    def test_temperature_flows_through_with_base_url(self) -> None:
        """Adding base_url/api_key must not swallow temperature via kwargs."""
        with (
            patch("arksim.llms.chat.providers.openai.OpenAI"),
            patch("arksim.llms.chat.providers.openai.AsyncOpenAI"),
        ):
            llm = OpenAILLM(
                model="gpt-4.1-mini",
                temperature=0.7,
                base_url="http://localhost:11434/v1",
            )
            assert llm.temperature == 0.7
            assert llm.model == "gpt-4.1-mini"
