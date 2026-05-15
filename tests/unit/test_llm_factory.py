# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.llms.chat.llm (LLM factory) and base_llm."""

from __future__ import annotations

import pytest

from arksim.llms.chat.llm import LLM


class TestLLMFactory:
    def test_invalid_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Model name is required"):
            LLM(model="")

    def test_unsupported_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            LLM(model="test-model", provider="nonexistent")

    def test_none_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Model name is required"):
            LLM(model=None)


class TestGetProvider:
    def test_openai_provider(self) -> None:
        cls = LLM._get_provider("openai")
        assert cls.__name__ == "OpenAILLM"

    def test_azure_provider(self) -> None:
        cls = LLM._get_provider("azure")
        assert cls.__name__ == "AzureOpenAILLM"

    def test_anthropic_provider(self) -> None:
        try:
            cls = LLM._get_provider("anthropic")
            assert cls.__name__ == "AnthropicLLM"
        except ModuleNotFoundError:
            pytest.skip("anthropic not installed")

    def test_google_provider(self) -> None:
        try:
            cls = LLM._get_provider("google")
            assert cls.__name__ == "GoogleLLM"
        except (ModuleNotFoundError, ImportError):
            pytest.skip("google-genai not installed")

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            LLM._get_provider("unknown")

    def test_responses_provider_resolves_to_openai_llm(self) -> None:
        cls = LLM._get_provider("responses")
        assert cls.__name__ == "OpenAILLM"

    def test_open_responses_provider_resolves_to_openai_llm(self) -> None:
        cls = LLM._get_provider("open_responses")
        assert cls.__name__ == "OpenAILLM"

    def test_open_responses_provider_emits_info_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        with caplog.at_level(logging.INFO, logger="arksim.llms.chat.llm"):
            LLM._get_provider("open_responses")
        assert any("open_responses" in rec.message for rec in caplog.records)

    def test_openai_provider_does_not_emit_info_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        with caplog.at_level(logging.INFO, logger="arksim.llms.chat.llm"):
            LLM._get_provider("openai")
        assert not any("open_responses" in rec.message for rec in caplog.records)
