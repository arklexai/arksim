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

    def test_claude_provider(self) -> None:
        try:
            cls = LLM._get_provider("claude")
            assert cls.__name__ == "ClaudeLLM"
        except ModuleNotFoundError:
            pytest.skip("anthropic not installed")

    def test_gemini_provider(self) -> None:
        try:
            cls = LLM._get_provider("gemini")
            assert cls.__name__ == "GeminiLLM"
        except (ModuleNotFoundError, ImportError):
            pytest.skip("google-genai not installed")

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            LLM._get_provider("unknown")
