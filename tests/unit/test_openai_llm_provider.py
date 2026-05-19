# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.llms.chat.providers.openai (OpenAILLM constructor)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from openai import OpenAIError

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

    def test_whitespace_base_url_and_api_key_treated_as_omitted(self) -> None:
        """Whitespace-only values must not reach the SDK; they would be
        URL-encoded into `%20...` and cause slow connection failures.
        """
        with (
            patch("arksim.llms.chat.providers.openai.OpenAI") as mock_sync,
            patch("arksim.llms.chat.providers.openai.AsyncOpenAI") as mock_async,
        ):
            OpenAILLM(model="gpt-4o-mini", base_url="   ", api_key="\t\n")
            mock_sync.assert_called_once_with()
            mock_async.assert_called_once_with()

    def test_base_url_and_api_key_are_stripped(self) -> None:
        """Leading and trailing whitespace is stripped before passing to SDK."""
        with (
            patch("arksim.llms.chat.providers.openai.OpenAI") as mock_sync,
            patch("arksim.llms.chat.providers.openai.AsyncOpenAI") as mock_async,
        ):
            OpenAILLM(
                model="llama3.1",
                base_url=" http://localhost:11434/v1 ",
                api_key=" ollama ",
            )
            mock_sync.assert_called_once_with(
                base_url="http://localhost:11434/v1", api_key="ollama"
            )
            mock_async.assert_called_once_with(
                base_url="http://localhost:11434/v1", api_key="ollama"
            )

    def test_missing_credentials_raises_actionable_error(self) -> None:
        """When the OpenAI SDK rejects no-credentials construction, wrap the
        error with an arksim-specific message pointing at the YAML field
        and docs URL. The user should not see a bare OpenAIError stack trace.
        """
        with (
            patch("arksim.llms.chat.providers.openai.OpenAI") as mock_sync,
            patch("arksim.llms.chat.providers.openai.AsyncOpenAI"),
        ):
            mock_sync.side_effect = OpenAIError(
                "The api_key client option must be set..."
            )
            with pytest.raises(ValueError, match="OpenAI SDK requires an api_key"):
                OpenAILLM(model="gpt-4o-mini")


class TestOpenAILLMCacheStats:
    def test_cache_stats_starts_at_zero(self) -> None:
        with (
            patch("arksim.llms.chat.providers.openai.OpenAI"),
            patch("arksim.llms.chat.providers.openai.AsyncOpenAI"),
        ):
            llm = OpenAILLM(model="gpt-4o-mini")
        stats = llm.cache_stats()
        assert stats == {
            "call_count": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "cache_hit_rate": 0.0,
        }

    def test_record_usage_accumulates(self) -> None:
        from types import SimpleNamespace

        with (
            patch("arksim.llms.chat.providers.openai.OpenAI"),
            patch("arksim.llms.chat.providers.openai.AsyncOpenAI"),
        ):
            llm = OpenAILLM(model="gpt-4o-mini")

        # First call: no cache hits
        response1 = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=1024,
                output_tokens=50,
                input_tokens_details=SimpleNamespace(cached_tokens=0),
            )
        )
        llm._record_usage(response1)

        # Second call: prefix hits cache
        response2 = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=1100,
                output_tokens=60,
                input_tokens_details=SimpleNamespace(cached_tokens=900),
            )
        )
        llm._record_usage(response2)

        stats = llm.cache_stats()
        assert stats["call_count"] == 2
        assert stats["input_tokens"] == 2124
        assert stats["cached_input_tokens"] == 900
        assert stats["output_tokens"] == 110
        assert stats["cache_hit_rate"] == 900 / 2124

    def test_record_usage_tolerates_missing_usage_field(self) -> None:
        """Self-hosted backends (Ollama, vLLM) may omit usage entirely.
        Telemetry must never raise.
        """
        from types import SimpleNamespace

        with (
            patch("arksim.llms.chat.providers.openai.OpenAI"),
            patch("arksim.llms.chat.providers.openai.AsyncOpenAI"),
        ):
            llm = OpenAILLM(model="llama3.1")

        # Response without usage attribute at all
        response = SimpleNamespace()
        llm._record_usage(response)
        assert llm.cache_stats()["call_count"] == 0

        # Response with usage but no input_tokens_details (e.g. older SDK)
        response = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=500, output_tokens=20)
        )
        llm._record_usage(response)
        assert llm.cache_stats()["call_count"] == 1
        assert llm.cache_stats()["cached_input_tokens"] == 0
