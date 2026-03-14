# SPDX-License-Identifier: Apache-2.0
"""Tests for MiniMaxLLM provider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from arksim.llms.chat.providers.minimax import DEFAULT_BASE_URL, MiniMaxLLM


class _DummySchema(BaseModel):
    answer: str


def _build_llm(
    temperature: float | None = None,
) -> MiniMaxLLM:
    """Instantiate MiniMaxLLM with mocked OpenAI clients."""
    with (
        patch.dict(
            "os.environ",
            {"MINIMAX_API_KEY": "test-key"},
        ),
        patch("arksim.llms.chat.providers.minimax.OpenAI"),
        patch("arksim.llms.chat.providers.minimax.AsyncOpenAI"),
    ):
        llm = MiniMaxLLM(model="MiniMax-M2.5", temperature=temperature)
    return llm


class TestMiniMaxLLMInit:
    def test_requires_api_key(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="MINIMAX_API_KEY"),
        ):
            MiniMaxLLM(model="MiniMax-M2.5")

    def test_creates_instance_with_api_key(self) -> None:
        llm = _build_llm()
        assert llm.model == "MiniMax-M2.5"

    def test_uses_default_base_url(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {"MINIMAX_API_KEY": "test-key"},
            ),
            patch("arksim.llms.chat.providers.minimax.OpenAI") as mock_openai,
            patch("arksim.llms.chat.providers.minimax.AsyncOpenAI"),
        ):
            MiniMaxLLM(model="MiniMax-M2.5")
            mock_openai.assert_called_once_with(
                api_key="test-key", base_url=DEFAULT_BASE_URL
            )

    def test_uses_custom_base_url(self) -> None:
        custom_url = "https://api.minimaxi.com/v1"
        with (
            patch.dict(
                "os.environ",
                {"MINIMAX_API_KEY": "test-key", "MINIMAX_BASE_URL": custom_url},
            ),
            patch("arksim.llms.chat.providers.minimax.OpenAI") as mock_openai,
            patch("arksim.llms.chat.providers.minimax.AsyncOpenAI"),
        ):
            MiniMaxLLM(model="MiniMax-M2.5")
            mock_openai.assert_called_once_with(api_key="test-key", base_url=custom_url)


class TestMiniMaxLLMPrepareMessages:
    def test_string_input(self) -> None:
        llm = _build_llm()
        result = llm._prepare_messages("hello")
        assert result == [{"role": "user", "content": "hello"}]

    def test_list_input(self) -> None:
        llm = _build_llm()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = llm._prepare_messages(messages)
        assert result == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]


class TestMiniMaxLLMPrepareParams:
    def test_basic_params(self) -> None:
        llm = _build_llm()
        params = llm._prepare_params("hello")
        assert params["model"] == "MiniMax-M2.5"
        assert params["messages"] == [{"role": "user", "content": "hello"}]
        assert params["temperature"] == 1.0

    def test_temperature_included(self) -> None:
        llm = _build_llm(temperature=0.7)
        params = llm._prepare_params("hello")
        assert params["temperature"] == 0.7

    def test_zero_temperature_clamped(self) -> None:
        llm = _build_llm(temperature=0.0)
        params = llm._prepare_params("hello")
        assert params["temperature"] == 0.01

    def test_schema_adds_system_prompt(self) -> None:
        llm = _build_llm()
        params = llm._prepare_params("hello", schema=_DummySchema)
        assert len(params["messages"]) == 2
        assert params["messages"][0]["role"] == "system"
        assert "JSON" in params["messages"][0]["content"]


class TestMiniMaxLLMParseJson:
    def test_direct_json(self) -> None:
        result = MiniMaxLLM._parse_json('{"answer": "yes"}')
        assert result == {"answer": "yes"}

    def test_json_in_code_block(self) -> None:
        text = 'Here is the answer:\n```json\n{"answer": "yes"}\n```'
        result = MiniMaxLLM._parse_json(text)
        assert result == {"answer": "yes"}

    def test_strips_think_tags(self) -> None:
        text = '<think>\nSome reasoning here\n</think>\n{"answer": "yes"}'
        result = MiniMaxLLM._parse_json(text)
        assert result == {"answer": "yes"}

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            MiniMaxLLM._parse_json("not json at all")


class TestMiniMaxLLMCall:
    @patch("time.sleep", return_value=None)
    def test_call_text(self, _sleep: MagicMock) -> None:
        llm = _build_llm()
        mock_message = SimpleNamespace(content="Hello from MiniMax!")
        mock_choice = SimpleNamespace(message=mock_message)
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        llm.client = MagicMock()
        llm.client.chat.completions.create.return_value = mock_response
        result = llm.call("hello")
        assert result == "Hello from MiniMax!"

    @patch("time.sleep", return_value=None)
    def test_call_schema(self, _sleep: MagicMock) -> None:
        llm = _build_llm()
        mock_message = SimpleNamespace(content='{"answer": "42"}')
        mock_choice = SimpleNamespace(message=mock_message)
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        llm.client = MagicMock()
        llm.client.chat.completions.create.return_value = mock_response
        result = llm.call("what is the answer?", schema=_DummySchema)
        assert isinstance(result, _DummySchema)
        assert result.answer == "42"

    @pytest.mark.asyncio
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_call_async_text(self, _sleep: AsyncMock) -> None:
        llm = _build_llm()
        mock_message = SimpleNamespace(content="Async hello!")
        mock_choice = SimpleNamespace(message=mock_message)
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        llm.async_client = MagicMock()
        llm.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        result = await llm.call_async("hello")
        assert result == "Async hello!"

    @pytest.mark.asyncio
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_call_async_schema(self, _sleep: AsyncMock) -> None:
        llm = _build_llm()
        mock_message = SimpleNamespace(content='{"answer": "async 42"}')
        mock_choice = SimpleNamespace(message=mock_message)
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        llm.async_client = MagicMock()
        llm.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        result = await llm.call_async("question", schema=_DummySchema)
        assert isinstance(result, _DummySchema)
        assert result.answer == "async 42"
