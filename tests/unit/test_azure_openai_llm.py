# SPDX-License-Identifier: Apache-2.0
"""Tests for AzureOpenAILLM parse-failure branches."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from arksim.llms.chat.providers.azure_openai import AzureOpenAILLM


class _DummySchema(BaseModel):
    answer: str


def _make_unparsed_response() -> MagicMock:
    """Build a mock response where choices[0].message.parsed is None."""
    message = SimpleNamespace(parsed=None, content="raw fallback text")
    choice = SimpleNamespace(message=message)
    response = MagicMock()
    response.choices = [choice]
    return response


def _build_llm() -> AzureOpenAILLM:
    """Instantiate AzureOpenAILLM with all external deps mocked out."""
    with (
        patch.dict(
            "os.environ",
            {
                "AZURE_OPENAI_ENDPOINT": "https://fake.openai.azure.com",
                "AZURE_OPENAI_API_VERSION": "2024-02-01",
                "AZURE_CLIENT_ID": "fake-client-id",
            },
        ),
        patch(
            "arksim.llms.chat.providers.azure_openai.get_azure_token_provider",
            return_value=lambda: "fake-token",
        ),
        patch(
            "arksim.llms.chat.providers.azure_openai.AzureOpenAI",
        ),
        patch(
            "arksim.llms.chat.providers.azure_openai.AsyncAzureOpenAI",
        ),
    ):
        llm = AzureOpenAILLM(model="gpt-4")
    return llm


class TestAzureOpenAICallParseFailure:
    @patch("time.sleep", return_value=None)
    def test_call_raises_when_parsed_is_none(self, _sleep: MagicMock) -> None:
        llm = _build_llm()
        llm.client = MagicMock()
        llm.client.beta.chat.completions.parse.return_value = _make_unparsed_response()
        with pytest.raises(ValueError, match="Failed to parse response"):
            llm.call("hello", schema=_DummySchema)

    @pytest.mark.asyncio
    @patch("asyncio.sleep", new_callable=AsyncMock)
    async def test_call_async_raises_when_parsed_is_none(
        self, _sleep: AsyncMock
    ) -> None:
        llm = _build_llm()
        llm.async_client = MagicMock()
        llm.async_client.beta.chat.completions.parse = AsyncMock(
            return_value=_make_unparsed_response()
        )
        with pytest.raises(ValueError, match="Failed to parse response"):
            await llm.call_async("hello", schema=_DummySchema)
