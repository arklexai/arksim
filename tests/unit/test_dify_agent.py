# SPDX-License-Identifier: Apache-2.0
"""Tests for the Dify integration agent (httpx).

Covers constructor validation, HTTP call mechanics, conversation_id
persistence, and error handling.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent


def _load_dify_agent_class() -> type[BaseAgent]:
    """Lazy-import DifyAgent to avoid module-level DIFY_API_KEY requirement."""
    from examples.integrations.dify.custom_agent import DifyAgent

    return DifyAgent


def _make_agent_config() -> AgentConfig:
    """Minimal AgentConfig for testing."""
    return AgentConfig(
        agent_type="custom",
        agent_name="dify-test",
        custom_config={"module_path": "./custom_agent.py"},
    )


def _dify_response(
    answer: str = "Hello!",
    conversation_id: str = "conv-123",
) -> dict[str, Any]:
    """Build a Dify blocking-mode response dict."""
    return {
        "answer": answer,
        "conversation_id": conversation_id,
        "message_id": str(uuid.uuid4()),
    }


class TestDifyAgentInit:
    """Constructor validation."""

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DIFY_API_KEY", raising=False)
        with pytest.raises(ValueError, match="DIFY_API_KEY"):
            _load_dify_agent_class()(_make_agent_config())


class TestDifyAgentExecute:
    """HTTP call and response parsing."""

    @pytest.fixture()
    def agent(self, monkeypatch: pytest.MonkeyPatch) -> BaseAgent:
        monkeypatch.setenv("DIFY_API_KEY", "test-key-123")
        return _load_dify_agent_class()(_make_agent_config())

    async def test_returns_answer(self, agent: BaseAgent) -> None:
        """Successful response returns the answer string."""
        mock_response = httpx.Response(
            200,
            json=_dify_response(answer="I can help with that."),
            request=httpx.Request("POST", "https://api.dify.ai/v1/chat-messages"),
        )
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(return_value=mock_response)
        result = await agent.execute("How do I return an item?")
        assert result == "I can help with that."

    async def test_conversation_id_persists(self, agent: BaseAgent) -> None:
        """Second call sends conversation_id from first response."""
        mock_response = httpx.Response(
            200,
            json=_dify_response(conversation_id="conv-abc"),
            request=httpx.Request("POST", "https://api.dify.ai/v1/chat-messages"),
        )
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(return_value=mock_response)
        await agent.execute("first message")
        assert agent._conversation_id == "conv-abc"
        await agent.execute("second message")
        second_call_kwargs = agent._client.post.call_args_list[1]
        body = second_call_kwargs.kwargs.get("json") or second_call_kwargs[1].get(
            "json"
        )
        assert body["conversation_id"] == "conv-abc"

    async def test_missing_answer_raises(self, agent: BaseAgent) -> None:
        """Response without 'answer' field raises RuntimeError."""
        mock_response = httpx.Response(
            200,
            json={"conversation_id": "conv-123"},
            request=httpx.Request("POST", "https://api.dify.ai/v1/chat-messages"),
        )
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(return_value=mock_response)
        with pytest.raises(RuntimeError, match="answer"):
            await agent.execute("hello")

    async def test_auth_error_mentions_api_key(self, agent: BaseAgent) -> None:
        """401 response raises RuntimeError mentioning DIFY_API_KEY."""
        mock_response = httpx.Response(
            401,
            json={"code": "invalid_api_key", "message": "Invalid API key"},
            request=httpx.Request("POST", "https://api.dify.ai/v1/chat-messages"),
        )
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(return_value=mock_response)
        with pytest.raises(RuntimeError, match="DIFY_API_KEY"):
            await agent.execute("hello")

    async def test_server_error_includes_status_and_body(
        self, agent: BaseAgent
    ) -> None:
        """5xx response raises RuntimeError with status code and response body."""
        mock_response = httpx.Response(
            500,
            text="Internal Server Error",
            request=httpx.Request("POST", "https://api.dify.ai/v1/chat-messages"),
        )
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(return_value=mock_response)
        with pytest.raises(RuntimeError, match="HTTP 500"):
            await agent.execute("hello")

    async def test_connect_error_raises_readable_message(
        self, agent: BaseAgent
    ) -> None:
        """Unreachable Dify server raises RuntimeError with diagnostic message."""
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        with pytest.raises(RuntimeError, match="Could not connect to Dify API"):
            await agent.execute("hello")

    async def test_timeout_raises_readable_message(self, agent: BaseAgent) -> None:
        """Timeout raises RuntimeError with diagnostic message."""
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
        with pytest.raises(RuntimeError, match="timed out"):
            await agent.execute("hello")
