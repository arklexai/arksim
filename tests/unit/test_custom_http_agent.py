# SPDX-License-Identifier: Apache-2.0
"""Tests for the custom_http agent connector.

Covers placeholder substitution (including nested, plus the {{chat_id}}
placeholder and its stability across turns, and the {{timestamp}}
placeholder and its freshness per request), response_path extraction
(including a list-index case), response_extras rendering, non-2xx and
missing-path errors, and factory routing for AgentType.CUSTOM_HTTP.

Renderer internals (shape normalization, hole substitution, caps) live in
test_response_render.py; the cases here cover the connector wiring.

HTTP is mocked with httpx.Response + AsyncMock on the agent's
``_client.post`` (matching the existing Dify agent tests), so no respx
dependency is required.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from arksim.config import AgentConfig, AgentType
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.simulation_engine.agent.clients.custom_http import CustomHTTPAgent
from arksim.simulation_engine.agent.factory import create_agent
from arksim.simulation_engine.tool_types import AgentResponse

ENDPOINT = "https://api.example.com/agent"


def _make_config(body: dict[str, Any]) -> AgentConfig:
    """Build a custom_http AgentConfig with the given body template."""
    return AgentConfig(
        agent_type="custom_http",
        agent_name="custom-http-test",
        api_config={
            "endpoint": ENDPOINT,
            "headers": {"Authorization": "Bearer test-token"},
            "body": body,
        },
    )


def _mock_response(payload: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json=payload,
        request=httpx.Request("POST", ENDPOINT),
    )


class TestFactoryRouting:
    """create_agent routes custom_http to CustomHTTPAgent."""

    def test_factory_creates_custom_http_agent(self) -> None:
        config = _make_config({"message": "{{message}}", "response_path": "data.reply"})
        agent = create_agent(config)
        assert isinstance(agent, CustomHTTPAgent)
        assert isinstance(agent, BaseAgent)
        assert config.agent_type == AgentType.CUSTOM_HTTP.value


class TestSubstitution:
    """{{message}} placeholder substitution, including nested templates."""

    async def test_top_level_substitution(self) -> None:
        agent = create_agent(
            _make_config({"query": "{{message}}", "response_path": "reply"})
        )
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(return_value=_mock_response({"reply": "hi"}))

        result = await agent.execute("hello there")

        sent_body = agent._client.post.call_args.kwargs["json"]
        assert sent_body == {"query": "hello there"}
        # response_path is config, not part of the outbound request.
        assert "response_path" not in sent_body
        assert isinstance(result, AgentResponse)
        assert result.content == "hi"
        assert result.tool_calls == []

    async def test_nested_substitution(self) -> None:
        body = {
            "inputs": {"messages": [{"role": "user", "text": "{{message}}"}]},
            "stream": False,
            "response_path": "data.reply",
        }
        agent = create_agent(_make_config(body))
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(
            return_value=_mock_response({"data": {"reply": "ok"}})
        )

        await agent.execute("nested query")

        sent_body = agent._client.post.call_args.kwargs["json"]
        assert sent_body["inputs"]["messages"][0]["text"] == "nested query"
        assert sent_body["stream"] is False
        assert "response_path" not in sent_body

    async def test_template_not_mutated_across_calls(self) -> None:
        body = {"query": "{{message}}", "response_path": "reply"}
        agent = create_agent(_make_config(body))
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(return_value=_mock_response({"reply": "x"}))

        await agent.execute("first")
        await agent.execute("second")

        second_body = agent._client.post.call_args_list[1].kwargs["json"]
        assert second_body == {"query": "second"}
        # Original config template still carries the placeholder.
        assert agent.config.body["query"] == "{{message}}"

    async def test_chat_id_substitution(self) -> None:
        body = {
            "userContext": {"sessionId": "{{chat_id}}"},
            "questionContext": {"userQuestion": "{{message}}"},
            "response_path": "reply",
        }
        agent = create_agent(_make_config(body))
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(return_value=_mock_response({"reply": "ok"}))

        await agent.execute("hi")

        sent_body = agent._client.post.call_args.kwargs["json"]
        assert sent_body["questionContext"]["userQuestion"] == "hi"
        assert sent_body["userContext"]["sessionId"] == await agent.get_chat_id()
        assert "response_path" not in sent_body

    async def test_chat_id_stable_across_turns(self) -> None:
        # Server-side session state relies on the chat id being identical
        # across turns of one conversation (and unique per conversation).
        body = {"sid": "{{chat_id}}", "q": "{{message}}", "response_path": "reply"}
        agent = create_agent(_make_config(body))
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(return_value=_mock_response({"reply": "x"}))

        await agent.execute("first")
        await agent.execute("second")

        first_sid = agent._client.post.call_args_list[0].kwargs["json"]["sid"]
        second_sid = agent._client.post.call_args_list[1].kwargs["json"]["sid"]
        assert first_sid == second_sid == await agent.get_chat_id()

    async def test_timestamp_substitution(self) -> None:
        body = {
            "q": "{{message}}",
            "pageContext": {"timestamp": "{{timestamp}}"},
            "response_path": "reply",
        }
        agent = create_agent(_make_config(body))
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(return_value=_mock_response({"reply": "ok"}))

        before = datetime.now(timezone.utc)
        await agent.execute("hi")
        after = datetime.now(timezone.utc)

        sent = agent._client.post.call_args.kwargs["json"]["pageContext"]["timestamp"]
        # ISO 8601 UTC with millisecond precision and a Z suffix, e.g.
        # 2025-12-09T18:21:15.673Z.
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", sent)
        parsed = datetime.fromisoformat(sent.replace("Z", "+00:00"))
        # Millisecond truncation can put the parsed value just before `before`.
        assert before - timedelta(seconds=1) <= parsed <= after

    async def test_timestamp_fresh_per_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Unlike {{chat_id}}, the timestamp must be regenerated on every turn.
        ticks = iter(["2025-12-09T18:21:15.673Z", "2025-12-09T18:21:19.001Z"])
        monkeypatch.setattr(
            "arksim.simulation_engine.agent.clients.custom_http._current_timestamp",
            lambda: next(ticks),
        )
        body = {"q": "{{message}}", "ts": "{{timestamp}}", "response_path": "reply"}
        agent = create_agent(_make_config(body))
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(return_value=_mock_response({"reply": "x"}))

        await agent.execute("first")
        await agent.execute("second")

        first_ts = agent._client.post.call_args_list[0].kwargs["json"]["ts"]
        second_ts = agent._client.post.call_args_list[1].kwargs["json"]["ts"]
        assert first_ts == "2025-12-09T18:21:15.673Z"
        assert second_ts == "2025-12-09T18:21:19.001Z"


class TestResponsePathExtraction:
    """response_path dot-path extraction, including list indices."""

    async def test_simple_dot_path(self) -> None:
        agent = create_agent(
            _make_config({"message": "{{message}}", "response_path": "data.reply"})
        )
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(
            return_value=_mock_response({"data": {"reply": "the answer"}})
        )

        result = await agent.execute("q")
        assert result.content == "the answer"

    async def test_list_index_path(self) -> None:
        agent = create_agent(
            _make_config(
                {
                    "message": "{{message}}",
                    "response_path": "choices.0.message.content",
                }
            )
        )
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(
            return_value=_mock_response(
                {"choices": [{"message": {"content": "indexed reply"}}]}
            )
        )

        result = await agent.execute("q")
        assert result.content == "indexed reply"


class TestErrors:
    """Error handling for non-2xx, missing path, and non-string path."""

    async def test_non_2xx_raises(self) -> None:
        agent = create_agent(
            _make_config({"message": "{{message}}", "response_path": "reply"})
        )
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(
            return_value=httpx.Response(
                500,
                text="Internal Server Error",
                request=httpx.Request("POST", ENDPOINT),
            )
        )
        with pytest.raises(RuntimeError, match="HTTP 500"):
            await agent.execute("q")

    async def test_missing_path_raises(self) -> None:
        agent = create_agent(
            _make_config({"message": "{{message}}", "response_path": "data.reply"})
        )
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(
            return_value=_mock_response({"data": {"other": "value"}})
        )
        with pytest.raises(RuntimeError, match="not found"):
            await agent.execute("q")

    async def test_non_string_leaf_raises(self) -> None:
        agent = create_agent(
            _make_config({"message": "{{message}}", "response_path": "data"})
        )
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(
            return_value=_mock_response({"data": {"nested": "value"}})
        )
        with pytest.raises(RuntimeError, match="did not resolve to a string"):
            await agent.execute("q")

    async def test_non_json_response_raises(self) -> None:
        agent = create_agent(
            _make_config({"message": "{{message}}", "response_path": "reply"})
        )
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(
            return_value=httpx.Response(
                200,
                text="not json at all",
                request=httpx.Request("POST", ENDPOINT),
            )
        )
        with pytest.raises(RuntimeError, match="non-JSON"):
            await agent.execute("q")

    async def test_list_index_out_of_range_raises(self) -> None:
        agent = create_agent(
            _make_config(
                {"message": "{{message}}", "response_path": "choices.5.content"}
            )
        )
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(
            return_value=_mock_response({"choices": [{"content": "only-one"}]})
        )
        with pytest.raises(RuntimeError, match="out of range"):
            await agent.execute("q")


class TestResponseExtras:
    """response_extras appends response fields the reply text does not carry."""

    async def test_extras_append_a_labelled_section(self) -> None:
        """The motivating case: ids from a sibling list reaching the turn."""
        agent = create_agent(
            _make_config(
                {
                    "query": "{{message}}",
                    "response_path": "response",
                    "response_extras": [
                        {
                            "path": "product_list",
                            "label": "IDs",
                            "template": "SKU-{product_id}",
                        }
                    ],
                }
            )
        )
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(
            return_value=_mock_response(
                {
                    "response": "Here are some available products",
                    "product_list": [
                        {"product_id": "abc", "product_name": "vacuum"},
                        {"product_id": "bcd", "product_name": "kitchen"},
                    ],
                }
            )
        )
        result = await agent.execute("q")
        assert result.content == (
            "Here are some available products\n\nIDs:\n- SKU-abc\n- SKU-bcd"
        )

    async def test_absent_field_drops_its_section(self) -> None:
        agent = create_agent(
            _make_config(
                {
                    "query": "{{message}}",
                    "response_path": "response",
                    "response_extras": [
                        {"path": "product_list", "label": "Recommended"}
                    ],
                }
            )
        )
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(
            return_value=_mock_response({"response": "Nothing matched."})
        )
        result = await agent.execute("q")
        assert result.content == "Nothing matched."

    async def test_extras_stripped_from_request_body(self) -> None:
        """response_extras is config, not part of the outbound request."""
        agent = create_agent(
            _make_config(
                {
                    "query": "{{message}}",
                    "response_path": "reply",
                    "response_extras": [{"path": "links"}],
                }
            )
        )
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(return_value=_mock_response({"reply": "hi"}))
        await agent.execute("hello")
        sent_body = agent._client.post.call_args.kwargs["json"]
        assert sent_body == {"query": "hello"}
        assert "response_extras" not in sent_body


class TestConfigValidation:
    """CustomHTTPConfig validation enforces the config contract."""

    def test_missing_response_path_rejected(self) -> None:
        with pytest.raises(ValueError, match="response_path"):
            _make_config({"message": "{{message}}"})

    def test_empty_response_path_rejected(self) -> None:
        with pytest.raises(ValueError, match="response_path"):
            _make_config({"message": "{{message}}", "response_path": "   "})

    def test_response_path_without_request_body_rejected(self) -> None:
        with pytest.raises(ValueError, match="request template"):
            _make_config({"response_path": "reply"})

    def test_extras_must_be_a_list(self) -> None:
        with pytest.raises(ValueError, match="must be a list"):
            _make_config(
                {
                    "message": "{{message}}",
                    "response_path": "reply",
                    "response_extras": {"path": "links"},
                }
            )

    def test_extras_entry_needs_a_path(self) -> None:
        with pytest.raises(ValueError, match="entry 0 must have a 'path'"):
            _make_config(
                {
                    "message": "{{message}}",
                    "response_path": "reply",
                    "response_extras": [{"label": "Links"}],
                }
            )

    def test_extras_entry_rejects_unsupported_keys(self) -> None:
        """An unsupported key is a typo or an expectation this cannot meet."""
        with pytest.raises(ValueError, match="unsupported key 'join'"):
            _make_config(
                {
                    "message": "{{message}}",
                    "response_path": "reply",
                    "response_extras": [{"path": "links", "join": ", "}],
                }
            )

    def test_missing_body_rejected(self) -> None:
        with pytest.raises(ValueError, match="body"):
            AgentConfig(
                agent_type="custom_http",
                agent_name="x",
                api_config={"endpoint": ENDPOINT},
            )


class TestRequestHeaders:
    """Per-request headers from the request_headers contextvar."""

    async def test_headers_merged_per_request_and_reset(self) -> None:
        from arksim.tracing.context import reset_request_headers, set_request_headers

        agent = create_agent(_make_config({"q": "{{message}}", "response_path": "r"}))
        agent._client = AsyncMock()
        agent._client.post = AsyncMock(return_value=_mock_response({"r": "hi"}))

        token = set_request_headers({"traceparent": "00-abc-def-01"})
        try:
            await agent.execute("hello")
        finally:
            reset_request_headers(token)
        await agent.execute("again")

        first, second = (
            c.kwargs["headers"] for c in agent._client.post.await_args_list
        )
        assert first == {
            "Authorization": "Bearer test-token",
            "traceparent": "00-abc-def-01",
        }
        assert second == {"Authorization": "Bearer test-token"}
        # The agent's own headers are never mutated.
        assert agent.headers == {"Authorization": "Bearer test-token"}
