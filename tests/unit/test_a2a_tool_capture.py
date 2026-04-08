# SPDX-License-Identifier: Apache-2.0
"""Tests for A2A tool call extraction from DataPart."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from a2a.types import DataPart, Message, Part, Role, TextPart

from arksim.simulation_engine.agent.clients.a2a import A2AAgent
from arksim.simulation_engine.tool_types import AgentResponse, ToolCallSource


def _make_message(
    parts: list[Part],
    context_id: str = "test-ctx",
) -> Message:
    """Build a Message with the given parts."""
    return Message(
        role=Role.agent,
        parts=parts,
        message_id=str(uuid.uuid4()),
        context_id=context_id,
    )


def _text_part(text: str) -> Part:
    return Part(root=TextPart(text=text))


def _data_part(data: dict[str, Any]) -> Part:
    return Part(root=DataPart(data=data))


class TestExtractToolCallsFromDataPart:
    """Tests for extracting tool calls from A2A Message parts."""

    def test_text_and_datapart_with_tool_calls(self) -> None:
        """Message with TextPart + DataPart(tool_calls) yields AgentResponse."""
        msg = _make_message(
            parts=[
                _text_part("The weather in Tokyo is 72F and sunny."),
                _data_part(
                    {
                        "tool_calls": [
                            {
                                "id": "call_001",
                                "name": "get_weather",
                                "arguments": {"city": "Tokyo"},
                                "result": "Weather in Tokyo: 72F, sunny",
                            }
                        ]
                    }
                ),
            ]
        )
        response = A2AAgent._extract_response(msg)

        assert isinstance(response, AgentResponse)
        assert response.content == "The weather in Tokyo is 72F and sunny."
        assert len(response.tool_calls) == 1

        tc = response.tool_calls[0]
        assert tc.id == "call_001"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "Tokyo"}
        assert tc.result == "Weather in Tokyo: 72F, sunny"

    def test_text_only_message_returns_empty_tool_calls(self) -> None:
        """Message with only TextPart yields AgentResponse with empty tool_calls."""
        msg = _make_message(parts=[_text_part("Hello, world!")])
        response = A2AAgent._extract_response(msg)

        assert response.content == "Hello, world!"
        assert response.tool_calls == []

    def test_datapart_without_tool_calls_key_is_ignored(self) -> None:
        """DataPart with other data (no 'tool_calls' key) is ignored."""
        msg = _make_message(
            parts=[
                _text_part("Some text."),
                _data_part({"metadata": {"session": "abc"}}),
            ]
        )
        response = A2AAgent._extract_response(msg)

        assert response.content == "Some text."
        assert response.tool_calls == []

    def test_tool_calls_have_source_field(self) -> None:
        """Extracted ToolCall objects have source='a2a_protocol'."""
        msg = _make_message(
            parts=[
                _text_part("Result."),
                _data_part(
                    {
                        "tool_calls": [
                            {
                                "id": "call_a",
                                "name": "lookup",
                                "arguments": {"q": "test"},
                                "result": "found",
                            }
                        ]
                    }
                ),
            ]
        )
        response = A2AAgent._extract_response(msg)

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].source == ToolCallSource.A2A_PROTOCOL

    def test_multiple_tool_calls_in_single_datapart(self) -> None:
        """Multiple tool calls in one DataPart are all extracted."""
        msg = _make_message(
            parts=[
                _text_part("Weather report."),
                _data_part(
                    {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "name": "get_weather",
                                "arguments": {"city": "Tokyo"},
                                "result": "72F",
                            },
                            {
                                "id": "call_2",
                                "name": "get_weather",
                                "arguments": {"city": "London"},
                                "result": "55F",
                            },
                        ]
                    }
                ),
            ]
        )
        response = A2AAgent._extract_response(msg)

        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[1].name == "get_weather"
        assert response.tool_calls[0].arguments == {"city": "Tokyo"}
        assert response.tool_calls[1].arguments == {"city": "London"}

    def test_multiple_dataparts_across_parts(self) -> None:
        """Tool calls from multiple DataParts are all collected."""
        msg = _make_message(
            parts=[
                _text_part("Results."),
                _data_part(
                    {
                        "tool_calls": [
                            {
                                "id": "call_x",
                                "name": "tool_a",
                                "arguments": {},
                                "result": "a_result",
                            }
                        ]
                    }
                ),
                _data_part(
                    {
                        "tool_calls": [
                            {
                                "id": "call_y",
                                "name": "tool_b",
                                "arguments": {"key": "val"},
                                "result": "b_result",
                            }
                        ]
                    }
                ),
            ]
        )
        response = A2AAgent._extract_response(msg)

        assert len(response.tool_calls) == 2
        names = {tc.name for tc in response.tool_calls}
        assert names == {"tool_a", "tool_b"}

    def test_empty_parts_list(self) -> None:
        """Message with empty parts yields empty content and no tool calls."""
        msg = _make_message(parts=[])
        response = A2AAgent._extract_response(msg)

        assert response.content == ""
        assert response.tool_calls == []

    def test_malformed_tool_call_missing_name(self) -> None:
        """Tool call entry missing required 'name' field is skipped."""
        msg = _make_message(
            parts=[
                _text_part("Partial data."),
                _data_part(
                    {
                        "tool_calls": [
                            {"id": "call_bad", "arguments": {"x": 1}},
                            {
                                "id": "call_good",
                                "name": "valid_tool",
                                "arguments": {},
                            },
                        ]
                    }
                ),
            ]
        )
        response = A2AAgent._extract_response(msg)

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "valid_tool"

    def test_tool_call_with_none_result(self) -> None:
        """Tool call with no result field defaults to None."""
        msg = _make_message(
            parts=[
                _text_part("Done."),
                _data_part(
                    {
                        "tool_calls": [
                            {
                                "id": "call_nr",
                                "name": "fire_and_forget",
                                "arguments": {"action": "notify"},
                            }
                        ]
                    }
                ),
            ]
        )
        response = A2AAgent._extract_response(msg)

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].result is None

    def test_tool_call_with_error_field(self) -> None:
        """Tool call with error field is preserved."""
        msg = _make_message(
            parts=[
                _text_part("Error occurred."),
                _data_part(
                    {
                        "tool_calls": [
                            {
                                "id": "call_err",
                                "name": "failing_tool",
                                "arguments": {},
                                "error": "ConnectionTimeout",
                            }
                        ]
                    }
                ),
            ]
        )
        response = A2AAgent._extract_response(msg)

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].error == "ConnectionTimeout"

    def test_tool_calls_not_a_list_is_ignored(self) -> None:
        """DataPart with tool_calls as a string is safely ignored."""
        msg = _make_message(
            parts=[
                _text_part("Text."),
                _data_part({"tool_calls": "not_a_list"}),
            ]
        )
        response = A2AAgent._extract_response(msg)

        assert response.content == "Text."
        assert response.tool_calls == []

    def test_tool_call_entry_not_a_dict_is_skipped(self) -> None:
        """Non-dict entries in tool_calls list are skipped."""
        msg = _make_message(
            parts=[
                _text_part("Text."),
                _data_part(
                    {
                        "tool_calls": [
                            42,
                            {
                                "id": "call_ok",
                                "name": "valid",
                                "arguments": {},
                            },
                        ]
                    }
                ),
            ]
        )
        response = A2AAgent._extract_response(msg)

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "valid"

    def test_arguments_not_a_dict_is_skipped(self) -> None:
        """Tool call with non-dict arguments is skipped."""
        msg = _make_message(
            parts=[
                _text_part("Text."),
                _data_part(
                    {
                        "tool_calls": [
                            {
                                "id": "call_bad",
                                "name": "bad_args",
                                "arguments": "not_a_dict",
                            },
                            {
                                "id": "call_ok",
                                "name": "good_args",
                                "arguments": {"key": "val"},
                            },
                        ]
                    }
                ),
            ]
        )
        response = A2AAgent._extract_response(msg)

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "good_args"


class TestA2AExecuteReturnsAgentResponse:
    """Tests for A2AAgent.execute() returning AgentResponse with tool calls."""

    @pytest.fixture()
    def agent_config(self) -> MagicMock:
        """Create a mock AgentConfig for A2A."""
        config = MagicMock()
        config.agent_type = "a2a"
        config.api_config = MagicMock()
        config.api_config.endpoint = "http://localhost:9998"
        config.api_config.get_headers.return_value = {}
        return config

    @pytest.mark.asyncio()
    async def test_execute_returns_agent_response_with_tool_calls(
        self, agent_config: MagicMock
    ) -> None:
        """execute() returns AgentResponse with tool_calls from DataPart."""
        agent = A2AAgent(agent_config)

        msg = _make_message(
            parts=[
                _text_part("Weather is sunny."),
                _data_part(
                    {
                        "tool_calls": [
                            {
                                "id": "call_w",
                                "name": "get_weather",
                                "arguments": {"city": "NYC"},
                                "result": "Sunny, 75F",
                            }
                        ]
                    }
                ),
            ]
        )

        async def mock_send_message(_payload: object) -> Any:  # noqa: ANN401
            """Async generator yielding a single Message."""
            yield msg

        mock_client = MagicMock()
        mock_client.send_message = mock_send_message

        agent._client = mock_client
        agent._initialized = True

        result = await agent.execute("What is the weather in NYC?")

        assert isinstance(result, AgentResponse)
        assert result.content == "Weather is sunny."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].source == ToolCallSource.A2A_PROTOCOL

    @pytest.mark.asyncio()
    async def test_execute_text_only_returns_agent_response(
        self, agent_config: MagicMock
    ) -> None:
        """execute() returns AgentResponse with empty tool_calls for text-only."""
        agent = A2AAgent(agent_config)

        msg = _make_message(parts=[_text_part("Just text.")])

        async def mock_send_message(_payload: object) -> Any:  # noqa: ANN401
            yield msg

        mock_client = MagicMock()
        mock_client.send_message = mock_send_message

        agent._client = mock_client
        agent._initialized = True

        result = await agent.execute("Hello")

        assert isinstance(result, AgentResponse)
        assert result.content == "Just text."
        assert result.tool_calls == []
