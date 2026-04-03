# SPDX-License-Identifier: Apache-2.0
"""Integration tests for Chat Completions tool call capture.

Verifies that tool_calls returned by real LLM APIs are parsed into
ToolCall objects with source="response_parse". Requires API keys.
"""

from __future__ import annotations

import os

import pytest

from arksim.simulation_engine.agent.response_parsers import parse_response
from arksim.simulation_engine.tool_types import AgentResponse

requires_openai = pytest.mark.usefixtures("_require_openai_key")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    }
]


@requires_openai
@pytest.mark.asyncio
async def test_openai_tool_calls_captured() -> None:
    """A real OpenAI API call with tools returns tool_calls that arksim captures."""

    response = await _call_openai(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the weather in Tokyo?"},
        ],
        tools=TOOLS,
    )

    result = parse_response(response)

    assert isinstance(result, AgentResponse)
    assert len(result.tool_calls) >= 1, (
        f"Expected at least 1 tool call, got {len(result.tool_calls)}. "
        f"Response content: {result.content!r}"
    )

    tc = result.tool_calls[0]
    assert tc.name == "get_weather"
    assert "city" in tc.arguments
    assert tc.source == "response_parse"
    assert tc.result is None  # Response parsing captures requests only


@requires_openai
@pytest.mark.asyncio
async def test_openai_text_response_has_no_tool_calls() -> None:
    """A regular text response (no tools) produces empty tool_calls."""
    response = await _call_openai(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in Japanese."},
        ],
        tools=None,
    )

    result = parse_response(response)

    assert isinstance(result, AgentResponse)
    assert len(result.tool_calls) == 0
    assert len(result.content) > 0


@requires_openai
@pytest.mark.asyncio
async def test_chat_completions_agent_returns_agent_response() -> None:
    """ChatCompletionsAgent.execute() returns AgentResponse with tool_calls."""
    from arksim.config import AgentConfig
    from arksim.simulation_engine.agent.clients.chat_completions import (
        ChatCompletionsAgent,
    )

    config = AgentConfig(
        agent_type="chat_completions",
        agent_name="test-tool-capture",
        api_config={
            "endpoint": "https://api.openai.com/v1/chat/completions",
            "headers": {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            },
            "body": {
                "model": os.environ.get("TEST_OPENAI_MODEL", "gpt-4.1-mini"),
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    }
                ],
                "tools": TOOLS,
            },
        },
    )

    agent = ChatCompletionsAgent(config)
    try:
        result = await agent.execute("What is the weather in Paris?")

        assert isinstance(result, AgentResponse)
        assert len(result.tool_calls) >= 1
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].source == "response_parse"
    finally:
        await agent.close()


async def _call_openai(
    messages: list[dict],
    tools: list[dict] | None = None,
) -> dict:
    """Make a real OpenAI API call and return the raw response dict."""
    import httpx

    payload: dict = {
        "model": os.environ.get("TEST_OPENAI_MODEL", "gpt-4.1-mini"),
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            },
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()
