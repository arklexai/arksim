# SPDX-License-Identifier: Apache-2.0
"""Integration tests for Chat Completions response parsing.

Verifies that real LLM API responses are parsed into AgentResponse
objects. Tool calls are captured via OTel tracing, not response parsing.
Requires API keys.
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
async def test_openai_tool_response_parsed() -> None:
    """A real OpenAI API call with tools returns a valid AgentResponse.

    Response parsers no longer extract tool calls (that is handled by OTel
    tracing). This test verifies the response is parsed without errors.
    """
    response = await _call_openai(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the weather in Tokyo?"},
        ],
        tools=TOOLS,
    )

    result = parse_response(response)

    assert isinstance(result, AgentResponse)
    assert result.tool_calls == []


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
    """ChatCompletionsAgent.execute() returns AgentResponse."""
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
        # Tool calls are no longer extracted from responses
        assert result.tool_calls == []
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
