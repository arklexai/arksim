# SPDX-License-Identifier: Apache-2.0
"""Weather agent using the OpenAI Agents SDK.

Demonstrates the A2A tool call capture pattern: invoke() returns both the
agent's text answer and the tool calls made during inference, so the server
can embed them in a DataPart for arksim to extract and evaluate.
"""

from __future__ import annotations

import json
from typing import Any

from agents import Agent as SDKAgent
from agents import Runner, RunResult, function_tool
from agents.items import ToolCallItem, ToolCallOutputItem
from openai.types.responses import ResponseFunctionToolCall


@function_tool
def get_weather(city: str) -> str:
    """Get current weather for a city.

    Args:
        city: The name of the city to query.

    Returns:
        A short weather summary for the requested city.
    """
    return f"Weather in {city}: 72F, sunny"


_SYSTEM_INSTRUCTIONS = (
    "You are a helpful weather assistant. "
    "When the user asks about weather in a specific location, call the "
    "get_weather tool and report the result. "
    "Keep responses concise."
)

_sdk_agent = SDKAgent(
    name="WeatherAgent",
    instructions=_SYSTEM_INSTRUCTIONS,
    tools=[get_weather],
    model="gpt-4.1-mini",
)


def _extract_tool_calls(result: RunResult) -> list[dict[str, Any]]:
    """Pair ToolCallItems with their ToolCallOutputItems from a RunResult.

    Iterates result.new_items twice: first to collect outputs keyed by
    call_id, then to build the call list in invocation order.

    Args:
        result: The RunResult from Runner.run().

    Returns:
        A list of dicts with keys: id, name, arguments, result.
    """
    outputs: dict[str, str] = {}
    for item in result.new_items:
        if isinstance(item, ToolCallOutputItem):
            raw = item.raw_item
            call_id = (
                raw.get("call_id", "")
                if isinstance(raw, dict)
                else getattr(raw, "call_id", "")
            )
            output = (
                raw.get("output", "")
                if isinstance(raw, dict)
                else getattr(raw, "output", "")
            )
            if isinstance(output, list):
                output = json.dumps(output)
            outputs[call_id] = str(output)

    tool_calls: list[dict[str, Any]] = []
    for item in result.new_items:
        if not isinstance(item, ToolCallItem):
            continue
        raw = item.raw_item
        if not isinstance(raw, ResponseFunctionToolCall):
            continue
        call_id = raw.call_id
        tool_calls.append(
            {
                "id": call_id,
                "name": raw.name,
                "arguments": json.loads(raw.arguments) if raw.arguments else {},
                "result": outputs.get(call_id),
            }
        )

    return tool_calls


class Agent:
    """Stateful per-session weather agent.

    Maintains conversation history across turns so the model has multi-turn
    context. Each invoke() call returns both the agent's text answer and the
    tool calls made during that turn.
    """

    def __init__(self) -> None:
        self._history: list[dict[str, Any]] = []

    async def invoke(self, question: str) -> tuple[str, list[dict[str, Any]]]:
        """Process a user message and return the answer with any tool calls.

        Args:
            question: The user's latest message.

        Returns:
            A tuple of (answer, tool_calls) where tool_calls is a list of
            dicts with keys: id, name, arguments, result.
        """
        self._history.append({"role": "user", "content": question})
        result = await Runner.run(_sdk_agent, input=self._history)
        answer: str = result.final_output
        self._history.append({"role": "assistant", "content": answer})
        tool_calls = _extract_tool_calls(result)
        return answer, tool_calls
