# SPDX-License-Identifier: Apache-2.0
"""Weather agent using the OpenAI Agents SDK.

invoke() returns both the agent's text answer and the tool calls made
during inference (extracted via the arksim helper). The server embeds
those tool calls in artifact metadata for arksim to extract and evaluate.
"""

from __future__ import annotations

from typing import Any

from agents import Agent as SDKAgent
from agents import Runner, function_tool

from arksim import ToolCall
from arksim.tracing.openai import extract_tool_calls


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
            JSON-serializable dicts ready to embed in an A2A artifact's
            metadata.
        """
        self._history.append({"role": "user", "content": question})
        result = await Runner.run(_sdk_agent, input=self._history)
        answer: str = result.final_output
        self._history.append({"role": "assistant", "content": answer})
        # extract_tool_calls returns ToolCall objects; serialize to plain
        # dicts so the server can include them in artifact metadata as JSON.
        # We exclude ``source`` because arksim sets it on extraction based
        # on the capture path; sending it over the wire would be overwritten
        # and clutters the payload documented in the extension schema.
        tool_calls: list[ToolCall] = extract_tool_calls(result)
        return answer, [
            tc.model_dump(exclude={"source"}, exclude_none=True) for tc in tool_calls
        ]
