# SPDX-License-Identifier: Apache-2.0
"""Stateful per-session agent for the A2A customer-service server.

Wraps the shared tools from ``tools.py`` in an OpenAI Agents SDK runner.
Each ``invoke()`` call returns the agent's text answer and the tool calls
made during that turn (serialized as plain dicts for artifact metadata).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from agents import Agent as SDKAgent
from agents import Runner

from arksim import ToolCall
from arksim.tracing.openai import extract_tool_calls

# Add parent directory so we can import the shared tools module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import AGENT_INSTRUCTIONS, TOOLS_LIST, init_db  # noqa: E402

_sdk_agent = SDKAgent(
    name="assistant",
    instructions=AGENT_INSTRUCTIONS,
    tools=TOOLS_LIST,
)


class Agent:
    """Stateful per-session customer-service agent.

    Maintains conversation history across turns so the model has
    multi-turn context.
    """

    def __init__(self) -> None:
        init_db()
        self._history: list[dict[str, Any]] = []

    async def invoke(self, question: str) -> tuple[str, list[dict[str, Any]]]:
        """Process a user message and return the answer with any tool calls.

        Returns:
            A tuple of (answer, tool_calls) where tool_calls is a list of
            JSON-serializable dicts ready for artifact metadata. The
            ``source`` field is excluded because arksim sets it client-side.
        """
        self._history.append({"role": "user", "content": question})
        result = await Runner.run(_sdk_agent, input=self._history)
        answer: str = result.final_output
        self._history.append({"role": "assistant", "content": answer})
        tool_calls: list[ToolCall] = extract_tool_calls(result)
        return answer, [
            tc.model_dump(exclude={"source"}, exclude_none=True) for tc in tool_calls
        ]
