# SPDX-License-Identifier: Apache-2.0
"""Customer-service agent returning explicit AgentResponse with tool calls.

Demonstrates the Python connector's explicit capture path: the agent
calls ``extract_tool_calls`` on the SDK's ``RunResult`` and returns
``AgentResponse(content=..., tool_calls=[...])``.

Tools and database setup are in ``tools.py`` (shared with the traced
and A2A variants of this example).

Install: pip install openai-agents
Auth:    export OPENAI_API_KEY="<your-key>"
"""

from __future__ import annotations

import uuid

from agents import Agent, Runner, RunResult
from tools import AGENT_INSTRUCTIONS, TOOLS_LIST, init_db

from arksim import AgentConfig, AgentResponse
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.tracing.openai import extract_tool_calls


class ToolCallExampleAgent(BaseAgent):
    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        init_db()
        self._chat_id = str(uuid.uuid4())
        self._agent = Agent(
            name="assistant",
            instructions=AGENT_INSTRUCTIONS,
            tools=TOOLS_LIST,
        )
        self._last_result: RunResult | None = None

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> AgentResponse:
        if self._last_result is not None:
            input_list = self._last_result.to_input_list() + [
                {"role": "user", "content": user_query}
            ]
        else:
            input_list = [{"role": "user", "content": user_query}]

        self._last_result = await Runner.run(self._agent, input=input_list)

        return AgentResponse(
            content=self._last_result.final_output,
            tool_calls=extract_tool_calls(self._last_result),
        )
