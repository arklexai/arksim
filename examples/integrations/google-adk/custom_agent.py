# SPDX-License-Identifier: Apache-2.0
"""Google ADK integration for arksim.

Install:
    pip install 'arksim[google-adk]'
Auth:
    export GOOGLE_API_KEY="<your-key>"

Wires arksim's Google ADK tracing adapter into an ``LlmAgent`` with two
mock tools (lookup_order, book_table). Running ``arksim simulate-evaluate``
produces a simulation.json whose ``tool_calls`` field is populated by the
captured invocations.
"""

from __future__ import annotations

import uuid

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types
from tools import book_table, lookup_order

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.tracing.integrations.google_adk import ArksimADKPlugin


class GoogleADKAgent(BaseAgent):
    """ADK ``LlmAgent`` with conversation history and tool-call tracing.

    ``ArksimADKPlugin`` extends ``BasePlugin`` and overrides
    ``after_tool_callback`` to emit one ``ToolCall`` per invocation. It is
    registered on the runner via ``plugins=[...]`` rather than on the agent,
    so the same instance can cover any sub-agents the ``LlmAgent`` spawns.
    """

    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._plugin = ArksimADKPlugin()
        adk_agent = LlmAgent(
            name="assistant",
            model="gemini-2.5-flash",
            instruction=(
                "You are a helpful assistant with access to two tools: "
                "lookup_order(order_id) and book_table(party_size, time). "
                "Use them when relevant to answer the user."
            ),
            tools=[lookup_order, book_table],
        )
        self._runner = InMemoryRunner(
            agent=adk_agent,
            app_name="arksim",
            plugins=[self._plugin],
        )
        self._user_id = f"arksim_{uuid.uuid4()}"
        self._session_id: str | None = None

    async def get_chat_id(self) -> str:
        if self._session_id is None:
            session = await self._runner.session_service.create_session(
                app_name="arksim",
                user_id=self._user_id,
            )
            self._session_id = session.id
        return self._session_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        if self._session_id is None:
            await self.get_chat_id()

        content = types.Content(role="user", parts=[types.Part(text=user_query)])
        result = ""
        async for event in self._runner.run_async(
            user_id=self._user_id,
            session_id=self._session_id,
            new_message=content,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                result = event.content.parts[0].text
        return result
