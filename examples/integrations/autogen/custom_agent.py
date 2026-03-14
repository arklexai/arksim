# SPDX-License-Identifier: Apache-2.0
"""AutoGen integration for ArkSim.

Install: pip install autogen-agentchat autogen-ext[openai]
Auth:    export OPENAI_API_KEY="<your-key>"
"""

from __future__ import annotations

import uuid

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent


class AutoGenAgent(BaseAgent):
    """AutoGen agent wrapper.

    AssistantAgent maintains internal model context across on_messages
    calls, so only the new user message is passed each turn.
    """

    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._chat_id = str(uuid.uuid4())
        model_client = OpenAIChatCompletionClient(model="gpt-4o")
        self._agent = AssistantAgent(
            name="assistant",
            system_message="You are a helpful assistant.",
            model_client=model_client,
        )

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        response = await self._agent.on_messages(
            [TextMessage(content=user_query, source="user")],
            cancellation_token=CancellationToken(),
        )
        return response.chat_message.content
