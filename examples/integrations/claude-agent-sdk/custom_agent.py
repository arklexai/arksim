# SPDX-License-Identifier: Apache-2.0
"""Claude Agent SDK integration for ArkSim.

Install: pip install claude-agent-sdk
Auth:    export ANTHROPIC_API_KEY="<your-key>"
"""

from __future__ import annotations

import uuid

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent


class ClaudeAgentSDKAgent(BaseAgent):
    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._chat_id = str(uuid.uuid4())
        self._client = ClaudeSDKClient(
            options=ClaudeAgentOptions(allowed_tools=[]),
        )
        self._connected = False

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        if not self._connected:
            await self._client.connect()
            self._connected = True

        await self._client.query(user_query)
        result = ""
        async for message in self._client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        result += block.text
            elif isinstance(message, ResultMessage):
                break
        return result
