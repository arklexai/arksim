# SPDX-License-Identifier: Apache-2.0
"""Claude Agent SDK integration for ArkSim.

Install: pip install claude-agent-sdk
Auth:    export ANTHROPIC_API_KEY="<your-key>"
"""

from __future__ import annotations

import uuid

from claude_agent_sdk import ClaudeAgentOptions, query

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent


class ClaudeAgentSDKAgent(BaseAgent):
    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._chat_id = str(uuid.uuid4())

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        result = None
        async for message in query(
            prompt=user_query,
            options=ClaudeAgentOptions(allowed_tools=[]),
        ):
            if hasattr(message, "result"):
                result = message.result
        return result or ""
