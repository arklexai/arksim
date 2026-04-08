# SPDX-License-Identifier: Apache-2.0
"""Starter agent for arksim init. Replace the execute() body with your agent logic."""

from __future__ import annotations

import uuid

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.simulation_engine.tool_types import AgentResponse


class MyAgent(BaseAgent):
    """A minimal agent that echoes back the user's query.

    Replace the execute() method with your own agent logic.
    This class is loaded by arksim via the module_path in config.yaml.
    """

    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)

    async def get_chat_id(self) -> str:
        return str(uuid.uuid4())

    async def execute(self, user_query: str, **kwargs: object) -> str | AgentResponse:
        # Replace this with your agent logic.
        # You have access to self.agent_config for any config you need.
        return f"You said: {user_query}"
