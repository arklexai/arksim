# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from arksim.config import AgentConfig


class BaseAgent:
    """Base class for all agent implementations."""

    def __init__(self, agent_config: AgentConfig) -> None:
        self.agent_config = agent_config

    async def get_chat_id(self) -> str:
        """Get the chat ID. Must be implemented by subclasses."""
        raise NotImplementedError("get_chat_id must be implemented by subclasses")

    async def execute(self, user_query: str, **kwargs: object) -> str:
        """Execute the user query. Must be implemented by subclasses."""
        raise NotImplementedError("execute must be implemented by subclasses")

    async def close(self) -> None:
        """Clean up and close the agent."""
        pass
