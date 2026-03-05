# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from abc import ABC, abstractmethod

from arksim.config import AgentConfig


class BaseAgent(ABC):
    """Base class for all agent implementations."""

    def __init__(self, agent_config: AgentConfig) -> None:
        self.agent_config = agent_config
        self.tool_call_result: str = '{"status": "ok"}'
        self.max_tool_call_rounds: int = 10

    @abstractmethod
    async def get_chat_id(self) -> str:
        """Get the chat ID. Must be implemented by subclasses."""

    @abstractmethod
    async def execute(self, user_query: str, **kwargs: object) -> str:
        """Execute the user query. Must be implemented by subclasses."""

    async def close(self) -> None:  # noqa: B027 - intentional non-abstract no-op
        """Clean up and close the agent. Override to release resources."""
