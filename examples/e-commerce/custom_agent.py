# SPDX-License-Identifier: Apache-2.0
"""Custom agent connector wrapping the e-commerce RAG agent.

This module exposes the existing Agent as a BaseAgent subclass so it can
be loaded directly by arksim with ``agent_type: custom`` — no HTTP server
required.
"""

from __future__ import annotations

from agent_server.core.agent import Agent

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent


class EcommerceCustomAgent(BaseAgent):
    """BaseAgent wrapper around the e-commerce RAG agent."""

    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._agent = Agent()

    async def get_chat_id(self) -> str:
        return self._agent.context_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        return await self._agent.invoke(user_query)
