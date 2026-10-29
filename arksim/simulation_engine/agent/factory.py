# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable

import httpx

from arksim.config import AgentConfig, AgentType

from .base import BaseAgent
from .clients.a2a import A2AAgent
from .clients.chat_completions import ChatCompletionsAgent
from .clients.custom import CustomAgent
from .clients.custom_http import CustomHTTPAgent


def create_agent(
    agent_config: AgentConfig,
    *,
    http_client_factory: Callable[..., httpx.AsyncClient] | None = None,
) -> BaseAgent:
    """Factory function to create an agent instance from AgentConfig."""
    agent_type = agent_config.agent_type

    if agent_type == AgentType.CHAT_COMPLETIONS.value:
        return ChatCompletionsAgent(
            agent_config, http_client_factory=http_client_factory
        )
    elif agent_type == AgentType.A2A.value:
        return A2AAgent(agent_config, http_client_factory=http_client_factory)
    elif agent_type == AgentType.CUSTOM.value:
        return CustomAgent(agent_config)
    elif agent_type == AgentType.CUSTOM_HTTP.value:
        return CustomHTTPAgent(agent_config, http_client_factory=http_client_factory)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
