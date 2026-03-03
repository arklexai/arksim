# SPDX-License-Identifier: Apache-2.0
from arksim.config import AgentConfig, AgentType

from .base import BaseAgent
from .clients.a2a import A2AAgent
from .clients.chat_completions import ChatCompletionsAgent


def create_agent(agent_config: AgentConfig) -> BaseAgent:
    """Factory function to create an agent instance from AgentConfig."""
    agent_type = agent_config.agent_type

    if agent_type == AgentType.CHAT_COMPLETIONS.value:
        return ChatCompletionsAgent(agent_config)
    elif agent_type == AgentType.A2A.value:
        return A2AAgent(agent_config)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
