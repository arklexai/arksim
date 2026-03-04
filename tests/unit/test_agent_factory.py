# SPDX-License-Identifier: Apache-2.0
"""Tests for simulation engine agent factory."""

import pytest

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.simulation_engine.agent.clients.a2a import A2AAgent
from arksim.simulation_engine.agent.clients.chat_completions import ChatCompletionsAgent
from arksim.simulation_engine.agent.factory import create_agent

try:
    from arksim.simulation_engine.agent.a2a import A2AAgent

    _has_a2a = True
except ImportError:
    _has_a2a = False


class TestCreateAgent:
    """Tests for create_agent factory function."""

    def test_creates_chat_completions_agent(
        self, valid_agent_config_chat_completions_new: dict
    ) -> None:
        """Test creates ChatCompletionsAgent for chat_completions type."""
        config = AgentConfig(**valid_agent_config_chat_completions_new)

        agent = create_agent(config)

        assert isinstance(agent, ChatCompletionsAgent)
        assert isinstance(agent, BaseAgent)

    @pytest.mark.skipif(not _has_a2a, reason="a2a package not installed")
    def test_creates_a2a_agent(self, valid_agent_config_a2a: dict) -> None:
        """Test creates A2AAgent for a2a type."""
        config = AgentConfig(**valid_agent_config_a2a)

        agent = create_agent(config)

        assert isinstance(agent, A2AAgent)
        assert isinstance(agent, BaseAgent)

    def test_agent_has_config(self, valid_agent_config_a2a: dict) -> None:
        """Test created agent has access to its config."""
        config = AgentConfig(**valid_agent_config_a2a)

        agent = create_agent(config)

        assert agent.agent_config == config
        assert agent.agent_config.agent_name == "test-a2a-agent"

    def test_unsupported_agent_type_raises(self) -> None:
        """Test unsupported agent type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported agent type"):
            AgentConfig(
                agent_type="unsupported",
                agent_name="test",
                api_config={},
            )


class _ConcreteAgent(BaseAgent):
    """Minimal concrete subclass for testing."""

    async def get_chat_id(self) -> str:
        return "test-id"

    async def execute(self, user_query: str, **kwargs: object) -> str:
        return "response"


class TestBaseAgent:
    """Tests for BaseAgent base class."""

    def test_cannot_instantiate_directly(self, valid_agent_config_a2a: dict) -> None:
        """Test BaseAgent cannot be instantiated (abstract)."""
        config = AgentConfig(**valid_agent_config_a2a)
        with pytest.raises(TypeError, match="abstract"):
            BaseAgent(config)

    def test_init_stores_config(self, valid_agent_config_a2a: dict) -> None:
        """Test concrete subclass stores config on init."""
        config = AgentConfig(**valid_agent_config_a2a)
        agent = _ConcreteAgent(config)
        assert agent.agent_config == config

    async def test_close_default_implementation(
        self, valid_agent_config_a2a: dict
    ) -> None:
        """Test close has default no-op implementation."""
        config = AgentConfig(**valid_agent_config_a2a)
        agent = _ConcreteAgent(config)
        await agent.close()  # Should not raise
