# SPDX-License-Identifier: Apache-2.0
"""Tests for simulation engine entities."""

import pytest
from pydantic import ValidationError

from arksim.simulation_engine.entities import (
    ConversationState,
    SimulationParams,
)


class TestSimulationParams:
    """Tests for SimulationParams model."""

    def test_default_values(self) -> None:
        """Test default values."""
        params = SimulationParams()

        assert params.num_convos_per_scenario == 1
        assert params.max_turns == 5
        assert params.num_workers == "auto"
        assert params.output_file_path == "./simulation.json"
        assert params.simulated_user_prompt_template is None

    def test_custom_values(self) -> None:
        """Test custom values."""
        params = SimulationParams(
            num_convos_per_scenario=20,
            max_turns=10,
            num_workers=4,
            output_file_path="/tmp/out.json",
            simulated_user_prompt_template="Hello {{ scenario.goal }}",
        )

        assert params.num_convos_per_scenario == 20
        assert params.max_turns == 10
        assert params.num_workers == 4
        assert params.output_file_path == "/tmp/out.json"
        assert "{{ scenario.goal }}" in params.simulated_user_prompt_template

    def test_all_fields_optional(self) -> None:
        """Test all fields have defaults."""
        params = SimulationParams()

        assert params is not None


class TestConversationState:
    """Tests for ConversationState model."""

    def test_valid_conversation(self) -> None:
        """Test creating valid ConversationState."""
        conv = ConversationState(
            conversation_id="conv-123",
            scenario_id="sc-1",
            conversation_history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            simulated_user_prompt_template="You are a helpful assistant.",
            simulated_user_profile="Young professional",
            user_goal="Get information about products",
            knowledge=["Product catalog knowledge"],
            agent_context="Product support agent",
        )

        assert conv.conversation_id == "conv-123"
        assert len(conv.conversation_history) == 2

    def test_empty_conversation_history_allowed(self) -> None:
        """Test empty conversation history is allowed."""
        conv = ConversationState(
            conversation_id="conv-123",
            scenario_id="sc-1",
            conversation_history=[],
            simulated_user_prompt_template="prompt",
            simulated_user_profile="profile",
            user_goal="goal",
            knowledge=["knowledge"],
            agent_context="context",
        )

        assert conv.conversation_history == []

    def test_empty_knowledge_allowed(self) -> None:
        """Test empty knowledge list is allowed."""
        conv = ConversationState(
            conversation_id="conv-123",
            scenario_id="sc-1",
            conversation_history=[],
            simulated_user_prompt_template="prompt",
            simulated_user_profile="profile",
            user_goal="goal",
            knowledge=[],
            agent_context="context",
        )

        assert conv.knowledge == []

    def test_requires_conversation_id(self) -> None:
        """Test conversation_id is required."""
        with pytest.raises(ValidationError):
            ConversationState(
                scenario_id="sc-1",
                conversation_history=[],
                simulated_user_prompt_template="prompt",
                simulated_user_profile="profile",
                user_goal="goal",
                knowledge=[],
                agent_context="context",
            )

    def test_requires_simulated_user_prompt_template(self) -> None:
        """Test simulated_user_prompt_template is required."""
        with pytest.raises(ValidationError):
            ConversationState(
                conversation_id="conv-123",
                scenario_id="sc-1",
                conversation_history=[],
                simulated_user_profile="profile",
                user_goal="goal",
                knowledge=[],
                agent_context="context",
            )

    def test_complex_conversation_history(self) -> None:
        """Test conversation with complex message structure."""
        conv = ConversationState(
            conversation_id="conv-123",
            scenario_id="sc-1",
            conversation_history=[
                {"role": "system", "content": "System message"},
                {"role": "user", "content": "User question", "metadata": {"ts": 123}},
                {"role": "assistant", "content": "Response", "tool_calls": []},
            ],
            simulated_user_prompt_template="prompt",
            simulated_user_profile="profile",
            user_goal="goal",
            knowledge=["knowledge"],
            agent_context="context",
        )

        assert len(conv.conversation_history) == 3
        assert conv.conversation_history[1]["metadata"]["ts"] == 123
