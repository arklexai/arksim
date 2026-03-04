# SPDX-License-Identifier: Apache-2.0
"""Tests for simulation engine entities."""

import pytest
from pydantic import ValidationError

from arksim.simulation_engine.entities import (
    Conversation,
    ConversationState,
    Message,
    SimulatedUserPrompt,
    Simulation,
    SimulationInput,
    SimulationParams,
)


class TestSimulationInput:
    """Tests for SimulationInput model (uses model_validator + Self)."""

    def test_valid_with_config_file(self) -> None:
        """Test creating SimulationInput with agent_config_file_path."""
        si = SimulationInput(
            agent_config_file_path="./agent.json",
            scenario_file_path="./scenarios.json",
        )
        assert si.agent_config_file_path == "./agent.json"
        assert si.scenario_file_path == "./scenarios.json"

    def test_defaults(self) -> None:
        """Test default field values."""
        si = SimulationInput.model_validate(
            {"agent_config_file_path": "./agent.json"},
            context={"skip_input_dir_validation": True},
        )
        assert si.num_conversations_per_scenario == 5
        assert si.max_turns == 5
        assert si.num_workers == "auto"
        assert si.output_file_path == "./simulation.json"
        assert si.simulated_user_prompt_template is None

    def test_missing_agent_config_raises(self) -> None:
        """Test that missing both agent_config and agent_config_file_path raises."""
        with pytest.raises(ValidationError):
            SimulationInput(scenario_file_path="./scenarios.json")

    def test_skip_validation_in_pipeline(self) -> None:
        """Test validation skip in pipeline mode."""
        si = SimulationInput.model_validate(
            {},
            context={"skip_input_dir_validation": True},
        )
        assert si.agent_config is None
        assert si.agent_config_file_path is None

    def test_invalid_num_workers(self) -> None:
        """Test invalid num_workers raises."""
        with pytest.raises(ValidationError):
            SimulationInput(
                agent_config_file_path="./agent.json",
                num_workers="bad",
            )

    def test_custom_values(self) -> None:
        """Test custom values."""
        si = SimulationInput(
            agent_config_file_path="./agent.json",
            num_conversations_per_scenario=10,
            max_turns=3,
            num_workers=2,
            output_file_path="/tmp/sim.json",
        )
        assert si.num_conversations_per_scenario == 10
        assert si.max_turns == 3
        assert si.num_workers == 2
        assert si.output_file_path == "/tmp/sim.json"


class TestMessage:
    """Tests for Message model."""

    def test_valid_message(self) -> None:
        """Test creating a valid Message."""
        msg = Message(turn_id=1, role="simulated_user", content="Hello")
        assert msg.turn_id == 1
        assert msg.role == "simulated_user"
        assert msg.content == "Hello"
        assert msg.message_id  # auto-generated UUID

    def test_invalid_role(self) -> None:
        """Test invalid role raises."""
        with pytest.raises(ValidationError):
            Message(turn_id=1, role="invalid", content="test")


class TestConversation:
    """Tests for Conversation model."""

    def test_valid_conversation(self) -> None:
        """Test creating a valid Conversation."""
        conv = Conversation(
            conversation_id="conv-1",
            scenario_id="sc-1",
            conversation_history=[
                Message(turn_id=1, role="simulated_user", content="Hi"),
                Message(turn_id=1, role="assistant", content="Hello!"),
            ],
            simulated_user_prompt=SimulatedUserPrompt(
                simulated_user_prompt_template="template",
                variables={"key": "val"},
            ),
        )
        assert conv.conversation_id == "conv-1"
        assert len(conv.conversation_history) == 2


class TestSimulation:
    """Tests for Simulation model."""

    def test_valid_simulation(self) -> None:
        """Test creating a valid Simulation."""
        sim = Simulation(
            schema_version="1.0",
            simulator_version="0.1",
            conversations=[],
        )
        assert sim.schema_version == "1.0"
        assert sim.simulation_id  # auto-generated
        assert sim.generated_at  # auto-generated


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
