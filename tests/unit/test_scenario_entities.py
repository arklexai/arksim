# SPDX-License-Identifier: Apache-2.0
"""Tests for scenario entities."""

import json
import os

import pytest
from pydantic import ValidationError

from arksim.scenario.entities import KnowledgeItem, Scenario, Scenarios


class TestKnowledgeItem:
    """Tests for KnowledgeItem model."""

    def test_default_content(self) -> None:
        """Test default content is empty string."""
        ki = KnowledgeItem()
        assert ki.content == ""

    def test_custom_content(self) -> None:
        """Test custom content."""
        ki = KnowledgeItem(content="Some knowledge")
        assert ki.content == "Some knowledge"


class TestScenario:
    """Tests for Scenario model (uses model_validator before + classmethod)."""

    def test_valid_scenario(self) -> None:
        """Test creating a valid Scenario."""
        s = Scenario(
            scenario_id="sc-1",
            user_id="u-1",
            goal="Get help with billing",
            agent_context="Billing support agent",
            knowledge=[KnowledgeItem(content="Billing FAQ")],
            user_profile="Frustrated customer",
            origin={"source": "manual"},
        )
        assert s.scenario_id == "sc-1"
        assert s.user_id == "u-1"
        assert s.goal == "Get help with billing"
        assert len(s.knowledge) == 1
        assert s.user_profile == "Frustrated customer"

    def test_empty_knowledge(self) -> None:
        """Test scenario with empty knowledge list."""
        s = Scenario(
            scenario_id="sc-1",
            user_id="u-1",
            goal="goal",
            agent_context="ctx",
            knowledge=[],
            user_profile="profile",
            origin={},
        )
        assert s.knowledge == []

    def test_legacy_user_attributes_without_profile_raises(self) -> None:
        """Test that user_attributes without user_profile raises error."""
        with pytest.raises(ValidationError, match="user_attributes"):
            Scenario(
                scenario_id="sc-1",
                user_id="u-1",
                goal="goal",
                agent_context="ctx",
                knowledge=[],
                user_profile="",
                origin={},
                user_attributes={"age": "30"},
            )

    def test_legacy_user_attributes_with_profile_ok(self) -> None:
        """Test that user_attributes WITH user_profile is accepted."""
        s = Scenario(
            scenario_id="sc-1",
            user_id="u-1",
            goal="goal",
            agent_context="ctx",
            knowledge=[],
            user_profile="Has a profile",
            origin={},
            user_attributes={"age": "30"},
        )
        assert s.user_profile == "Has a profile"

    def test_requires_scenario_id(self) -> None:
        """Test scenario_id is required."""
        with pytest.raises(ValidationError):
            Scenario(
                user_id="u-1",
                goal="goal",
                agent_context="ctx",
                knowledge=[],
                user_profile="profile",
                origin={},
            )


class TestScenarios:
    """Tests for Scenarios model."""

    def test_valid_scenarios(self) -> None:
        """Test creating a valid Scenarios container."""
        scenarios = Scenarios(
            schema_version="1.0",
            scenarios=[
                Scenario(
                    scenario_id="sc-1",
                    user_id="u-1",
                    goal="goal",
                    agent_context="ctx",
                    knowledge=[],
                    user_profile="profile",
                    origin={},
                ),
            ],
        )
        assert scenarios.schema_version == "1.0"
        assert len(scenarios.scenarios) == 1

    def test_empty_scenarios(self) -> None:
        """Test Scenarios with empty list."""
        scenarios = Scenarios(schema_version="1.0", scenarios=[])
        assert scenarios.scenarios == []

    def test_load_from_file(self, tmp_path: str) -> None:
        """Test loading scenarios from a JSON file."""
        data = {
            "schema_version": "1.0",
            "scenarios": [
                {
                    "scenario_id": "sc-1",
                    "user_id": "u-1",
                    "goal": "goal",
                    "agent_context": "ctx",
                    "knowledge": [{"content": "kb"}],
                    "user_profile": "profile",
                    "origin": {},
                }
            ],
        }
        path = os.path.join(str(tmp_path), "scenarios.json")
        with open(path, "w") as f:
            json.dump(data, f)

        scenarios = Scenarios.load(path)
        assert len(scenarios.scenarios) == 1
        assert scenarios.scenarios[0].scenario_id == "sc-1"
