# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.scenario.entities validators."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from arksim.scenario.entities import (
    BehaviorParams,
    GoalChecklistItem,
    Scenario,
)


class TestScenarioValidator:
    def _base(self) -> dict:
        return {
            "scenario_id": "sc-1",
            "user_id": "u-1",
            "goal": "buy a ticket",
            "agent_context": "travel agent",
            "knowledge": [{"content": "info"}],
            "user_profile": "frequent traveler",
            "origin": {"source": "manual"},
        }

    def test_valid_scenario(self) -> None:
        s = Scenario(**self._base())
        assert s.scenario_id == "sc-1"

    def test_user_attributes_with_profile_ok(self) -> None:
        data = self._base()
        data["user_attributes"] = {"age": 30}
        s = Scenario(**data)
        assert s.user_profile == "frequent traveler"

    def test_user_attributes_without_profile_raises(self) -> None:
        data = self._base()
        data["user_profile"] = ""
        data["user_attributes"] = {"age": 30}
        with pytest.raises(ValidationError, match="user_attributes"):
            Scenario(**data)

    def test_no_user_attributes_no_profile_ok(self) -> None:
        data = self._base()
        data["user_profile"] = ""
        s = Scenario(**data)
        assert s.user_profile == ""

    def test_without_knowledge_and_origin(self) -> None:
        data = self._base()
        del data["knowledge"]
        del data["origin"]
        s = Scenario(**data)
        assert s.knowledge == []
        assert s.origin == {}


class TestCompiledScenarioFields:
    def _base(self) -> dict:
        return {
            "scenario_id": "sc-1",
            "user_id": "u-1",
            "goal": "buy a ticket",
            "agent_context": "travel agent",
            "user_profile": "frequent traveler",
        }

    def test_uncompiled_scenario_defaults(self) -> None:
        s = Scenario(**self._base())
        assert s.behavior is None
        assert s.goal_checklist == []

    def test_compiled_scenario_round_trip(self) -> None:
        data = self._base()
        data["knowledge"] = [{"id": "kc_01", "content": "trains run hourly"}]
        data["behavior"] = {"patience": 2}
        data["goal_checklist"] = [
            {
                "description": "Learn how often trains run",
                "satisfied_when": "The agent states a specific interval",
                "knowledge_chunk_ids": ["kc_01"],
            }
        ]

        s = Scenario(**data)
        assert s.behavior == BehaviorParams(patience=2)
        assert s.knowledge[0].id == "kc_01"
        item = s.goal_checklist[0]
        assert item == GoalChecklistItem(
            type="knowledge_answer",
            description="Learn how often trains run",
            satisfied_when="The agent states a specific interval",
            mandatory=True,
            knowledge_chunk_ids=["kc_01"],
        )
        assert Scenario(**s.model_dump()) == s
