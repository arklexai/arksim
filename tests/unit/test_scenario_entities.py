# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.scenario.entities validators."""

import pytest
from pydantic import ValidationError

from arksim.scenario.entities import Scenario


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
