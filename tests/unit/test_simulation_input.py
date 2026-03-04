# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.simulation_engine.entities.SimulationInput validator."""

import pytest
from pydantic import ValidationError

from arksim.simulation_engine.entities import SimulationInput


class TestSimulationInputValidator:
    def test_valid_with_config_file(self) -> None:
        si = SimulationInput(agent_config_file_path="agent.json")
        assert si.agent_config_file_path == "agent.json"

    def test_no_agent_config_raises(self) -> None:
        with pytest.raises(ValidationError, match="agent_config"):
            SimulationInput()

    def test_skip_validation_context(self) -> None:
        si = SimulationInput.model_validate(
            {}, context={"skip_input_dir_validation": True}
        )
        assert si.agent_config is None

    def test_invalid_num_workers_string(self) -> None:
        with pytest.raises(ValidationError, match="num_workers"):
            SimulationInput(agent_config_file_path="a.json", num_workers="fast")
