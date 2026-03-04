# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.simulation_engine.entities.SimulationInput validator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


class TestSimulationInputPathResolution:
    """Tests for config-relative path resolution in SimulationInput."""

    def _base_data(self, **kwargs: Any) -> dict:
        return {"agent_config_file_path": "agent.json", **kwargs}

    def _ctx(self, tmp_path: Path, **kwargs: Any) -> dict:
        return {"config_path": str(tmp_path / "config.yaml"), **kwargs}

    def test_scenario_resolves_to_config_relative(self, tmp_path: Path) -> None:
        """scenario_file_path is always resolved relative to config dir."""
        si = SimulationInput.model_validate(
            self._base_data(scenario_file_path="./scenarios.json"),
            context=self._ctx(tmp_path),
        )
        assert si.scenario_file_path == str(tmp_path / "scenarios.json")

    def test_no_config_path_leaves_scenario_unchanged(self) -> None:
        """When config_path is absent, scenario_file_path is not resolved."""
        si = SimulationInput.model_validate(
            self._base_data(scenario_file_path="./scenarios.json"),
            context={},
        )
        assert si.scenario_file_path == "./scenarios.json"

    def test_cli_override_skips_scenario_resolution(self, tmp_path: Path) -> None:
        """CLI-provided scenario_file_path is not resolved config-relatively."""
        si = SimulationInput.model_validate(
            self._base_data(scenario_file_path="./scenarios.json"),
            context=self._ctx(tmp_path, cli_overrides={"scenario_file_path"}),
        )
        assert si.scenario_file_path == "./scenarios.json"

    def test_output_resolves_independently_of_scenario(self, tmp_path: Path) -> None:
        """output_file_path resolves config-relatively even when scenario is None."""
        si = SimulationInput.model_validate(
            self._base_data(),
            context=self._ctx(tmp_path),
        )
        assert si.scenario_file_path is None
        assert si.output_file_path == str(tmp_path / "simulation.json")

    def test_cli_override_prevents_output_resolution(self, tmp_path: Path) -> None:
        """output_file_path stays as-is when set via CLI."""
        si = SimulationInput.model_validate(
            self._base_data(output_file_path="./my_output.json"),
            context=self._ctx(tmp_path, cli_overrides={"output_file_path"}),
        )
        assert si.output_file_path == "./my_output.json"

    def test_absolute_path_passes_through_unchanged(self, tmp_path: Path) -> None:
        """Absolute scenario_file_path is not modified."""
        abs_path = str(tmp_path / "abs" / "scenarios.json")
        si = SimulationInput.model_validate(
            self._base_data(scenario_file_path=abs_path),
            context=self._ctx(tmp_path),
        )
        assert si.scenario_file_path == abs_path
