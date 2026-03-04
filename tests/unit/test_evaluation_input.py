# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.evaluator.entities.EvaluationInput validator."""

from pathlib import Path
from typing import Any

from arksim.evaluator.entities import EvaluationInput


class TestEvaluationInputPathResolution:
    """Tests for config-relative path resolution in EvaluationInput."""

    def _ctx(self, tmp_path: Path, **kwargs: Any) -> dict:  # noqa: ANN401
        return {"config_path": str(tmp_path / "config.yaml"), **kwargs}

    def test_scenario_resolves_to_config_relative(self, tmp_path: Path) -> None:
        """scenario_file_path is always resolved relative to config dir."""
        ei = EvaluationInput.model_validate(
            {"scenario_file_path": "./scenarios.json"},
            context=self._ctx(tmp_path),
        )
        assert ei.scenario_file_path == str(tmp_path / "scenarios.json")

    def test_simulation_file_resolves_to_config_relative(self, tmp_path: Path) -> None:
        """simulation_file_path is always resolved relative to config dir."""
        ei = EvaluationInput.model_validate(
            {"simulation_file_path": "./simulation.json"},
            context=self._ctx(tmp_path),
        )
        assert ei.simulation_file_path == str(tmp_path / "simulation.json")

    def test_output_dir_resolves_independently(self, tmp_path: Path) -> None:
        """output_dir resolves config-relatively regardless of other paths."""
        ei = EvaluationInput.model_validate(
            {"output_dir": "./evaluation"},
            context=self._ctx(tmp_path),
        )
        assert ei.output_dir == str(tmp_path / "evaluation")

    def test_no_config_path_leaves_all_unchanged(self) -> None:
        """When config_path is absent, no paths are resolved."""
        ei = EvaluationInput.model_validate(
            {
                "scenario_file_path": "./scenarios.json",
                "simulation_file_path": "./simulation.json",
                "output_dir": "./evaluation",
            },
            context={},
        )
        assert ei.scenario_file_path == "./scenarios.json"
        assert ei.simulation_file_path == "./simulation.json"
        assert ei.output_dir == "./evaluation"

    def test_cli_override_skips_resolution_for_that_path(self, tmp_path: Path) -> None:
        """CLI-overridden path stays as-is; non-overridden path still resolves."""
        ei = EvaluationInput.model_validate(
            {
                "scenario_file_path": "./scenarios.json",
                "simulation_file_path": "./simulation.json",
            },
            context=self._ctx(tmp_path, cli_overrides={"simulation_file_path"}),
        )
        assert ei.scenario_file_path == str(tmp_path / "scenarios.json")
        assert ei.simulation_file_path == "./simulation.json"

    def test_cli_override_prevents_output_dir_resolution(self, tmp_path: Path) -> None:
        """output_dir stays as-is when set via CLI."""
        ei = EvaluationInput.model_validate(
            {"output_dir": "./my_eval"},
            context=self._ctx(tmp_path, cli_overrides={"output_dir"}),
        )
        assert ei.output_dir == "./my_eval"

    def test_absolute_path_passes_through_unchanged(self, tmp_path: Path) -> None:
        """Absolute output_dir is not modified."""
        abs_dir = str(tmp_path / "abs" / "evaluation")
        ei = EvaluationInput.model_validate(
            {"output_dir": abs_dir},
            context=self._ctx(tmp_path),
        )
        assert ei.output_dir == abs_dir
