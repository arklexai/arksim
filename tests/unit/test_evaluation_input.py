# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.evaluator.entities.EvaluationInput validator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from arksim.evaluator.entities import EvaluationInput


class TestEvaluationInputFields:
    """Tests for newly added optional LLM transport fields."""

    def test_evaluation_input_accepts_base_url_and_api_key(self) -> None:
        settings = EvaluationInput(
            model="llama3.1",
            provider="responses",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
        assert settings.base_url == "http://localhost:11434/v1"
        assert settings.api_key == "ollama"

    def test_evaluation_input_defaults_base_url_and_api_key_to_none(self) -> None:
        settings = EvaluationInput()
        assert settings.base_url is None
        assert settings.api_key is None


class TestEvaluationInputEvaluatorOverrides:
    """Tests for the evaluator_* LLM override fields."""

    def test_evaluator_overrides_default_to_none(self) -> None:
        settings = EvaluationInput()
        assert settings.evaluator_model is None
        assert settings.evaluator_provider is None
        assert settings.evaluator_base_url is None
        assert settings.evaluator_api_key is None

    def test_evaluator_overrides_accept_values(self) -> None:
        settings = EvaluationInput(
            model="llama3.1",
            provider="responses",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            evaluator_model="gpt-4o-mini",
            evaluator_provider="openai",
            evaluator_base_url="https://api.openai.com/v1",
            evaluator_api_key="sk-test",
        )
        assert settings.evaluator_model == "gpt-4o-mini"
        assert settings.evaluator_provider == "openai"
        assert settings.evaluator_base_url == "https://api.openai.com/v1"
        assert settings.evaluator_api_key == "sk-test"

    def test_evaluator_overrides_can_be_partial(self) -> None:
        """A user can override just provider/base_url and keep shared
        model/api_key, or any other partial pattern.
        """
        settings = EvaluationInput(
            provider="responses",
            base_url="http://localhost:11434/v1",
            evaluator_provider="openai",
        )
        assert settings.provider == "responses"
        assert settings.evaluator_provider == "openai"
        assert settings.evaluator_base_url is None
        assert settings.evaluator_api_key is None


class TestEvaluationInputPathResolution:
    """Tests for config-relative path resolution in EvaluationInput."""

    def _ctx(self, tmp_path: Path, **kwargs: Any) -> dict:  # noqa: ANN401
        return {"config_path": str(tmp_path / "config.yaml"), **kwargs}

    def test_scenario_resolves_to_config_relative(self, tmp_path: Path) -> None:
        """scenario_file_path is resolved relative to config dir."""
        ei = EvaluationInput.model_validate(
            {"scenario_file_path": "./scenarios.json"},
            context=self._ctx(tmp_path),
        )
        assert ei.scenario_file_path == str(tmp_path / "scenarios.json")

    def test_simulation_file_resolves_to_config_relative(self, tmp_path: Path) -> None:
        """simulation_file_path is resolved relative to config dir."""
        ei = EvaluationInput.model_validate(
            {"simulation_file_path": "./simulation.json"},
            context=self._ctx(tmp_path),
        )
        assert ei.simulation_file_path == str(tmp_path / "simulation.json")

    def test_output_dir_resolves_to_config_relative(self, tmp_path: Path) -> None:
        """output_dir resolves config-relatively (including defaults)."""
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

    def test_custom_metrics_file_paths_resolve(self, tmp_path: Path) -> None:
        """custom_metrics_file_paths are resolved config-relatively."""
        ei = EvaluationInput.model_validate(
            {"custom_metrics_file_paths": ["./m1.py", "./m2.py"]},
            context=self._ctx(tmp_path),
        )
        assert ei.custom_metrics_file_paths == [
            str(tmp_path / "m1.py"),
            str(tmp_path / "m2.py"),
        ]

    def test_default_custom_metrics_not_touched(self, tmp_path: Path) -> None:
        """Default empty custom_metrics_file_paths is not touched."""
        ei = EvaluationInput.model_validate(
            {},
            context=self._ctx(tmp_path),
        )
        assert ei.custom_metrics_file_paths == []
