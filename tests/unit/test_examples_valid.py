# SPDX-License-Identifier: Apache-2.0
"""Validate that all example config and scenario files parse correctly.

These tests do not import SDK-specific code or run simulations. They verify
that the YAML configs produce valid AgentConfig objects and the scenario
JSON files conform to the Scenarios schema, catching structural errors
(missing fields, typos, bad types) before users hit them at runtime.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from arksim.config import AgentConfig
from arksim.scenario import Scenarios

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"

# Discover all example directories that contain a config*.yaml
_EXAMPLE_CONFIGS: list[Path] = sorted(EXAMPLES_DIR.glob("*/config*.yaml"))


@pytest.fixture(params=_EXAMPLE_CONFIGS, ids=lambda p: f"{p.parent.name}/{p.name}")
def example_config_path(request: pytest.FixtureRequest) -> Path:
    return request.param


class TestExampleConfigs:
    """Every example config.yaml must produce a valid AgentConfig."""

    def test_agent_config_parses(self, example_config_path: Path) -> None:
        with open(example_config_path) as f:
            data = yaml.safe_load(f)

        agent_data = data.get("agent_config")
        if agent_data is None:
            pytest.skip("No agent_config in this config file")

        config = AgentConfig.model_validate(agent_data)
        assert config.agent_name
        assert config.agent_type


# Discover all scenario JSON files
_SCENARIO_FILES: list[Path] = sorted(EXAMPLES_DIR.glob("*/scenarios.json"))


@pytest.fixture(params=_SCENARIO_FILES, ids=lambda p: p.parent.name)
def scenario_path(request: pytest.FixtureRequest) -> Path:
    return request.param


class TestExampleScenarios:
    """Every example scenarios.json must conform to the Scenarios schema."""

    def test_scenarios_parse(self, scenario_path: Path) -> None:
        scenarios = Scenarios.load(str(scenario_path))
        assert len(scenarios.scenarios) > 0

    def test_scenarios_have_required_fields(self, scenario_path: Path) -> None:
        scenarios = Scenarios.load(str(scenario_path))
        for scenario in scenarios.scenarios:
            assert scenario.scenario_id
            assert scenario.goal
            assert scenario.user_profile
