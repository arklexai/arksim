# SPDX-License-Identifier: Apache-2.0
"""Tests for the arksim init command."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from arksim.cli import EXIT_CONFIG_ERROR, main


class TestInit:
    """Tests for arksim init subcommand."""

    def test_init_creates_config_scenarios_and_agent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """arksim init creates config.yaml, scenarios.json, and my_agent.py."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["arksim", "init"])
        main()

        config_path = tmp_path / "config.yaml"
        scenarios_path = tmp_path / "scenarios.json"
        agent_path = tmp_path / "my_agent.py"
        assert config_path.exists()
        assert scenarios_path.exists()
        assert agent_path.exists()

        # Config defaults to custom agent type
        config = yaml.safe_load(config_path.read_text())
        assert config["agent_config"]["agent_type"] == "custom"
        assert "module_path" in config["agent_config"]["custom_config"]

        # Scenarios is valid JSON with expected structure
        scenarios = json.loads(scenarios_path.read_text())
        assert scenarios["schema_version"] == "v1"
        assert len(scenarios["scenarios"]) == 4

        # Agent file contains BaseAgent subclass
        agent_content = agent_path.read_text()
        assert "class MyAgent(BaseAgent)" in agent_content
        assert "async def execute" in agent_content

    def test_init_chat_completions_type(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """arksim init --agent-type chat_completions creates HTTP config."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            sys, "argv", ["arksim", "init", "--agent-type", "chat_completions"]
        )
        main()

        config = yaml.safe_load((tmp_path / "config.yaml").read_text())
        assert config["agent_config"]["agent_type"] == "chat_completions"
        assert "endpoint" in config["agent_config"]["api_config"]

        # No my_agent.py for HTTP type
        assert not (tmp_path / "my_agent.py").exists()

    def test_init_a2a_type(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """arksim init --agent-type a2a creates A2A config."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["arksim", "init", "--agent-type", "a2a"])
        main()

        config = yaml.safe_load((tmp_path / "config.yaml").read_text())
        assert config["agent_config"]["agent_type"] == "a2a"
        assert "endpoint" in config["agent_config"]["api_config"]

        # No my_agent.py for A2A type
        assert not (tmp_path / "my_agent.py").exists()

    def test_init_does_not_overwrite_existing_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """arksim init refuses to overwrite existing config.yaml."""
        monkeypatch.chdir(tmp_path)
        existing = tmp_path / "config.yaml"
        existing.write_text("existing content")

        monkeypatch.setattr(sys, "argv", ["arksim", "init"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == EXIT_CONFIG_ERROR
        assert existing.read_text() == "existing content"

    def test_init_does_not_overwrite_existing_scenarios(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """arksim init refuses to overwrite existing scenarios.json."""
        monkeypatch.chdir(tmp_path)
        existing = tmp_path / "scenarios.json"
        existing.write_text("existing content")

        monkeypatch.setattr(sys, "argv", ["arksim", "init"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == EXIT_CONFIG_ERROR
        assert existing.read_text() == "existing content"

    def test_init_does_not_overwrite_existing_agent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """arksim init refuses to overwrite existing my_agent.py."""
        monkeypatch.chdir(tmp_path)
        existing = tmp_path / "my_agent.py"
        existing.write_text("existing content")

        monkeypatch.setattr(sys, "argv", ["arksim", "init"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == EXIT_CONFIG_ERROR
        assert existing.read_text() == "existing content"

    def test_init_does_not_overwrite_when_all_exist(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """arksim init refuses to overwrite when all files exist."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.yaml").write_text("c")
        (tmp_path / "scenarios.json").write_text("s")
        (tmp_path / "my_agent.py").write_text("a")

        monkeypatch.setattr(sys, "argv", ["arksim", "init"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == EXIT_CONFIG_ERROR

        assert (tmp_path / "config.yaml").read_text() == "c"
        assert (tmp_path / "scenarios.json").read_text() == "s"
        assert (tmp_path / "my_agent.py").read_text() == "a"

    def test_init_scenarios_have_distinct_ids(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Each starter scenario has a unique scenario_id."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["arksim", "init"])
        main()

        scenarios = json.loads((tmp_path / "scenarios.json").read_text())
        ids = [s["scenario_id"] for s in scenarios["scenarios"]]
        assert len(ids) == len(set(ids))

    def test_init_scenarios_include_negative_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """At least one starter scenario tests an out-of-scope request."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["arksim", "init"])
        main()

        scenarios = json.loads((tmp_path / "scenarios.json").read_text())
        ids = [s["scenario_id"] for s in scenarios["scenarios"]]
        assert "out_of_scope" in ids

    def test_init_scenarios_include_ambiguous_intent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Starter scenarios include an ambiguous intent scenario."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["arksim", "init"])
        main()

        scenarios = json.loads((tmp_path / "scenarios.json").read_text())
        ids = [s["scenario_id"] for s in scenarios["scenarios"]]
        assert "ambiguous_intent" in ids

    def test_init_scenarios_validate_against_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Scaffolded scenarios.json validates against the Scenarios Pydantic model."""
        from arksim.scenario.entities import Scenarios

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["arksim", "init"])
        main()

        data = json.loads((tmp_path / "scenarios.json").read_text())
        parsed = Scenarios.model_validate(data)
        assert len(parsed.scenarios) == 4

    def test_init_force_overwrites_existing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """arksim init --force overwrites existing files."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.yaml").write_text("old config")
        (tmp_path / "scenarios.json").write_text("old scenarios")
        (tmp_path / "my_agent.py").write_text("old agent")

        monkeypatch.setattr(sys, "argv", ["arksim", "init", "--force"])
        main()

        config = yaml.safe_load((tmp_path / "config.yaml").read_text())
        assert config["agent_config"]["agent_type"] == "custom"
        assert (tmp_path / "config.yaml").read_text() != "old config"
        assert (tmp_path / "scenarios.json").read_text() != "old scenarios"
        assert (tmp_path / "my_agent.py").read_text() != "old agent"

    def test_init_chat_completions_ignores_existing_agent_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """arksim init --agent-type chat_completions ignores existing my_agent.py."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "my_agent.py").write_text("existing agent")

        monkeypatch.setattr(
            sys,
            "argv",
            ["arksim", "init", "--agent-type", "chat_completions"],
        )
        main()

        # config.yaml created, my_agent.py untouched
        assert (tmp_path / "config.yaml").exists()
        assert (tmp_path / "my_agent.py").read_text() == "existing agent"
