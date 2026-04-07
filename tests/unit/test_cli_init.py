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

    def test_init_creates_config_and_scenarios(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """arksim init creates config.yaml and scenarios.json in cwd."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["arksim", "init"])
        main()

        config_path = tmp_path / "config.yaml"
        scenarios_path = tmp_path / "scenarios.json"
        assert config_path.exists()
        assert scenarios_path.exists()

        # Config is valid YAML with expected structure
        config = yaml.safe_load(config_path.read_text())
        assert config["agent_config"]["agent_type"] == "chat_completions"
        assert "endpoint" in config["agent_config"]["api_config"]

        # Scenarios is valid JSON with expected structure
        scenarios = json.loads(scenarios_path.read_text())
        assert scenarios["schema_version"] == "v1"
        assert len(scenarios["scenarios"]) == 3

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

        # Original content preserved
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

        # Original content preserved
        assert existing.read_text() == "existing content"

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
