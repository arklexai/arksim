# SPDX-License-Identifier: Apache-2.0
"""Tests for the arksim setup-claude command."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from arksim.cli import EXIT_CONFIG_ERROR, _run_setup_claude, main


def _make_integration_dir(tmp_path: Path) -> Path:
    """Create a fake integration directory with test skill files."""
    integration_dir = tmp_path / "integrations" / "claude_code"
    skills_dir = integration_dir / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "test.md").write_text("# Test skill\nRun tests.")
    (skills_dir / "evaluate.md").write_text("# Evaluate skill\nRun eval.")
    return integration_dir


class TestSetupClaudeFreshProject:
    """setup-claude on a fresh project with no .claude/ directory."""

    def test_creates_settings_json_with_mcp_config(self, tmp_path: Path) -> None:
        """Creates .claude/settings.json with mcpServers.arksim entry."""
        integration_dir = _make_integration_dir(tmp_path)
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        with patch("arksim.cli._find_integration_dir", return_value=integration_dir):
            _run_setup_claude(project_dir=str(project_dir))

        settings_path = project_dir / ".claude" / "settings.json"
        assert settings_path.exists()
        settings = json.loads(settings_path.read_text())
        assert "mcpServers" in settings
        assert "arksim" in settings["mcpServers"]
        server = settings["mcpServers"]["arksim"]
        assert server["command"] == "python"
        assert server["args"] == [
            "-m",
            "integrations.claude_code.mcp_server.server",
        ]

    def test_copies_skills_to_claude_skills_arksim(self, tmp_path: Path) -> None:
        """Copies skills/*.md files to .claude/skills/arksim/."""
        integration_dir = _make_integration_dir(tmp_path)
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        with patch("arksim.cli._find_integration_dir", return_value=integration_dir):
            _run_setup_claude(project_dir=str(project_dir))

        skills_dir = project_dir / ".claude" / "skills" / "arksim"
        assert skills_dir.is_dir()
        assert (skills_dir / "test.md").exists()
        assert (skills_dir / "evaluate.md").exists()
        assert (skills_dir / "test.md").read_text() == "# Test skill\nRun tests."


class TestSetupClaudeMerge:
    """setup-claude merges with existing .claude/settings.json."""

    def test_preserves_existing_hooks(self, tmp_path: Path) -> None:
        """Merges mcpServers without overwriting existing hooks key."""
        integration_dir = _make_integration_dir(tmp_path)
        project_dir = tmp_path / "project"
        claude_dir = project_dir / ".claude"
        claude_dir.mkdir(parents=True)

        existing_settings = {
            "hooks": {
                "SessionStart": [
                    {
                        "matcher": "",
                        "hooks": [{"type": "command", "command": "echo hi"}],
                    }
                ]
            }
        }
        (claude_dir / "settings.json").write_text(
            json.dumps(existing_settings, indent=2)
        )

        with patch("arksim.cli._find_integration_dir", return_value=integration_dir):
            _run_setup_claude(project_dir=str(project_dir))

        settings = json.loads((claude_dir / "settings.json").read_text())
        assert settings["hooks"] == existing_settings["hooks"]
        assert "arksim" in settings["mcpServers"]

    def test_preserves_existing_mcp_servers(self, tmp_path: Path) -> None:
        """Adds arksim without removing other MCP server entries."""
        integration_dir = _make_integration_dir(tmp_path)
        project_dir = tmp_path / "project"
        claude_dir = project_dir / ".claude"
        claude_dir.mkdir(parents=True)

        existing_settings = {
            "mcpServers": {"other-tool": {"command": "other", "args": []}}
        }
        (claude_dir / "settings.json").write_text(
            json.dumps(existing_settings, indent=2)
        )

        with patch("arksim.cli._find_integration_dir", return_value=integration_dir):
            _run_setup_claude(project_dir=str(project_dir))

        settings = json.loads((claude_dir / "settings.json").read_text())
        assert "other-tool" in settings["mcpServers"]
        assert "arksim" in settings["mcpServers"]


class TestSetupClaudeSkillConflict:
    """setup-claude exits when skills directory already exists."""

    def test_exits_error_when_skills_exist_without_force(self, tmp_path: Path) -> None:
        """Exits with EXIT_CONFIG_ERROR when skills already exist and --force not set."""
        integration_dir = _make_integration_dir(tmp_path)
        project_dir = tmp_path / "project"
        skills_dir = project_dir / ".claude" / "skills" / "arksim"
        skills_dir.mkdir(parents=True)
        (skills_dir / "old.md").write_text("old content")

        with (
            patch(
                "arksim.cli._find_integration_dir",
                return_value=integration_dir,
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            _run_setup_claude(project_dir=str(project_dir))

        assert exc_info.value.code == EXIT_CONFIG_ERROR
        # Old file should still be there (no overwrite)
        assert (skills_dir / "old.md").read_text() == "old content"

    def test_overwrites_skills_with_force(self, tmp_path: Path) -> None:
        """With --force, overwrites existing skills directory."""
        integration_dir = _make_integration_dir(tmp_path)
        project_dir = tmp_path / "project"
        skills_dir = project_dir / ".claude" / "skills" / "arksim"
        skills_dir.mkdir(parents=True)
        (skills_dir / "old.md").write_text("old content")

        with patch("arksim.cli._find_integration_dir", return_value=integration_dir):
            _run_setup_claude(project_dir=str(project_dir), force=True)

        # Old file removed, new files present
        assert not (skills_dir / "old.md").exists()
        assert (skills_dir / "test.md").exists()
        assert (skills_dir / "evaluate.md").exists()


class TestSetupClaudeUninstall:
    """setup-claude --uninstall removes arksim integration."""

    def test_removes_mcp_entry_and_skills(self, tmp_path: Path) -> None:
        """Uninstall removes arksim from mcpServers and deletes skills dir."""
        project_dir = tmp_path / "project"
        claude_dir = project_dir / ".claude"
        claude_dir.mkdir(parents=True)
        skills_dir = claude_dir / "skills" / "arksim"
        skills_dir.mkdir(parents=True)
        (skills_dir / "test.md").write_text("skill content")

        settings = {
            "hooks": {"SessionStart": []},
            "mcpServers": {
                "arksim": {
                    "command": "python",
                    "args": ["-m", "integrations.claude_code.mcp_server.server"],
                },
                "other-tool": {"command": "other", "args": []},
            },
        }
        (claude_dir / "settings.json").write_text(json.dumps(settings, indent=2))

        _run_setup_claude(project_dir=str(project_dir), uninstall=True)

        updated = json.loads((claude_dir / "settings.json").read_text())
        assert "arksim" not in updated.get("mcpServers", {})
        assert "other-tool" in updated["mcpServers"]
        assert updated["hooks"] == {"SessionStart": []}
        assert not skills_dir.exists()

    def test_uninstall_noop_when_not_installed(self, tmp_path: Path) -> None:
        """Uninstall succeeds gracefully when arksim is not installed."""
        project_dir = tmp_path / "project"
        claude_dir = project_dir / ".claude"
        claude_dir.mkdir(parents=True)

        settings = {"hooks": {"SessionStart": []}}
        (claude_dir / "settings.json").write_text(json.dumps(settings, indent=2))

        # Should not raise
        _run_setup_claude(project_dir=str(project_dir), uninstall=True)

        updated = json.loads((claude_dir / "settings.json").read_text())
        assert updated == {"hooks": {"SessionStart": []}}


class TestSetupClaudeCLIIntegration:
    """Tests that setup-claude is wired into the argparse CLI."""

    def test_setup_claude_subcommand_recognized(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """arksim setup-claude runs without argparse errors."""
        integration_dir = _make_integration_dir(tmp_path)
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        monkeypatch.setattr(
            sys,
            "argv",
            ["arksim", "setup-claude", "--project-dir", str(project_dir)],
        )
        with patch("arksim.cli._find_integration_dir", return_value=integration_dir):
            main()

        assert (project_dir / ".claude" / "settings.json").exists()

    def test_setup_claude_uninstall_flag(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """arksim setup-claude --uninstall passes uninstall=True."""
        project_dir = tmp_path / "project"
        claude_dir = project_dir / ".claude"
        claude_dir.mkdir(parents=True)
        (claude_dir / "settings.json").write_text("{}")

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "arksim",
                "setup-claude",
                "--uninstall",
                "--project-dir",
                str(project_dir),
            ],
        )
        main()
