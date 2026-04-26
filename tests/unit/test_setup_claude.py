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
    """Create a fake integration directory with per-skill SKILL.md files."""
    integration_dir = tmp_path / "integrations" / "claude_code"
    skills_dir = integration_dir / "skills"

    test_dir = skills_dir / "arksim-test"
    test_dir.mkdir(parents=True)
    (test_dir / "SKILL.md").write_text("# Test skill\nRun tests.")

    evaluate_dir = skills_dir / "arksim-evaluate"
    evaluate_dir.mkdir(parents=True)
    (evaluate_dir / "SKILL.md").write_text("# Evaluate skill\nRun eval.")

    return integration_dir


class TestSetupClaudeFreshProject:
    """setup-claude on a fresh project with no .claude/ directory."""

    def test_creates_mcp_json_with_server_config(self, tmp_path: Path) -> None:
        """Creates .mcp.json with mcpServers.arksim entry."""
        integration_dir = _make_integration_dir(tmp_path)
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        with (
            patch("arksim.cli._find_integration_dir", return_value=integration_dir),
            patch("shutil.which", return_value="/usr/local/bin/arksim-mcp"),
        ):
            _run_setup_claude(project_dir=str(project_dir))

        mcp_path = project_dir / ".mcp.json"
        assert mcp_path.exists()
        mcp_config = json.loads(mcp_path.read_text())
        assert "mcpServers" in mcp_config
        assert "arksim" in mcp_config["mcpServers"]
        server = mcp_config["mcpServers"]["arksim"]
        assert "command" in server
        assert "args" in server

    def test_copies_skills_to_claude_skills_directories(self, tmp_path: Path) -> None:
        """Copies arksim-*/SKILL.md directories to .claude/skills/."""
        integration_dir = _make_integration_dir(tmp_path)
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        with (
            patch("arksim.cli._find_integration_dir", return_value=integration_dir),
            patch("shutil.which", return_value="/usr/local/bin/arksim-mcp"),
        ):
            _run_setup_claude(project_dir=str(project_dir))

        skills_dir = project_dir / ".claude" / "skills"
        assert (skills_dir / "arksim-test").is_dir()
        assert (skills_dir / "arksim-evaluate").is_dir()
        assert (skills_dir / "arksim-test" / "SKILL.md").exists()
        assert (skills_dir / "arksim-evaluate" / "SKILL.md").exists()
        assert (
            skills_dir / "arksim-test" / "SKILL.md"
        ).read_text() == "# Test skill\nRun tests."


class TestSetupClaudeMerge:
    """setup-claude merges with existing .mcp.json."""

    def test_preserves_existing_mcp_servers(self, tmp_path: Path) -> None:
        """Adds arksim without removing other MCP server entries."""
        integration_dir = _make_integration_dir(tmp_path)
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        existing_mcp = {"mcpServers": {"other-tool": {"command": "other", "args": []}}}
        (project_dir / ".mcp.json").write_text(json.dumps(existing_mcp, indent=2))

        with (
            patch("arksim.cli._find_integration_dir", return_value=integration_dir),
            patch("shutil.which", return_value="/usr/local/bin/arksim-mcp"),
        ):
            _run_setup_claude(project_dir=str(project_dir))

        mcp_config = json.loads((project_dir / ".mcp.json").read_text())
        assert "other-tool" in mcp_config["mcpServers"]
        assert "arksim" in mcp_config["mcpServers"]


class TestSetupClaudeSkillConflict:
    """setup-claude exits when arksim-* skill directories already exist."""

    def test_exits_error_when_skills_exist_without_force(self, tmp_path: Path) -> None:
        """Exits with EXIT_CONFIG_ERROR when arksim-* dirs exist and --force not set."""
        integration_dir = _make_integration_dir(tmp_path)
        project_dir = tmp_path / "project"
        old_skill = project_dir / ".claude" / "skills" / "arksim-test"
        old_skill.mkdir(parents=True)
        (old_skill / "SKILL.md").write_text("old content")

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
        assert (old_skill / "SKILL.md").read_text() == "old content"

    def test_overwrites_skills_with_force(self, tmp_path: Path) -> None:
        """With --force, overwrites existing arksim-* skill directories."""
        integration_dir = _make_integration_dir(tmp_path)
        project_dir = tmp_path / "project"
        old_skill = project_dir / ".claude" / "skills" / "arksim-test"
        old_skill.mkdir(parents=True)
        (old_skill / "SKILL.md").write_text("old content")

        with (
            patch("arksim.cli._find_integration_dir", return_value=integration_dir),
            patch("shutil.which", return_value="/usr/local/bin/arksim-mcp"),
        ):
            _run_setup_claude(project_dir=str(project_dir), force=True)

        skills_dir = project_dir / ".claude" / "skills"
        # Old content replaced, new skills present
        assert (skills_dir / "arksim-test" / "SKILL.md").read_text() != "old content"
        assert (skills_dir / "arksim-test" / "SKILL.md").exists()
        assert (skills_dir / "arksim-evaluate" / "SKILL.md").exists()


class TestSetupClaudeUninstall:
    """setup-claude --uninstall removes arksim integration."""

    def test_removes_mcp_entry_and_skills(self, tmp_path: Path) -> None:
        """Uninstall removes arksim from .mcp.json and deletes arksim-* dirs."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        claude_dir = project_dir / ".claude"
        skills_dir = claude_dir / "skills"
        test_skill = skills_dir / "arksim-test"
        test_skill.mkdir(parents=True)
        (test_skill / "SKILL.md").write_text("skill content")
        eval_skill = skills_dir / "arksim-evaluate"
        eval_skill.mkdir(parents=True)
        (eval_skill / "SKILL.md").write_text("eval content")

        mcp_config = {
            "mcpServers": {
                "arksim": {"command": "arksim-mcp", "args": []},
                "other-tool": {"command": "other", "args": []},
            },
        }
        (project_dir / ".mcp.json").write_text(json.dumps(mcp_config, indent=2))

        _run_setup_claude(project_dir=str(project_dir), uninstall=True)

        updated = json.loads((project_dir / ".mcp.json").read_text())
        assert "arksim" not in updated.get("mcpServers", {})
        assert "other-tool" in updated["mcpServers"]
        assert not test_skill.exists()
        assert not eval_skill.exists()

    def test_uninstall_removes_mcp_json_when_empty(self, tmp_path: Path) -> None:
        """Uninstall removes .mcp.json entirely when no servers remain."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        mcp_config = {"mcpServers": {"arksim": {"command": "arksim-mcp", "args": []}}}
        (project_dir / ".mcp.json").write_text(json.dumps(mcp_config))

        _run_setup_claude(project_dir=str(project_dir), uninstall=True)

        assert not (project_dir / ".mcp.json").exists()

    def test_uninstall_noop_when_not_installed(self, tmp_path: Path) -> None:
        """Uninstall succeeds gracefully when arksim is not installed."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Should not raise (no .mcp.json, no skills)
        _run_setup_claude(project_dir=str(project_dir), uninstall=True)


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
        with (
            patch("arksim.cli._find_integration_dir", return_value=integration_dir),
            patch("shutil.which", return_value="/usr/local/bin/arksim-mcp"),
        ):
            main()

        assert (project_dir / ".mcp.json").exists()

    def test_setup_claude_uninstall_flag(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """arksim setup-claude --uninstall removes arksim from .mcp.json."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        claude_dir = project_dir / ".claude"
        test_skill = claude_dir / "skills" / "arksim-test"
        test_skill.mkdir(parents=True)
        (test_skill / "SKILL.md").write_text("# test")

        mcp_config = {"mcpServers": {"arksim": {"command": "arksim-mcp", "args": []}}}
        (project_dir / ".mcp.json").write_text(json.dumps(mcp_config))

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

        # .mcp.json should be removed (no servers remain)
        assert not (project_dir / ".mcp.json").exists()
        assert not test_skill.exists()


class TestSetupClaudeCorruptedMcpJson:
    """setup-claude handles corrupted .mcp.json gracefully."""

    def test_install_exits_on_invalid_json(self, tmp_path: Path) -> None:
        """Invalid JSON in .mcp.json during install causes clean exit."""
        integration_dir = _make_integration_dir(tmp_path)
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / ".mcp.json").write_text("{not valid json")

        with (
            patch(
                "arksim.cli._find_integration_dir",
                return_value=integration_dir,
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            _run_setup_claude(project_dir=str(project_dir))

        assert exc_info.value.code == EXIT_CONFIG_ERROR

    def test_uninstall_exits_on_invalid_json(self, tmp_path: Path) -> None:
        """Invalid JSON in .mcp.json during uninstall causes clean exit."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / ".mcp.json").write_text("{not valid json")

        with pytest.raises(SystemExit) as exc_info:
            _run_setup_claude(project_dir=str(project_dir), uninstall=True)

        assert exc_info.value.code == EXIT_CONFIG_ERROR


class TestBuildMcpServerConfig:
    """_build_mcp_server_config exits when arksim-mcp is not on PATH."""

    def test_exits_when_arksim_mcp_not_found(self, tmp_path: Path) -> None:
        integration_dir = _make_integration_dir(tmp_path)
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        with (
            patch(
                "arksim.cli._find_integration_dir",
                return_value=integration_dir,
            ),
            patch("shutil.which", return_value=None),
            pytest.raises(SystemExit) as exc_info,
        ):
            _run_setup_claude(project_dir=str(project_dir))

        assert exc_info.value.code == EXIT_CONFIG_ERROR


class TestFindIntegrationDirFailure:
    """_find_integration_dir exits cleanly when integration is not found."""

    def test_exits_when_integration_not_found(self, tmp_path: Path) -> None:
        """Missing integration dir causes SystemExit with EXIT_CONFIG_ERROR."""
        from arksim.cli import _find_integration_dir

        # Point __file__ at a directory where integrations/ does not exist
        fake_cli = tmp_path / "arksim" / "cli.py"
        fake_cli.parent.mkdir(parents=True)
        fake_cli.touch()

        with (
            patch("arksim.cli.__file__", str(fake_cli)),
            patch("arksim.cli.resources") as mock_resources,
            pytest.raises(SystemExit) as exc_info,
        ):
            mock_resources.files.side_effect = ModuleNotFoundError("no module")
            _find_integration_dir()

        assert exc_info.value.code == EXIT_CONFIG_ERROR
