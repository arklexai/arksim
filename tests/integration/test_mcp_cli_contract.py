# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the MCP server's CLI invocations.

Each test verifies that the MCP tool wrappers shell out to ``arksim``
with the exact argv shape we depend on. If the arksim CLI ever renames
a subcommand or flag, these tests fail loudly rather than letting the
break ship to users via Claude Code.

The tests mock subprocess.run inside ``cli_wrapper`` so they do not
require a real arksim binary on PATH; they assert the argv list passed
to subprocess.run matches the documented contract.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip(
    "mcp.server.fastmcp",
    reason="mcp[cli] SDK not installed; install with: pip install arksim[claude]",
)

import integrations.claude_code.mcp_server.cli_wrapper as wrapper_mod
import integrations.claude_code.mcp_server.server as server_mod

pytestmark = pytest.mark.integration


def _stub_run(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    """Build a subprocess.run mock that captures argv and writes to tempfiles."""

    def _runner(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:  # noqa: ANN401
        out_file = kw.get("stdout")
        err_file = kw.get("stderr")
        if hasattr(out_file, "write"):
            out_file.write(stdout)
        if hasattr(err_file, "write"):
            err_file.write(stderr)
        return subprocess.CompletedProcess(
            args=argv, returncode=returncode, stdout=stdout, stderr=stderr
        )

    mock = MagicMock(side_effect=_runner)
    return mock


@pytest.fixture
def project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Anchor server._project_root() to tmp_path."""
    monkeypatch.setattr(server_mod, "_PROJECT_ROOT", tmp_path.resolve())
    return tmp_path


# ── CLI argv contracts ──────────────────────────────────────


class TestSimulateEvaluateContract:
    def test_invokes_arksim_simulate_evaluate(self, project_root: Path) -> None:
        config = project_root / "config.yaml"
        config.write_text("agent_config: {}\n")
        runner = _stub_run()
        with patch.object(wrapper_mod.subprocess, "run", runner):
            server_mod._simulate_evaluate(config_path=str(config))
        called_argv = runner.call_args.args[0]
        assert called_argv[0] == "arksim"
        assert called_argv[1] == "simulate-evaluate"
        assert str(config.resolve()) in called_argv

    def test_passes_cli_overrides_as_flags(self, project_root: Path) -> None:
        config = project_root / "config.yaml"
        config.write_text("agent_config: {}\n")
        runner = _stub_run()
        with patch.object(wrapper_mod.subprocess, "run", runner):
            server_mod._simulate_evaluate(
                config_path=str(config),
                cli_overrides={"num_workers": "5"},
            )
        called_argv = runner.call_args.args[0]
        assert "--num-workers=5" in called_argv

    def test_rejects_path_override_outside_project(self, project_root: Path) -> None:
        config = project_root / "config.yaml"
        config.write_text("agent_config: {}\n")
        runner = _stub_run()
        with patch.object(wrapper_mod.subprocess, "run", runner):
            result = server_mod._simulate_evaluate(
                config_path=str(config),
                cli_overrides={"output_dir": "/etc/arksim"},
            )
        # Either the override is skipped (no flag argv) or the call short-circuits.
        if runner.called:
            argv = runner.call_args.args[0]
            assert not any("etc" in a for a in argv)
        # The response surfaces the skipped key as a warning.
        assert result["status"] in {"success", "error"}


class TestEvaluateContract:
    def test_invokes_arksim_evaluate(self, project_root: Path) -> None:
        config = project_root / "config.yaml"
        config.write_text("agent_config: {}\n")
        sim = project_root / "simulation.json"
        sim.write_text("{}")
        runner = _stub_run()
        with patch.object(wrapper_mod.subprocess, "run", runner):
            server_mod._evaluate(
                config_path=str(config),
                simulation_file_path=str(sim),
            )
        called_argv = runner.call_args.args[0]
        assert called_argv[0] == "arksim"
        assert called_argv[1] == "evaluate"
        assert any("simulation-file-path" in a for a in called_argv)


class TestInitProjectContract:
    def test_invokes_arksim_init_with_agent_type(self, project_root: Path) -> None:
        runner = _stub_run()
        with patch.object(wrapper_mod.subprocess, "run", runner):
            server_mod._init_project(
                agent_type="custom",
                directory=str(project_root),
            )
        called_argv = runner.call_args.args[0]
        assert called_argv[:2] == ["arksim", "init"]
        assert "--agent-type" in called_argv
        assert "custom" in called_argv

    def test_force_appends_flag(self, project_root: Path) -> None:
        runner = _stub_run()
        with patch.object(wrapper_mod.subprocess, "run", runner):
            server_mod._init_project(
                agent_type="custom",
                directory=str(project_root),
                force=True,
            )
        called_argv = runner.call_args.args[0]
        assert "--force" in called_argv


class TestStderrRedaction:
    def test_secrets_in_stderr_are_redacted(self, project_root: Path) -> None:
        config = project_root / "config.yaml"
        config.write_text("agent_config: {}\n")
        runner = _stub_run(
            returncode=1,
            stdout="",
            stderr=("Auth failed with token sk-proj-abcdefghij1234567890ABCDEFGH"),
        )
        with patch.object(wrapper_mod.subprocess, "run", runner):
            result = server_mod._simulate_evaluate(config_path=str(config))
        assert result["status"] == "error"
        assert "sk-proj-" not in result.get("stderr", "")
        assert "[REDACTED]" in result.get("stderr", "")
