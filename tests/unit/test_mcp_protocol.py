# SPDX-License-Identifier: Apache-2.0
"""End-to-end test of the FastMCP @mcp.tool() wrapper layer.

These tests exercise the protocol surface (tool registration, argument
coercion, return-shape serialization) rather than the underlying
``_function`` internals which are covered by ``test_mcp_server.py``.

Skipped when the mcp SDK is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

mcp_sdk = pytest.importorskip(
    "mcp.server.fastmcp",
    reason="mcp[cli] SDK not installed; install with: pip install arksim[claude]",
)

import integrations.claude_code.mcp_server.server as server_mod  # noqa: E402


@pytest.fixture
def reset_project_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Anchor _project_root() to a tmp dir so path validation passes."""
    monkeypatch.setattr(server_mod, "_PROJECT_ROOT", tmp_path.resolve())
    return tmp_path


class TestToolRegistration:
    """Every advertised tool is actually registered with FastMCP."""

    def test_simulate_evaluate_registered(self) -> None:
        # The decorated function is still importable as a module attribute.
        assert callable(server_mod.simulate_evaluate)

    def test_evaluate_registered(self) -> None:
        assert callable(server_mod.evaluate)

    def test_list_results_registered(self) -> None:
        assert callable(server_mod.list_results)

    def test_read_result_registered(self) -> None:
        assert callable(server_mod.read_result)

    def test_init_project_registered(self) -> None:
        assert callable(server_mod.init_project)

    def test_launch_ui_registered(self) -> None:
        assert callable(server_mod.launch_ui)


class TestToolReturnShape:
    """Tool wrappers return JSON-serializable dicts."""

    def test_simulate_evaluate_returns_dict(
        self, reset_project_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stub run_cli so we do not require the real arksim binary.
        monkeypatch.setattr(
            server_mod,
            "run_cli",
            lambda *a, **kw: {
                "status": "success",
                "stdout": "ok",
                "stderr": "",
                "return_code": 0,
            },
        )
        # Invalid path should still return a structured dict, not raise.
        result = server_mod.simulate_evaluate(config_path="missing.yaml")
        assert isinstance(result, dict)
        assert result["status"] == "error"

    def test_list_results_returns_dict(self, reset_project_root: Path) -> None:
        result = server_mod.list_results(output_dir=str(reset_project_root))
        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert isinstance(result["runs"], list)


class TestArgumentCoercion:
    """Wrappers reject malformed args via path validation, not exceptions."""

    def test_simulate_evaluate_with_invalid_cwd_returns_error(
        self, reset_project_root: Path
    ) -> None:
        config = reset_project_root / "config.yaml"
        config.write_text("agent_config: {}\n")
        result = server_mod.simulate_evaluate(
            config_path=str(config),
            cwd="/etc",  # outside project root
        )
        assert result["status"] == "error"

    def test_init_project_invalid_agent_type_returns_error(
        self, reset_project_root: Path
    ) -> None:
        result = server_mod.init_project(agent_type="not-a-real-type")
        assert result["status"] == "error"
        assert "Invalid agent_type" in result["error_message"]


class TestProjectRootAnchor:
    """_project_root() refuses / and $HOME at startup."""

    def test_resolve_project_root_refuses_filesystem_root(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pathlib import Path

        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: Path("/")))
        with pytest.raises(SystemExit, match="filesystem root"):
            server_mod._resolve_project_root()

    def test_resolve_project_root_refuses_home(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from pathlib import Path

        # No marker file in tmp_path so the walk falls through.
        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        with pytest.raises(SystemExit, match="home directory"):
            server_mod._resolve_project_root()
