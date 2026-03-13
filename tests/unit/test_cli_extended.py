# SPDX-License-Identifier: Apache-2.0
"""Tests for CLI interface: argument parsing, overrides validation, examples, and prompts."""

from __future__ import annotations

import io
import os
import sys
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from arksim.cli import (
    EXIT_CONFIG_ERROR,
    _log_config_summary,
    _run_examples,
    _run_show_prompts,
    build_parser,
    validate_overrides,
)

# ── validate_overrides ──────────────────────────────────


class TestValidateOverrides:
    """Tests for validate_overrides."""

    def test_valid_keys_pass(self) -> None:
        """No exit when all keys are valid."""
        validate_overrides({"model": "gpt-4"}, {"model", "provider"})

    def test_invalid_keys_exit(self) -> None:
        """Exits with EXIT_CONFIG_ERROR when unknown keys are present."""
        with pytest.raises(SystemExit) as exc_info:
            validate_overrides({"bad_key": "val"}, {"model"})
        assert exc_info.value.code == EXIT_CONFIG_ERROR

    def test_empty_overrides_pass(self) -> None:
        """Empty overrides always pass."""
        validate_overrides({}, {"model"})


# ── _log_config_summary ─────────────────────────────────


class TestLogConfigSummary:
    def test_runs_without_error(self) -> None:
        _log_config_summary("Test", {"a": 1, "b": "hello"})


# ── _run_show_prompts ────────────────────────────────────


class TestRunShowPrompts:
    def test_valid_category(self, capsys: pytest.CaptureFixture[str]) -> None:
        _run_show_prompts("helpfulness")
        captured = capsys.readouterr()
        assert "helpfulness" in captured.out.lower()

    def test_all_categories(self, capsys: pytest.CaptureFixture[str]) -> None:
        _run_show_prompts(None)
        captured = capsys.readouterr()
        assert "helpfulness" in captured.out.lower()
        assert "coherence" in captured.out.lower()

    def test_invalid_category_exits(self) -> None:
        with pytest.raises(SystemExit):
            _run_show_prompts("nonexistent_category")


# ── build_parser ─────────────────────────────────────────


class TestBuildParser:
    """Tests for build_parser subcommand structure."""

    def test_all_subcommands_registered(self) -> None:
        """Parser has all expected subcommands."""
        parser = build_parser()
        for cmd in [
            "simulate",
            "evaluate",
            "simulate-evaluate",
            "show-prompts",
            "examples",
            "ui",
        ]:
            args = parser.parse_args([cmd])
            assert args.command == cmd

    def test_simulate_with_config_and_overrides(self) -> None:
        """Simulate parses config_file and REMAINDER overrides."""
        parser = build_parser()
        args = parser.parse_args(["simulate", "config.yaml", "--max-turns", "3"])
        assert args.command == "simulate"
        assert args.config_file == "config.yaml"
        assert args.additional_args == ["--max-turns", "3"]

    def test_simulate_without_config(self) -> None:
        """Simulate with no config sets config_file to None."""
        parser = build_parser()
        args = parser.parse_args(["simulate"])
        assert args.config_file is None

    def test_examples_list_flag(self) -> None:
        """Examples --list sets list_only to True."""
        parser = build_parser()
        args = parser.parse_args(["examples", "--list"])
        assert args.command == "examples"
        assert args.list_only is True
        assert args.name is None

    def test_examples_with_name(self) -> None:
        """Examples with a name sets args.name."""
        parser = build_parser()
        args = parser.parse_args(["examples", "bank-insurance"])
        assert args.name == "bank-insurance"
        assert args.list_only is False

    def test_show_prompts_category(self) -> None:
        """Show-prompts --category sets args.category."""
        parser = build_parser()
        args = parser.parse_args(["show-prompts", "--category", "faithfulness"])
        assert args.category == "faithfulness"

    def test_show_prompts_default_category(self) -> None:
        """Show-prompts without --category defaults to None."""
        parser = build_parser()
        args = parser.parse_args(["show-prompts"])
        assert args.category is None

    def test_ui_custom_port(self) -> None:
        """UI --port sets custom port."""
        parser = build_parser()
        args = parser.parse_args(["ui", "--port", "9090"])
        assert args.port == 9090

    def test_ui_default_port(self) -> None:
        """UI without --port defaults to 8080."""
        parser = build_parser()
        args = parser.parse_args(["ui"])
        assert args.port == 8080


# ── _run_examples ────────────────────────────────────────


def _make_tarball(members: dict[str, str]) -> bytes:
    """Build an in-memory .tar.gz with given name→content entries."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in members.items():
            data = content.encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class TestRunExamples:
    """Tests for _run_examples."""

    @pytest.fixture()
    def fake_tarball(self) -> bytes:
        """A minimal tarball mimicking the GitHub archive."""
        return _make_tarball(
            {
                "arksim-main/examples/demo/config.yaml": "model: gpt-4",
                "arksim-main/examples/demo/scenario.json": "[]",
                "arksim-main/examples/bank-insurance/config.yaml": "model: gpt-4",
                "arksim-main/README.md": "# arksim",
            }
        )

    def test_list_only(
        self, fake_tarball: bytes, capsys: pytest.CaptureFixture
    ) -> None:
        """--list prints available example names."""
        resp = MagicMock()
        resp.read.return_value = fake_tarball
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=resp):
            _run_examples(list_only=True)

        out = capsys.readouterr().out
        assert "demo" in out
        assert "bank-insurance" in out

    def test_unknown_name_exits(self, fake_tarball: bytes) -> None:
        """Unknown example name exits with EXIT_CONFIG_ERROR."""
        resp = MagicMock()
        resp.read.return_value = fake_tarball
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=resp):
            with pytest.raises(SystemExit) as exc_info:
                _run_examples(name="nonexistent")
            assert exc_info.value.code == EXIT_CONFIG_ERROR

    def test_dest_exists_exits(self, fake_tarball: bytes, tmp_path: Path) -> None:
        """Exits with EXIT_CONFIG_ERROR when destination already exists."""
        resp = MagicMock()
        resp.read.return_value = fake_tarball
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)

        with (
            patch("urllib.request.urlopen", return_value=resp),
            patch("os.path.exists", return_value=True),
            pytest.raises(SystemExit) as exc_info,
        ):
            _run_examples()
        assert exc_info.value.code == EXIT_CONFIG_ERROR

    def test_path_traversal_skipped(self, tmp_path: Path) -> None:
        """Members with '..' or absolute paths are skipped."""
        tarball = _make_tarball(
            {
                "arksim-main/examples/demo/safe.txt": "ok",
                "arksim-main/examples/demo/../../etc/passwd": "evil",
                "arksim-main/examples/demo//tmp/evil": "evil",
            }
        )
        resp = MagicMock()
        resp.read.return_value = tarball
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)

        with (
            patch("urllib.request.urlopen", return_value=resp),
            patch("os.path.exists", return_value=False),
            patch("tarfile.TarFile.extract") as mock_extract,
        ):
            _run_examples(name="demo")

        extracted_names = [call.args[0].name for call in mock_extract.call_args_list]
        for name in extracted_names:
            assert ".." not in name.split("/")
            assert not os.path.isabs(name)

        for call in mock_extract.call_args_list:
            if sys.version_info >= (3, 12):
                assert call.kwargs.get("filter") == "data"
            else:
                assert "filter" not in call.kwargs
