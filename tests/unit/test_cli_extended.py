# SPDX-License-Identifier: Apache-2.0
"""Tests for additional CLI functions: validate_overrides, _log_config_summary, _run_show_prompts."""

from __future__ import annotations

import pytest

from arksim.cli import _log_config_summary, _run_show_prompts, validate_overrides


class TestValidateOverrides:
    def test_valid_keys_pass(self) -> None:
        validate_overrides({"model": "gpt-4"}, {"model", "provider"})

    def test_invalid_keys_exit(self) -> None:
        with pytest.raises(SystemExit):
            validate_overrides({"bad_key": "v"}, {"model"})

    def test_empty_overrides_pass(self) -> None:
        validate_overrides({}, {"model"})


class TestLogConfigSummary:
    def test_runs_without_error(self) -> None:
        _log_config_summary("Test", {"a": 1, "b": "hello"})


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
