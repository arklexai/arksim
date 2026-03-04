# SPDX-License-Identifier: Apache-2.0
"""Tests for UI API path validation functions."""

import os

import pytest

from arksim.ui.api.routes_filesystem import _resolve_path, _validate_write_path
from arksim.ui.api.routes_results import _validate_results_dir


@pytest.fixture()
def fake_root(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> str:
    root = str(tmp_path)
    monkeypatch.setattr("arksim.ui.api.routes_filesystem.PROJECT_ROOT", root)
    monkeypatch.setattr("arksim.ui.api.routes_results.PROJECT_ROOT", root)
    return root


# -- _resolve_path -----------------------------------------------------------


class TestResolvePath:
    def test_empty_string_returns_empty(self, fake_root: str) -> None:
        assert _resolve_path("") == ""

    def test_relative_path_resolves_against_project_root(self, fake_root: str) -> None:
        result = _resolve_path("subdir/file.txt")
        expected = os.path.join(fake_root, "subdir", "file.txt")
        assert result == expected

    def test_absolute_path_within_project(self, fake_root: str) -> None:
        inner = os.path.join(fake_root, "data", "out.csv")
        assert _resolve_path(inner) == inner

    def test_traversal_raises(self, fake_root: str) -> None:
        with pytest.raises(ValueError, match="Path must be within"):
            _resolve_path("../../etc/passwd")

    def test_absolute_path_outside_project_raises(self, fake_root: str) -> None:
        with pytest.raises(ValueError, match="Path must be within"):
            _resolve_path("/etc/passwd")


# -- _validate_write_path ----------------------------------------------------


class TestValidateWritePath:
    def test_path_within_project(self, fake_root: str) -> None:
        inner = os.path.join(fake_root, "config.yaml")
        assert _validate_write_path(inner) == inner

    def test_path_outside_project_raises(self, fake_root: str) -> None:
        with pytest.raises(ValueError, match="Write path must be within"):
            _validate_write_path("/tmp/evil.yaml")

    def test_tilde_expansion_outside_project_raises(self, fake_root: str) -> None:
        with pytest.raises(ValueError, match="Write path must be within"):
            _validate_write_path("~/sneaky.yaml")


# -- _validate_results_dir ---------------------------------------------------


class TestValidateResultsDir:
    def test_valid_subdir(self, fake_root: str) -> None:
        subdir = os.path.join(fake_root, "results", "run1")
        result = _validate_results_dir(subdir)
        assert result == subdir

    def test_path_outside_project_returns_none(self, fake_root: str) -> None:
        assert _validate_results_dir("/tmp/other") is None

    def test_path_equal_to_root(self, fake_root: str) -> None:
        assert _validate_results_dir(fake_root) == fake_root
