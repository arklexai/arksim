# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.utils.output.utils.resolve_output_dir."""

from __future__ import annotations

import os

from arksim.utils.output.utils import resolve_output_dir


class TestResolveOutputDir:
    def test_nonexistent_returns_as_is(self, temp_dir: str) -> None:
        path = os.path.join(temp_dir, "new_dir")
        assert resolve_output_dir(path) == path

    def test_existing_file_gets_timestamp(self, temp_dir: str) -> None:
        path = os.path.join(temp_dir, "output.json")
        with open(path, "w") as f:
            f.write("{}")

        result = resolve_output_dir(path)
        assert result != path
        assert result.startswith(os.path.join(temp_dir, "output_"))
        assert result.endswith(".json")

    def test_existing_directory_gets_timestamp(self, temp_dir: str) -> None:
        path = os.path.join(temp_dir, "results")
        os.makedirs(path)

        result = resolve_output_dir(path)
        assert result != path
        assert result.startswith(path + "_")
