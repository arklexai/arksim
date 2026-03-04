# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.utils.concurrency.workers."""

from __future__ import annotations

import pytest

from arksim.utils.concurrency.workers import resolve_num_workers, validate_num_workers


class TestValidateNumWorkers:
    def test_valid_int(self) -> None:
        validate_num_workers(4)

    def test_auto_string(self) -> None:
        validate_num_workers("auto")

    def test_invalid_string_raises(self) -> None:
        with pytest.raises(ValueError, match="num_workers must be a positive integer"):
            validate_num_workers("fast")

    def test_float_raises(self) -> None:
        with pytest.raises(ValueError, match="num_workers must be a positive integer"):
            validate_num_workers(3.5)

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="num_workers must be a positive integer"):
            validate_num_workers(0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="num_workers must be a positive integer"):
            validate_num_workers(-1)


class TestResolveNumWorkers:
    def test_auto_returns_auto_value(self) -> None:
        assert resolve_num_workers("auto", 100) == 100

    def test_int_returns_itself(self) -> None:
        assert resolve_num_workers(4, 8) == 4

    def test_one_returns_one(self) -> None:
        assert resolve_num_workers(1, 8) == 1
