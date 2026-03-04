# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.evaluator.utils.constants.score_label."""

from __future__ import annotations

from arksim.evaluator.utils.constants import score_label


class TestScoreLabel:
    def test_below_one(self) -> None:
        assert score_label(0.5) == "Poor"

    def test_poor_range(self) -> None:
        assert score_label(1.0) == "Poor"
        assert score_label(1.5) == "Poor"

    def test_needs_improvement(self) -> None:
        assert score_label(2.0) == "Needs Improvement"
        assert score_label(2.9) == "Needs Improvement"

    def test_good(self) -> None:
        assert score_label(3.0) == "Good"
        assert score_label(3.9) == "Good"

    def test_excellent(self) -> None:
        assert score_label(4.0) == "Excellent"
        assert score_label(4.9) == "Excellent"

    def test_perfect_five(self) -> None:
        assert score_label(5.0) == "Excellent"
