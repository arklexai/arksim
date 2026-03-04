# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.evaluator.prompt_registry."""

from __future__ import annotations

from arksim.evaluator.prompt_registry import (
    PROMPT_REGISTRY,
    get_categories,
    get_prompts_by_category,
)


class TestPromptRegistry:
    def test_registry_not_empty(self) -> None:
        assert len(PROMPT_REGISTRY) > 0

    def test_all_entries_have_prompts(self) -> None:
        for cat in PROMPT_REGISTRY:
            assert len(cat.prompts) > 0
            for prompt in cat.prompts:
                assert prompt.name
                assert prompt.text


class TestGetCategories:
    def test_returns_sorted(self) -> None:
        cats = get_categories()
        assert cats == sorted(cats)

    def test_includes_expected(self) -> None:
        cats = get_categories()
        for expected in ["helpfulness", "coherence", "goal_completion"]:
            assert expected in cats


class TestGetPromptsByCategory:
    def test_none_returns_all(self) -> None:
        result = get_prompts_by_category(None)
        assert len(result) == len(PROMPT_REGISTRY)

    def test_specific_category(self) -> None:
        result = get_prompts_by_category("helpfulness")
        assert len(result) == 1
        assert result[0].category == "helpfulness"

    def test_nonexistent_category(self) -> None:
        result = get_prompts_by_category("nonexistent")
        assert result == []

    def test_case_insensitive(self) -> None:
        result = get_prompts_by_category("HELPFULNESS")
        assert len(result) == 1
