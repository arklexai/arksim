# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.simulation_engine.core.multi_knowledge_handling."""

import random

import pytest

from arksim.simulation_engine.core.multi_knowledge_handling import (
    combine_all_turn_knowledge,
    combine_knowledge,
    pick_one_for_turn,
    pick_one_turn_knowledge,
)


class TestCombineKnowledge:
    def test_empty(self) -> None:
        assert combine_knowledge([]) == ""

    def test_single_item(self) -> None:
        result = combine_knowledge(["fact A"])
        assert result == "Knowledge 1:\nfact A"

    def test_multiple_items(self) -> None:
        result = combine_knowledge(["alpha", "beta", "gamma"])
        assert "Knowledge 1:\nalpha" in result
        assert "Knowledge 2:\nbeta" in result
        assert "Knowledge 3:\ngamma" in result


class TestPickOneForTurn:
    def test_empty_knowledge(self) -> None:
        content, used = pick_one_for_turn([])
        assert content == ""
        assert used == set()

    def test_single_item(self) -> None:
        content, used = pick_one_for_turn(["only one"])
        assert content == "only one"
        assert used == {0}

    def test_rotation_without_repeat(self) -> None:
        items = ["a", "b", "c"]
        used: set[int] = set()
        seen: list[str] = []
        for _ in range(3):
            content, used = pick_one_for_turn(items, used_indices=used)
            seen.append(content)
        assert sorted(seen) == sorted(items)

    def test_cycle_after_exhaustion(self) -> None:
        items = ["a", "b"]
        used = {0, 1}
        content, new_used = pick_one_for_turn(items, used_indices=used)
        assert content in ("a", "b")
        assert len(new_used) == 1

    def test_seed_reproducibility(self) -> None:
        items = ["x", "y", "z"]
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        c1, _ = pick_one_for_turn(items, rng=rng1)
        c2, _ = pick_one_for_turn(items, rng=rng2)
        assert c1 == c2

    def test_strips_whitespace(self) -> None:
        content, _ = pick_one_for_turn(["  padded  "])
        assert content == "padded"


class TestPickOneTurnKnowledge:
    @pytest.mark.asyncio
    async def test_empty_returns_empty(self) -> None:
        content, state = await pick_one_turn_knowledge(None, [], [], {})
        assert content == ""

    @pytest.mark.asyncio
    async def test_single_string(self) -> None:
        content, state = await pick_one_turn_knowledge(None, [], "  hello  ", {})
        assert content == "hello"

    @pytest.mark.asyncio
    async def test_list_rotates(self) -> None:
        items = ["a", "b"]
        state: dict = {}
        seen: list[str] = []
        for _ in range(2):
            content, state = await pick_one_turn_knowledge(None, [], items, state)
            seen.append(content)
        assert sorted(seen) == ["a", "b"]


class TestCombineAllTurnKnowledge:
    @pytest.mark.asyncio
    async def test_empty_returns_empty(self) -> None:
        content, state = await combine_all_turn_knowledge(None, [], [], {})
        assert content == ""

    @pytest.mark.asyncio
    async def test_single_string(self) -> None:
        content, _ = await combine_all_turn_knowledge(None, [], "  text  ", {})
        assert content == "text"

    @pytest.mark.asyncio
    async def test_list_combines(self) -> None:
        content, _ = await combine_all_turn_knowledge(None, [], ["a", "b"], {})
        assert "Knowledge 1:" in content
        assert "Knowledge 2:" in content
