# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.simulation_engine.utils.utils."""

from arksim.simulation_engine.utils.utils import flip_hist


class TestFlipHist:
    def test_empty_list(self) -> None:
        assert flip_hist([]) == []

    def test_user_becomes_assistant(self) -> None:
        hist = [{"role": "user", "content": "hi"}]
        result = flip_hist(hist)
        assert result == [{"role": "assistant", "content": "hi"}]

    def test_assistant_becomes_user(self) -> None:
        hist = [{"role": "assistant", "content": "hello"}]
        result = flip_hist(hist)
        assert result == [{"role": "user", "content": "hello"}]

    def test_system_messages_skipped(self) -> None:
        hist = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hi"},
        ]
        result = flip_hist(hist)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"

    def test_extra_fields_preserved(self) -> None:
        hist = [{"role": "user", "content": "hi", "timestamp": "2024-01-01"}]
        result = flip_hist(hist)
        assert result[0]["timestamp"] == "2024-01-01"
        assert result[0]["role"] == "assistant"

    def test_turn_without_role_kept(self) -> None:
        hist = [{"content": "orphan"}]
        result = flip_hist(hist)
        assert result == [{"content": "orphan"}]

    def test_full_conversation(self) -> None:
        hist = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        result = flip_hist(hist)
        assert len(result) == 3
        assert [m["role"] for m in result] == ["assistant", "user", "assistant"]
