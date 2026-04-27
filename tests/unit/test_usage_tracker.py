# SPDX-License-Identifier: Apache-2.0
"""Unit tests for arksim.llms.chat.base.usage."""

from __future__ import annotations

import concurrent.futures
import importlib.util
import sys
import threading
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load usage.py directly to avoid heavy-dep import chain
# ---------------------------------------------------------------------------
_USAGE_PATH = Path(__file__).resolve().parents[2] / "arksim/llms/chat/base/usage.py"
_MODULE_NAME = "arksim.llms.chat.base.usage"


@pytest.fixture(scope="module")
def usage() -> types.ModuleType:
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _USAGE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# UsageTracker accumulation
# ---------------------------------------------------------------------------
class TestUsageTrackerAccumulation:
    def test_empty_tracker(self, usage: types.ModuleType) -> None:
        t = usage.UsageTracker()
        assert t.total_input_tokens == 0
        assert t.total_output_tokens == 0
        assert t.summary() == {}

    def test_single_record(self, usage: types.ModuleType) -> None:
        t = usage.UsageTracker()
        t.record("gpt-4o", "openai", 10, 5)
        assert t.total_input_tokens == 10
        assert t.total_output_tokens == 5

    def test_accumulates_multiple_records(self, usage: types.ModuleType) -> None:
        t = usage.UsageTracker()
        t.record("gpt-4o", "openai", 100, 50)
        t.record("gpt-4o", "openai", 200, 80)
        assert t.total_input_tokens == 300
        assert t.total_output_tokens == 130

    def test_summary_groups_by_provider_and_model(
        self, usage: types.ModuleType
    ) -> None:
        t = usage.UsageTracker()
        t.record("gpt-4o", "openai", 100, 40)
        t.record("gpt-4o", "openai", 50, 10)
        t.record("claude-3-5-sonnet", "anthropic", 200, 80)
        s = t.summary()
        assert s["openai/gpt-4o"] == {
            "input_tokens": 150,
            "output_tokens": 50,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
        }
        assert s["anthropic/claude-3-5-sonnet"] == {
            "input_tokens": 200,
            "output_tokens": 80,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
        }

    def test_records_and_aggregates_cached_and_reasoning(
        self, usage: types.ModuleType
    ) -> None:
        t = usage.UsageTracker()
        t.record("gpt-5", "openai", 100, 40, cached_tokens=30, reasoning_tokens=10)
        t.record("gpt-5", "openai", 50, 20, cached_tokens=5, reasoning_tokens=4)
        assert t.total_input_tokens == 150
        assert t.total_output_tokens == 60
        assert t.total_cached_tokens == 35
        assert t.total_reasoning_tokens == 14
        assert t.summary()["openai/gpt-5"] == {
            "input_tokens": 150,
            "output_tokens": 60,
            "cached_tokens": 35,
            "reasoning_tokens": 14,
        }

    def test_track_usage_passes_cached_and_reasoning(
        self, usage: types.ModuleType
    ) -> None:
        tracker = usage.UsageTracker()
        token = usage.set_current_tracker(tracker)
        try:
            usage.track_usage(
                "claude", "anthropic", 200, 80, cached_tokens=120, reasoning_tokens=0
            )
            assert tracker.total_cached_tokens == 120
            assert tracker.total_reasoning_tokens == 0
        finally:
            usage.reset_current_tracker(token)

    def test_same_model_name_different_providers_stay_separate(
        self, usage: types.ModuleType
    ) -> None:
        t = usage.UsageTracker()
        t.record("gpt-4o", "openai", 100, 40)
        t.record("gpt-4o", "azure", 200, 80)
        s = t.summary()
        assert "openai/gpt-4o" in s
        assert "azure/gpt-4o" in s
        assert s["openai/gpt-4o"]["input_tokens"] == 100
        assert s["azure/gpt-4o"]["input_tokens"] == 200


# ---------------------------------------------------------------------------
# track_usage no-op when no tracker is active
# ---------------------------------------------------------------------------
class TestTrackUsageNoOp:
    def test_no_tracker_is_noop(self, usage: types.ModuleType) -> None:
        usage.reset_current_tracker(usage.set_current_tracker(usage.UsageTracker()))
        # Ensure no tracker is set before calling
        assert usage._current_tracker.get() is None
        usage.track_usage("gpt-4o", "openai", 999, 999)
        assert usage._current_tracker.get() is None

    def test_no_tracker_by_default(self, usage: types.ModuleType) -> None:
        assert usage._current_tracker.get() is None


# ---------------------------------------------------------------------------
# set_current_tracker / reset_current_tracker round-trip
# ---------------------------------------------------------------------------
class TestTrackerContextVar:
    def test_set_and_get(self, usage: types.ModuleType) -> None:
        tracker = usage.UsageTracker()
        token = usage.set_current_tracker(tracker)
        try:
            assert usage._current_tracker.get() is tracker
        finally:
            usage.reset_current_tracker(token)

    def test_reset_restores_none(self, usage: types.ModuleType) -> None:
        tracker = usage.UsageTracker()
        token = usage.set_current_tracker(tracker)
        usage.reset_current_tracker(token)
        assert usage._current_tracker.get() is None

    def test_track_usage_records_on_active_tracker(
        self, usage: types.ModuleType
    ) -> None:
        tracker = usage.UsageTracker()
        token = usage.set_current_tracker(tracker)
        try:
            usage.track_usage("claude-3-5-sonnet", "anthropic", 50, 20)
            assert tracker.total_input_tokens == 50
            assert tracker.total_output_tokens == 20
        finally:
            usage.reset_current_tracker(token)

    def test_nested_tracker_inner_shadows_outer(self, usage: types.ModuleType) -> None:
        outer = usage.UsageTracker()
        inner = usage.UsageTracker()
        t1 = usage.set_current_tracker(outer)
        t2 = usage.set_current_tracker(inner)
        try:
            usage.track_usage("m", "p", 5, 2)
            assert inner.total_input_tokens == 5
            assert outer.total_input_tokens == 0
        finally:
            usage.reset_current_tracker(t2)
            usage.reset_current_tracker(t1)


# ---------------------------------------------------------------------------
# Threads without copy_context() do NOT inherit the parent tracker
# ---------------------------------------------------------------------------
class TestThreadIsolation:
    def test_raw_thread_does_not_inherit_tracker(self, usage: types.ModuleType) -> None:
        """Documents why copy_context().run is required in the evaluator."""
        parent_tracker = usage.UsageTracker()
        token = usage.set_current_tracker(parent_tracker)

        thread_saw_tracker: list[bool] = []

        def worker() -> None:
            # A plain thread starts with a fresh context — no tracker.
            thread_saw_tracker.append(usage._current_tracker.get() is not None)
            usage.track_usage("m", "p", 100, 50)

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        usage.reset_current_tracker(token)

        assert thread_saw_tracker == [False]
        assert parent_tracker.total_input_tokens == 0

    def test_executor_without_copy_context_does_not_inherit(
        self, usage: types.ModuleType
    ) -> None:
        parent_tracker = usage.UsageTracker()
        token = usage.set_current_tracker(parent_tracker)

        def worker() -> bool:
            usage.track_usage("m", "p", 100, 50)
            return usage._current_tracker.get() is None

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            saw_no_tracker = ex.submit(worker).result()

        usage.reset_current_tracker(token)

        assert saw_no_tracker
        assert parent_tracker.total_input_tokens == 0


# ---------------------------------------------------------------------------
# Concurrent track_usage() via copy_context().run gives correct totals
# ---------------------------------------------------------------------------
class TestConcurrentTracking:
    def test_copy_context_run_inherits_tracker(self, usage: types.ModuleType) -> None:
        import contextvars

        tracker = usage.UsageTracker()
        token = usage.set_current_tracker(tracker)

        ctx = contextvars.copy_context()

        def worker() -> None:
            usage.track_usage("gpt-4o", "openai", 10, 5)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(ctx.run, worker) for _ in range(10)]
            concurrent.futures.wait(futures)

        usage.reset_current_tracker(token)

        assert tracker.total_input_tokens == 100
        assert tracker.total_output_tokens == 50

    def test_concurrent_records_are_all_captured(self, usage: types.ModuleType) -> None:
        import contextvars

        tracker = usage.UsageTracker()
        token = usage.set_current_tracker(tracker)

        ctx = contextvars.copy_context()
        n_threads = 20

        def worker() -> None:
            usage.track_usage("m", "p", 1, 1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as ex:
            concurrent.futures.wait(
                [ex.submit(ctx.run, worker) for _ in range(n_threads)]
            )

        usage.reset_current_tracker(token)

        assert len(tracker.records) == n_threads
        assert tracker.total_input_tokens == n_threads
        assert tracker.total_output_tokens == n_threads


# ---------------------------------------------------------------------------
# usage_label: labels are attached to records and merge across nested layers
# ---------------------------------------------------------------------------
class TestUsageLabel:
    def test_no_label_yields_empty_label_dict(self, usage: types.ModuleType) -> None:
        with usage.usage_scope() as tracker:
            usage.track_usage("m", "p", 10, 5)
        assert tracker.records[0].labels == {}

    def test_single_label_attached_to_record(self, usage: types.ModuleType) -> None:
        with (
            usage.usage_scope() as tracker,
            usage.usage_label(component="conversation"),
        ):
            usage.track_usage("m", "p", 10, 5)
        assert tracker.records[0].labels == {"component": "conversation"}

    def test_nested_labels_merge(self, usage: types.ModuleType) -> None:
        with (
            usage.usage_scope() as tracker,
            usage.usage_label(component="conversation", scenario_id="s1"),
            usage.usage_label(conversation_id="c1", turn_id="0"),
        ):
            usage.track_usage("m", "p", 10, 5)
        assert tracker.records[0].labels == {
            "component": "conversation",
            "scenario_id": "s1",
            "conversation_id": "c1",
            "turn_id": "0",
        }

    def test_inner_layer_overrides_outer_for_same_key(
        self, usage: types.ModuleType
    ) -> None:
        with (
            usage.usage_scope() as tracker,
            usage.usage_label(component="conversation"),
        ):
            with usage.usage_label(component="multi_knowledge"):
                usage.track_usage("m", "p", 1, 1)
            usage.track_usage("m", "p", 2, 2)
        assert tracker.records[0].labels["component"] == "multi_knowledge"
        assert tracker.records[1].labels["component"] == "conversation"

    def test_label_layer_pops_on_exit(self, usage: types.ModuleType) -> None:
        with usage.usage_scope() as tracker:
            with usage.usage_label(turn_id="0"):
                usage.track_usage("m", "p", 1, 1)
            usage.track_usage("m", "p", 2, 2)
        assert tracker.records[0].labels == {"turn_id": "0"}
        assert tracker.records[1].labels == {}

    def test_label_layer_pops_on_exception(self, usage: types.ModuleType) -> None:
        with usage.usage_scope() as tracker:
            with pytest.raises(RuntimeError), usage.usage_label(turn_id="0"):
                raise RuntimeError("boom")
            usage.track_usage("m", "p", 1, 1)
        assert tracker.records[0].labels == {}

    def test_label_values_coerced_to_str(self, usage: types.ModuleType) -> None:
        with usage.usage_scope() as tracker, usage.usage_label(turn_id=3):
            usage.track_usage("m", "p", 1, 1)
        assert tracker.records[0].labels == {"turn_id": "3"}


# ---------------------------------------------------------------------------
# summary_by: grouping, key filtering, where clause
# ---------------------------------------------------------------------------
class TestSummaryBy:
    def test_groups_by_single_key(self, usage: types.ModuleType) -> None:
        t = usage.UsageTracker()
        t.record("m", "p", 10, 5, labels={"conversation_id": "c1"})
        t.record("m", "p", 20, 10, labels={"conversation_id": "c1"})
        t.record("m", "p", 30, 15, labels={"conversation_id": "c2"})
        rows = t.summary_by("conversation_id")
        by_id = {r["conversation_id"]: r for r in rows}
        assert by_id["c1"]["input_tokens"] == 30
        assert by_id["c1"]["output_tokens"] == 15
        assert by_id["c1"]["calls"] == 2
        assert by_id["c2"]["input_tokens"] == 30
        assert by_id["c2"]["calls"] == 1

    def test_groups_by_multiple_keys(self, usage: types.ModuleType) -> None:
        t = usage.UsageTracker()
        t.record("m", "p", 1, 1, labels={"conversation_id": "c1", "turn_id": "0"})
        t.record("m", "p", 2, 2, labels={"conversation_id": "c1", "turn_id": "0"})
        t.record("m", "p", 4, 4, labels={"conversation_id": "c1", "turn_id": "1"})
        rows = t.summary_by("conversation_id", "turn_id")
        bucket = {(r["conversation_id"], r["turn_id"]): r for r in rows}
        assert bucket[("c1", "0")]["input_tokens"] == 3
        assert bucket[("c1", "0")]["calls"] == 2
        assert bucket[("c1", "1")]["input_tokens"] == 4

    def test_skips_records_missing_requested_keys(
        self, usage: types.ModuleType
    ) -> None:
        t = usage.UsageTracker()
        t.record("m", "p", 10, 5, labels={"component": "conversation"})
        t.record(
            "m",
            "p",
            20,
            10,
            labels={"component": "conversation", "conversation_id": "c1"},
        )
        rows = t.summary_by("conversation_id")
        assert len(rows) == 1
        assert rows[0]["conversation_id"] == "c1"
        assert rows[0]["input_tokens"] == 20

    def test_where_filter(self, usage: types.ModuleType) -> None:
        t = usage.UsageTracker()
        t.record(
            "m",
            "p",
            10,
            5,
            labels={"component": "conversation", "conversation_id": "c1"},
        )
        t.record(
            "m",
            "p",
            100,
            50,
            labels={"component": "multi_knowledge", "conversation_id": "c1"},
        )
        rows = t.summary_by("conversation_id", where={"component": "conversation"})
        assert len(rows) == 1
        assert rows[0]["input_tokens"] == 10

    def test_empty_tracker_returns_empty(self, usage: types.ModuleType) -> None:
        t = usage.UsageTracker()
        assert t.summary_by("conversation_id") == []


# ---------------------------------------------------------------------------
# TokenUsage exposes a breakdowns field for serialization
# ---------------------------------------------------------------------------
class TestTokenUsageBreakdowns:
    def test_breakdowns_default_empty(self, usage: types.ModuleType) -> None:
        tu = usage.TokenUsage()
        assert tu.breakdowns == {}

    def test_breakdowns_round_trip(self, usage: types.ModuleType) -> None:
        tu = usage.TokenUsage(
            breakdowns={
                "by_conversation": [
                    {
                        "conversation_id": "c1",
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "cached_tokens": 0,
                        "reasoning_tokens": 0,
                        "calls": 1,
                    }
                ]
            }
        )
        dumped = tu.model_dump()
        assert dumped["breakdowns"]["by_conversation"][0]["conversation_id"] == "c1"


# ---------------------------------------------------------------------------
# track_usage merges the active label stack onto each record
# ---------------------------------------------------------------------------
class TestTrackUsageMergesLabels:
    def test_track_usage_picks_up_active_labels(self, usage: types.ModuleType) -> None:
        with (
            usage.usage_scope() as tracker,
            usage.usage_label(component="evaluation", metric="helpfulness"),
        ):
            usage.track_usage("gpt-4o", "openai", 100, 40)
        assert tracker.records[0].labels == {
            "component": "evaluation",
            "metric": "helpfulness",
        }
