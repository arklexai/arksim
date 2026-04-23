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
        assert s["openai/gpt-4o"] == {"input_tokens": 150, "output_tokens": 50}
        assert s["anthropic/claude-3-5-sonnet"] == {
            "input_tokens": 200,
            "output_tokens": 80,
        }

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

    def test_nested_tracker_not_supported(self, usage: types.ModuleType) -> None:
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
