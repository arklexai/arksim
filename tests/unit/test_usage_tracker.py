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
        assert t.total_tokens == 0
        assert t.records == []

    def test_single_record(self, usage: types.ModuleType) -> None:
        t = usage.UsageTracker()
        t.record("gpt-4o", "openai", 10, 5)
        assert t.total_input_tokens == 10
        assert t.total_output_tokens == 5
        assert t.total_tokens == 15

    def test_accumulates_multiple_records(self, usage: types.ModuleType) -> None:
        t = usage.UsageTracker()
        t.record("gpt-4o", "openai", 100, 50)
        t.record("gpt-4o", "openai", 200, 80)
        assert t.total_input_tokens == 300
        assert t.total_output_tokens == 130

    def test_records_cache_and_reasoning(self, usage: types.ModuleType) -> None:
        t = usage.UsageTracker()
        t.record(
            "gpt-5",
            "openai",
            100,
            40,
            cache_read_tokens=30,
            reasoning_tokens=10,
        )
        t.record(
            "gpt-5",
            "openai",
            50,
            20,
            cache_read_tokens=5,
            reasoning_tokens=4,
        )
        assert t.total_cache_read_tokens == 35
        assert t.total_cache_creation_tokens == 0
        assert t.total_reasoning_tokens == 14

    def test_records_cache_creation(self, usage: types.ModuleType) -> None:
        t = usage.UsageTracker()
        t.record(
            "claude",
            "anthropic",
            500,
            80,
            cache_read_tokens=100,
            cache_creation_tokens=300,
        )
        assert t.total_cache_read_tokens == 100
        assert t.total_cache_creation_tokens == 300

    def test_explicit_total_tokens_overrides_default(
        self, usage: types.ModuleType
    ) -> None:
        t = usage.UsageTracker()
        # Some providers report a total that differs from input + output.
        t.record("gemini-2.0", "google", 100, 40, total_tokens=145)
        assert t.total_tokens == 145

    def test_record_carries_component_when_passed(
        self, usage: types.ModuleType
    ) -> None:
        t = usage.UsageTracker()
        t.record("m", "p", 1, 1, component="goal_generation")
        assert t.records[0].component == "goal_generation"


# ---------------------------------------------------------------------------
# track_usage no-op when no tracker is active
# ---------------------------------------------------------------------------
class TestTrackUsageNoOp:
    def test_no_tracker_is_noop(self, usage: types.ModuleType) -> None:
        # Ensure no tracker is set before calling
        assert usage._current_tracker.get() is None
        usage.track_usage("gpt-4o", "openai", 999, 999)
        assert usage._current_tracker.get() is None


# ---------------------------------------------------------------------------
# set_current_tracker / reset_current_tracker round-trip
# ---------------------------------------------------------------------------
class TestTrackerContextVar:
    def test_set_and_reset(self, usage: types.ModuleType) -> None:
        tracker = usage.UsageTracker()
        token = usage.set_current_tracker(tracker)
        try:
            assert usage._current_tracker.get() is tracker
        finally:
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
        """Documents why copy_context().run is required to fan out work."""
        parent_tracker = usage.UsageTracker()
        token = usage.set_current_tracker(parent_tracker)

        thread_saw_tracker: list[bool] = []

        def worker() -> None:
            thread_saw_tracker.append(usage._current_tracker.get() is not None)
            usage.track_usage("m", "p", 100, 50)

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        usage.reset_current_tracker(token)

        assert thread_saw_tracker == [False]
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
        assert len(tracker.records) == 10


# ---------------------------------------------------------------------------
# usage_run: no args, fresh tag context
# ---------------------------------------------------------------------------
class TestUsageRun:
    def test_yields_fresh_tracker(self, usage: types.ModuleType) -> None:
        with usage.usage_run() as tracker:
            assert isinstance(tracker, usage.UsageTracker)
            assert tracker.records == []

    def test_tracker_active_inside_block(self, usage: types.ModuleType) -> None:
        with usage.usage_run() as tracker:
            usage.track_usage("m", "p", 1, 1)
        assert tracker.total_input_tokens == 1

    def test_tracker_deactivates_on_exit(self, usage: types.ModuleType) -> None:
        with usage.usage_run():
            assert usage._current_tracker.get() is not None
        assert usage._current_tracker.get() is None

    def test_nested_run_does_not_inherit_outer_tags(
        self, usage: types.ModuleType
    ) -> None:
        with (
            usage.usage_run(),
            usage.usage_tags(component="goal_generation"),
            usage.usage_run() as inner,
        ):
            usage.track_usage("m", "p", 1, 1)
        assert inner.records[0].component is None


# ---------------------------------------------------------------------------
# usage_tags: only component is supported now
# ---------------------------------------------------------------------------
class TestUsageTags:
    def test_no_tag_yields_none_component(self, usage: types.ModuleType) -> None:
        with usage.usage_run() as tracker:
            usage.track_usage("m", "p", 10, 5)
        assert tracker.records[0].component is None

    def test_component_tag_attached(self, usage: types.ModuleType) -> None:
        with (
            usage.usage_run() as tracker,
            usage.usage_tags(component="profile_generation"),
        ):
            usage.track_usage("m", "p", 10, 5)
        assert tracker.records[0].component == "profile_generation"

    def test_inner_layer_overrides_outer(self, usage: types.ModuleType) -> None:
        with (
            usage.usage_run() as tracker,
            usage.usage_tags(component="goal_generation"),
        ):
            with usage.usage_tags(component="error_detection"):
                usage.track_usage("m", "p", 1, 1)
            usage.track_usage("m", "p", 2, 2)
        assert tracker.records[0].component == "error_detection"
        assert tracker.records[1].component == "goal_generation"

    def test_tag_layer_pops_on_exception(self, usage: types.ModuleType) -> None:
        with usage.usage_run() as tracker:
            with pytest.raises(RuntimeError), usage.usage_tags(component="x"):
                raise RuntimeError("boom")
            usage.track_usage("m", "p", 1, 1)
        assert tracker.records[0].component is None


# ---------------------------------------------------------------------------
# track_usage merges the active component tag onto each record
# ---------------------------------------------------------------------------
class TestTrackUsageMergesTags:
    def test_track_usage_picks_up_active_component(
        self, usage: types.ModuleType
    ) -> None:
        with (
            usage.usage_run() as tracker,
            usage.usage_tags(component="score"),
        ):
            usage.track_usage("gpt-4o", "openai", 100, 40)
        rec = tracker.records[0]
        assert rec.component == "score"
        assert rec.model == "gpt-4o"
        assert rec.provider == "openai"


# ---------------------------------------------------------------------------
# TokenUsage exposes only totals
# ---------------------------------------------------------------------------
class TestTokenUsage:
    def test_defaults_zero(self, usage: types.ModuleType) -> None:
        tu = usage.TokenUsage()
        assert tu.total_input_tokens == 0
        assert tu.total_output_tokens == 0
        assert tu.total_tokens == 0

    def test_round_trip(self, usage: types.ModuleType) -> None:
        tu = usage.TokenUsage(
            total_input_tokens=100,
            total_output_tokens=40,
            total_tokens=140,
        )
        dumped = tu.model_dump()
        assert dumped["total_input_tokens"] == 100
        assert dumped["total_tokens"] == 140
