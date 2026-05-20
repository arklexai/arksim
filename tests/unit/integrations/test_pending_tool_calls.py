# SPDX-License-Identifier: Apache-2.0
"""Tests for PendingToolCalls split-event correlation helper."""

from __future__ import annotations

import threading
from uuid import uuid4

import pytest

from arksim.tracing.integrations import _pending
from arksim.tracing.integrations._pending import PendingToolCalls


def test_add_then_pop_returns_payload() -> None:
    pending = PendingToolCalls()
    payload = {"name": "search", "args": {"q": "x"}}
    pending.add("r1", payload)
    assert pending.pop("r1") == payload


def test_pop_missing_returns_none() -> None:
    pending = PendingToolCalls()
    assert pending.pop("never-added") is None


def test_stale_entries_swept_on_add(monkeypatch: pytest.MonkeyPatch) -> None:
    pending = PendingToolCalls(max_age_seconds=60.0)
    now = 1000.0
    monkeypatch.setattr(_pending.time, "monotonic", lambda: now)
    pending.add("r1", {"k": "v"})

    later = now + 100.0
    monkeypatch.setattr(_pending.time, "monotonic", lambda: later)
    pending.add("r2", {"k2": "v2"})

    assert pending.pop("r1") is None
    assert pending.pop("r2") == {"k2": "v2"}


def test_stale_entries_swept_on_pop(monkeypatch: pytest.MonkeyPatch) -> None:
    pending = PendingToolCalls(max_age_seconds=60.0)
    now = 1000.0
    monkeypatch.setattr(_pending.time, "monotonic", lambda: now)
    pending.add("r1", {"k": "v"})

    later = now + 100.0
    monkeypatch.setattr(_pending.time, "monotonic", lambda: later)
    assert pending.pop("r1") is None


def test_pop_consumes_entry() -> None:
    pending = PendingToolCalls()
    payload = {"k": "v"}
    pending.add("r1", payload)
    assert pending.pop("r1") == payload
    assert pending.pop("r1") is None


def test_concurrent_add_pop_thread_safety() -> None:
    """Concurrent add/pop from 10 threads must not raise or corrupt state.

    LangChain dispatches sync ``BaseCallbackHandler`` methods on a thread
    pool when tools block, so ``PendingToolCalls`` is reachable from
    multiple threads. Without the internal lock, ``dict`` iteration during
    sweep can race with mutation and raise ``RuntimeError: dictionary
    changed size during iteration``.
    """
    pending = PendingToolCalls()
    iterations = 100
    errors: list[BaseException] = []
    barrier = threading.Barrier(10)

    def worker() -> None:
        barrier.wait()
        try:
            for _ in range(iterations):
                rid = uuid4().hex
                pending.add(rid, {"k": rid})
                pending.pop(rid)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    # Each worker pops what it adds, so the map ends empty.
    assert pending.pop("anything") is None
