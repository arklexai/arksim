# SPDX-License-Identifier: Apache-2.0
"""Track in-flight tool calls keyed by SDK run id (split-event correlation)."""

from __future__ import annotations

import threading
import time
from typing import Any


class PendingToolCalls:
    """Bounded-growth pending map for split-event correlation.

    Stale entries are eagerly swept on every add() and pop() call,
    evicting anything older than max_age_seconds (default 60s).
    Per-adapter-instance, not class-level.

    Thread-safe: a ``threading.Lock`` guards every read and write of the
    internal dict so concurrent ``add``/``pop`` calls from LangChain's
    thread-pool dispatch (where blocking tools run on a worker thread)
    cannot race. The lock scope covers only in-memory dict operations.
    """

    def __init__(self, max_age_seconds: float = 60.0) -> None:
        self._pending: dict[str, tuple[dict[str, Any], float]] = {}
        self._max_age = max_age_seconds
        self._lock = threading.Lock()

    def add(self, run_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._sweep_stale_locked()
            self._pending[run_id] = (payload, time.monotonic())

    def pop(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._sweep_stale_locked()
            entry = self._pending.pop(run_id, None)
        return entry[0] if entry else None

    def _sweep_stale_locked(self) -> None:
        """Evict stale entries. Caller must hold ``self._lock``."""
        now = time.monotonic()
        stale = [
            rid for rid, (_, t) in self._pending.items() if now - t > self._max_age
        ]
        for rid in stale:
            self._pending.pop(rid, None)
