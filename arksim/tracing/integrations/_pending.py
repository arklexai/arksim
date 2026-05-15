# SPDX-License-Identifier: Apache-2.0
"""Track in-flight tool calls keyed by SDK run id (split-event correlation)."""

from __future__ import annotations

import time
from typing import Any


class PendingToolCalls:
    """Bounded-growth pending map for split-event correlation.

    Stale entries are eagerly swept on every add() and pop() call,
    evicting anything older than max_age_seconds (default 60s).
    Per-adapter-instance, not class-level.
    """

    def __init__(self, max_age_seconds: float = 60.0) -> None:
        self._pending: dict[str, tuple[dict[str, Any], float]] = {}
        self._max_age = max_age_seconds

    def add(self, run_id: str, payload: dict[str, Any]) -> None:
        self._sweep_stale()
        self._pending[run_id] = (payload, time.monotonic())

    def pop(self, run_id: str) -> dict[str, Any] | None:
        self._sweep_stale()
        entry = self._pending.pop(run_id, None)
        return entry[0] if entry else None

    def _sweep_stale(self) -> None:
        now = time.monotonic()
        stale = [
            rid for rid, (_, t) in self._pending.items() if now - t > self._max_age
        ]
        for rid in stale:
            self._pending.pop(rid, None)
