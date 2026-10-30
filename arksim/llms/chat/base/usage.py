# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, replace

from pydantic import BaseModel


def clean_usage_value(value: object) -> int:
    """Coerce a provider usage field to a non-negative int."""
    if value is None:
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


class TokenUsage(BaseModel):
    """Token usage totals from LLM calls."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class UsageRecord:
    """Token usage from a single LLM call."""

    model: str = ""
    provider: str = ""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    component: str | None = None


@dataclass
class UsageTracker:
    """Accumulates token usage across multiple LLM calls."""

    records: list[UsageRecord] = field(default_factory=list)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def record(
        self,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
        reasoning_tokens: int = 0,
        total_tokens: int | None = None,
        component: str | None = None,
    ) -> None:
        with self._lock:
            self.records.append(
                UsageRecord(
                    model=model,
                    provider=provider,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_creation_tokens=cache_creation_tokens,
                    reasoning_tokens=reasoning_tokens,
                    total_tokens=(
                        total_tokens
                        if total_tokens is not None
                        else input_tokens + output_tokens
                    ),
                    component=component,
                )
            )

    @property
    def total_input_tokens(self) -> int:
        with self._lock:
            return sum(r.input_tokens for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        with self._lock:
            return sum(r.output_tokens for r in self.records)

    @property
    def total_cache_read_tokens(self) -> int:
        with self._lock:
            return sum(r.cache_read_tokens for r in self.records)

    @property
    def total_cache_creation_tokens(self) -> int:
        with self._lock:
            return sum(r.cache_creation_tokens for r in self.records)

    @property
    def total_reasoning_tokens(self) -> int:
        with self._lock:
            return sum(r.reasoning_tokens for r in self.records)

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return sum(r.total_tokens for r in self.records)


_current_tracker: ContextVar[UsageTracker | None] = ContextVar(
    "llm_usage_tracker", default=None
)

_EMPTY_TAGS = UsageRecord()
# UsageRecord is frozen, so sharing one default across contexts is safe.
_current_tags: ContextVar[UsageRecord] = ContextVar(
    "llm_usage_tags",
    default=_EMPTY_TAGS,  # noqa: B039
)


def set_current_tracker(tracker: UsageTracker) -> Token[UsageTracker | None]:
    """Set the current tracker. Returns a token for reset_current_tracker."""
    return _current_tracker.set(tracker)


def reset_current_tracker(token: Token[UsageTracker | None]) -> None:
    """Reset the tracker context variable using the token from set_current_tracker."""
    _current_tracker.reset(token)


@contextmanager
def usage_run() -> Iterator[UsageTracker]:
    """Activate a fresh :class:`UsageTracker` for the duration of the block.

    Resets the per-record tag context so a nested run does not inherit
    component tags from an outer one.
    """
    tracker = UsageTracker()
    tracker_token = set_current_tracker(tracker)
    tags_token = _current_tags.set(_EMPTY_TAGS)
    try:
        yield tracker
    finally:
        _current_tags.reset(tags_token)
        reset_current_tracker(tracker_token)


@contextmanager
def usage_tags(*, component: str | None = None) -> Iterator[None]:
    """Push a component tag onto the active run context."""
    if component is None:
        yield
        return
    token = _current_tags.set(replace(_current_tags.get(), component=component))
    try:
        yield
    finally:
        _current_tags.reset(token)


def track_usage(
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    reasoning_tokens: int = 0,
    total_tokens: int | None = None,
) -> None:
    """Record token usage on the current tracker, if one is active.

    Callers are expected to pass clean non-negative ints; provider
    adapters normalize None / negative values before calling this.
    """
    tracker = _current_tracker.get()
    if tracker is None:
        return
    tracker.record(
        model,
        provider,
        input_tokens,
        output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
        component=_current_tags.get().component,
    )
