# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field

from pydantic import BaseModel
from pydantic import Field as PydanticField

logger = logging.getLogger(__name__)


def clean_usage_value(value: object) -> int:
    """Coerce a provider usage field to a non-negative int."""
    if value is None:
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


class TokenUsage(BaseModel):
    """Token usage summary from LLM calls."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    total_reasoning_tokens: int = 0
    by_model: dict[str, dict[str, int]] = PydanticField(default_factory=dict)
    breakdowns: dict[str, list[dict[str, str | int]]] = PydanticField(
        default_factory=dict
    )


@dataclass
class UsageRecord:
    """Token usage from a single LLM call."""

    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class UsageTracker:
    """Accumulates token usage across multiple LLM calls.

    Intended to be set as a context variable around a run
    (simulation, evaluation, scenario build) so that every LLM call
    within that scope automatically records its token usage.

    Example::

        with usage_scope() as tracker:
            with usage_label(component="conversation", conversation_id="c1"):
                result = run_simulation(...)

        print(tracker.total_input_tokens, tracker.total_output_tokens)
        print(tracker.summary_by("conversation_id"))
    """

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
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
        labels: dict[str, str] | None = None,
    ) -> None:
        with self._lock:
            self.records.append(
                UsageRecord(
                    model=model,
                    provider=provider,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=cached_tokens,
                    reasoning_tokens=reasoning_tokens,
                    labels=dict(labels) if labels else {},
                )
            )

    @property
    def total_input_tokens(self) -> int:
        with self._lock:
            records = list(self.records)
        return sum(r.input_tokens for r in records)

    @property
    def total_output_tokens(self) -> int:
        with self._lock:
            records = list(self.records)
        return sum(r.output_tokens for r in records)

    @property
    def total_cached_tokens(self) -> int:
        with self._lock:
            records = list(self.records)
        return sum(r.cached_tokens for r in records)

    @property
    def total_reasoning_tokens(self) -> int:
        with self._lock:
            records = list(self.records)
        return sum(r.reasoning_tokens for r in records)

    def summary(self) -> dict[str, dict[str, int]]:
        """Per-model breakdown keyed by ``provider/model``."""
        with self._lock:
            records = list(self.records)
        breakdown: dict[str, dict[str, int]] = {}
        for r in records:
            key = f"{r.provider}/{r.model}"
            if key not in breakdown:
                breakdown[key] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_tokens": 0,
                    "reasoning_tokens": 0,
                }
            breakdown[key]["input_tokens"] += r.input_tokens
            breakdown[key]["output_tokens"] += r.output_tokens
            breakdown[key]["cached_tokens"] += r.cached_tokens
            breakdown[key]["reasoning_tokens"] += r.reasoning_tokens
        return breakdown

    def summary_by(
        self,
        *keys: str,
        where: dict[str, str] | None = None,
    ) -> list[dict[str, str | int]]:
        """Group records by the given label keys.

        Records that do not carry all requested keys are skipped, so callers
        can ask for ``summary_by("conversation_id", "turn_id")`` without
        polluting it with auxiliary calls (e.g., multi-knowledge prep) that
        live outside the conversation/turn scope.

        Pass ``where`` to keep only records whose labels match every key/value
        pair (e.g., ``where={"component": "conversation"}``).
        """
        with self._lock:
            records = list(self.records)
        grouped: dict[tuple[str, ...], dict[str, int]] = {}
        for r in records:
            if where and any(r.labels.get(k) != v for k, v in where.items()):
                continue
            if not all(k in r.labels for k in keys):
                continue
            values = tuple(r.labels[k] for k in keys)
            bucket = grouped.setdefault(
                values,
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_tokens": 0,
                    "reasoning_tokens": 0,
                    "calls": 0,
                },
            )
            bucket["input_tokens"] += r.input_tokens
            bucket["output_tokens"] += r.output_tokens
            bucket["cached_tokens"] += r.cached_tokens
            bucket["reasoning_tokens"] += r.reasoning_tokens
            bucket["calls"] += 1
        rows: list[dict[str, str | int]] = []
        for values, counts in grouped.items():
            row: dict[str, str | int] = dict(zip(keys, values, strict=True))
            row.update(counts)
            rows.append(row)
        return rows

    def log_summary(self) -> None:
        """Log a human-readable token usage summary."""
        if not self.records:
            return
        parts = []
        for model, counts in self.summary().items():
            parts.append(
                f"{model}: {counts['input_tokens']}in"
                f"({counts['cached_tokens']} cached)/"
                f"{counts['output_tokens']}out"
                f"({counts['reasoning_tokens']} reasoning)"
            )
        logger.info(
            f"Token usage - "
            f"input: {self.total_input_tokens:,} "
            f"(cached: {self.total_cached_tokens:,})  "
            f"output: {self.total_output_tokens:,} "
            f"(reasoning: {self.total_reasoning_tokens:,})  "
            f"({', '.join(parts)})"
        )


_current_tracker: ContextVar[UsageTracker | None] = ContextVar(
    "llm_usage_tracker", default=None
)

_current_labels: ContextVar[tuple[dict[str, str], ...]] = ContextVar(
    "llm_usage_labels", default=()
)


def set_current_tracker(tracker: UsageTracker) -> Token[UsageTracker | None]:
    """Set the current tracker. Returns a token for reset_current_tracker."""
    return _current_tracker.set(tracker)


def reset_current_tracker(token: Token[UsageTracker | None]) -> None:
    """Reset the tracker context variable using the token from set_current_tracker."""
    _current_tracker.reset(token)


@contextmanager
def usage_scope() -> Iterator[UsageTracker]:
    """Activate a fresh UsageTracker for the duration of the block."""
    tracker = UsageTracker()
    token = set_current_tracker(tracker)
    try:
        yield tracker
    finally:
        reset_current_tracker(token)


@contextmanager
def usage_label(**labels: str) -> Iterator[None]:
    """Push a layer of labels onto the active label stack.

    Labels are merged with outer layers; inner layers override outer ones for
    the same key. Every ``track_usage`` call inside the block records the
    flattened labels on the resulting :class:`UsageRecord`.

    Values are coerced to ``str`` so JSON serialization stays straightforward.
    """
    layer = {k: str(v) for k, v in labels.items()}
    stack = _current_labels.get()
    token = _current_labels.set(stack + (layer,))
    try:
        yield
    finally:
        _current_labels.reset(token)


def _flatten_labels() -> dict[str, str]:
    merged: dict[str, str] = {}
    for layer in _current_labels.get():
        merged.update(layer)
    return merged


def track_usage(
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> None:
    """Record token usage on the current tracker, if one is active.

    Callers are expected to pass clean non-negative ints; provider
    adapters normalize None / negative values before calling this.
    """
    tracker = _current_tracker.get()
    if tracker is not None:
        tracker.record(
            model,
            provider,
            input_tokens,
            output_tokens,
            cached_tokens,
            reasoning_tokens,
            labels=_flatten_labels(),
        )
