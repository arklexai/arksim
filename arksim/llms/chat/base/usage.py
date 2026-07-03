# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class UsageRecord:
    """Token usage from a single LLM call."""

    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    component: str | None = None


@dataclass
class UsageTracker:
    """Accumulates token usage across multiple LLM calls within a run."""

    records: list[UsageRecord] = field(default_factory=list)

    def record(
        self,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        component: str | None = None,
    ) -> None:
        self.records.append(
            UsageRecord(
                model=model,
                provider=provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                component=component,
            )
        )

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)


_current_tracker: ContextVar[UsageTracker | None] = ContextVar(
    "llm_usage_tracker", default=None
)
_current_component: ContextVar[str | None] = ContextVar(
    "llm_usage_component", default=None
)


def get_current_tracker() -> UsageTracker | None:
    return _current_tracker.get()


def set_current_tracker(tracker: UsageTracker) -> object:
    return _current_tracker.set(tracker)


def reset_current_tracker(token: object) -> None:
    _current_tracker.reset(token)  # type: ignore[arg-type]


def track_usage(
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    component: str | None = None,
) -> None:
    """Record token usage on the active tracker, if one is set."""
    tracker = _current_tracker.get()
    if tracker is not None:
        effective_component = component or _current_component.get()
        tracker.record(
            model, provider, input_tokens, output_tokens, effective_component
        )


@contextmanager
def usage_tags(*, component: str | None = None) -> Iterator[None]:
    """Tag LLM calls within this block with a component label."""
    token = _current_component.set(component)
    try:
        yield
    finally:
        _current_component.reset(token)


@contextmanager
def usage_run() -> Iterator[UsageTracker]:
    """Context manager that activates a fresh UsageTracker for the duration."""
    tracker = UsageTracker()
    token = set_current_tracker(tracker)
    try:
        yield tracker
    finally:
        reset_current_tracker(token)
