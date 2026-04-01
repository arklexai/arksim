# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
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


@dataclass
class UsageTracker:
    """Accumulates token usage across multiple LLM calls.

    Intended to be set as a context variable around a run
    (simulation, evaluation, scenario build) so that every LLM call
    within that scope automatically records its token usage.

    Example::

        tracker = UsageTracker()
        token = set_current_tracker(tracker)
        try:
            result = run_simulation(...)
        finally:
            reset_current_tracker(token)

        print(tracker.total_input_tokens, tracker.total_output_tokens)
    """

    records: list[UsageRecord] = field(default_factory=list)

    def record(
        self,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        self.records.append(
            UsageRecord(
                model=model,
                provider=provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)

    def summary(self) -> dict[str, dict[str, int]]:
        """Per-model breakdown: {model: {input_tokens, output_tokens}}."""
        breakdown: dict[str, dict[str, int]] = {}
        for r in self.records:
            if r.model not in breakdown:
                breakdown[r.model] = {"input_tokens": 0, "output_tokens": 0}
            breakdown[r.model]["input_tokens"] += r.input_tokens
            breakdown[r.model]["output_tokens"] += r.output_tokens
        return breakdown

    def log_summary(self) -> None:
        """Log a human-readable token usage summary."""
        if not self.records:
            return
        parts = []
        for model, counts in self.summary().items():
            parts.append(
                f"{model}: {counts['input_tokens']}in/{counts['output_tokens']}out"
            )
        logger.info(
            f"Token usage - "
            f"input: {self.total_input_tokens:,}  "
            f"output: {self.total_output_tokens:,}  "
            f"({', '.join(parts)})"
        )


_current_tracker: ContextVar[UsageTracker | None] = ContextVar(
    "llm_usage_tracker", default=None
)


def get_current_tracker() -> UsageTracker | None:
    return _current_tracker.get()


def set_current_tracker(tracker: UsageTracker) -> object:
    """Set the current tracker. Returns a token for reset_current_tracker."""
    return _current_tracker.set(tracker)


def reset_current_tracker(token: object) -> None:
    """Reset the tracker context variable using the token from set_current_tracker."""
    _current_tracker.reset(token)  # type: ignore[arg-type]


def track_usage(
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Record token usage on the current tracker, if one is active."""
    tracker = _current_tracker.get()
    if tracker is not None:
        tracker.record(model, provider, input_tokens, output_tokens)
