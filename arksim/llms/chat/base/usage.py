# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, replace

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
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_tokens: int = 0
    by_model: dict[str, dict[str, int]] = PydanticField(default_factory=dict)
    breakdowns: dict[str, list[dict[str, str | int]]] = PydanticField(
        default_factory=dict
    )


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
    input_token_details: dict[str, int] = field(default_factory=dict)
    output_token_details: dict[str, int] = field(default_factory=dict)

    module: str | None = None
    run_id: str | None = None
    component: str | None = None

    step: str | None = None
    conversation_id: str | None = None
    turn_id: int | None = None


@dataclass
class UsageTracker:
    """Accumulates token usage across multiple LLM calls."""

    module: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
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
        input_token_details: dict[str, int] | None = None,
        output_token_details: dict[str, int] | None = None,
        component: str | None = None,
        step: str | None = None,
        conversation_id: str | None = None,
        turn_id: int | None = None,
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
                    input_token_details=dict(input_token_details or {}),
                    output_token_details=dict(output_token_details or {}),
                    module=self.module,
                    run_id=self.run_id,
                    component=component,
                    step=step,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
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
    def total_cache_read_tokens(self) -> int:
        with self._lock:
            records = list(self.records)
        return sum(r.cache_read_tokens for r in records)

    @property
    def total_cache_creation_tokens(self) -> int:
        with self._lock:
            records = list(self.records)
        return sum(r.cache_creation_tokens for r in records)

    @property
    def total_reasoning_tokens(self) -> int:
        with self._lock:
            records = list(self.records)
        return sum(r.reasoning_tokens for r in records)

    @property
    def total_tokens(self) -> int:
        with self._lock:
            records = list(self.records)
        return sum(r.total_tokens for r in records)

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
                    "cache_read_tokens": 0,
                    "cache_creation_tokens": 0,
                    "reasoning_tokens": 0,
                    "total_tokens": 0,
                }
            breakdown[key]["input_tokens"] += r.input_tokens
            breakdown[key]["output_tokens"] += r.output_tokens
            breakdown[key]["cache_read_tokens"] += r.cache_read_tokens
            breakdown[key]["cache_creation_tokens"] += r.cache_creation_tokens
            breakdown[key]["reasoning_tokens"] += r.reasoning_tokens
            breakdown[key]["total_tokens"] += r.total_tokens
        return breakdown

    def summary_by(
        self,
        *keys: str,
        where: dict[str, object] | None = None,
    ) -> list[dict[str, str | int]]:
        """Group records by the given record attributes."""
        with self._lock:
            records = list(self.records)
        grouped: dict[tuple[object, ...], dict[str, int]] = {}
        for r in records:
            if where and any(getattr(r, k, None) != v for k, v in where.items()):
                continue
            values = tuple(getattr(r, k, None) for k in keys)
            if any(v is None for v in values):
                continue
            bucket = grouped.setdefault(
                values,
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_creation_tokens": 0,
                    "reasoning_tokens": 0,
                    "total_tokens": 0,
                    "calls": 0,
                },
            )
            bucket["input_tokens"] += r.input_tokens
            bucket["output_tokens"] += r.output_tokens
            bucket["cache_read_tokens"] += r.cache_read_tokens
            bucket["cache_creation_tokens"] += r.cache_creation_tokens
            bucket["reasoning_tokens"] += r.reasoning_tokens
            bucket["total_tokens"] += r.total_tokens
            bucket["calls"] += 1
        rows: list[dict[str, str | int]] = []
        for values, counts in grouped.items():
            row: dict[str, str | int] = dict(zip(keys, values, strict=True))  # type: ignore[arg-type]
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
                f"({counts['cache_read_tokens']} cache_read,"
                f"{counts['cache_creation_tokens']} cache_create)/"
                f"{counts['output_tokens']}out"
                f"({counts['reasoning_tokens']} reasoning)"
            )
        logger.info(
            f"Token usage [{self.module} run={self.run_id}] - "
            f"input: {self.total_input_tokens:,} "
            f"(cache_read: {self.total_cache_read_tokens:,}, "
            f"cache_create: {self.total_cache_creation_tokens:,})  "
            f"output: {self.total_output_tokens:,} "
            f"(reasoning: {self.total_reasoning_tokens:,})  "
            f"total: {self.total_tokens:,}  "
            f"({', '.join(parts)})"
        )


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
def usage_run(*, module: str, run_id: str | None = None) -> Iterator[UsageTracker]:
    """Activate a fresh :class:`UsageTracker` for the duration of the block.

    Args:
        module: Required identifier for the kind of work being run
            (e.g. ``"simulation"``, ``"evaluation"``, ``"scenario_generation"``).
        run_id: Optional caller-supplied run identifier. A UUID4 hex is
            generated when not given. Available on the yielded tracker as
            ``tracker.run_id``.

    The run also resets the per-record tag context (component, step,
    conversation_id, turn_id) so a nested run does not inherit tags
    from an outer one.
    """
    if not module:
        raise ValueError("usage_run requires a non-empty module")
    tracker = UsageTracker(module=module, run_id=run_id or uuid.uuid4().hex)
    tracker_token = set_current_tracker(tracker)
    tags_token = _current_tags.set(_EMPTY_TAGS)
    try:
        yield tracker
    finally:
        _current_tags.reset(tags_token)
        reset_current_tracker(tracker_token)


@contextmanager
def usage_tags(
    *,
    component: str | None = None,
    step: str | None = None,
    conversation_id: str | None = None,
    turn_id: int | None = None,
) -> Iterator[None]:
    """Push a tag layer onto the active run context."""
    overrides = {
        k: v
        for k, v in (
            ("component", component),
            ("step", step),
            ("conversation_id", conversation_id),
            ("turn_id", turn_id),
        )
        if v is not None
    }
    token = _current_tags.set(replace(_current_tags.get(), **overrides))
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
    input_token_details: dict[str, int] | None = None,
    output_token_details: dict[str, int] | None = None,
) -> None:
    """Record token usage on the current tracker, if one is active.

    Callers are expected to pass clean non-negative ints; provider
    adapters normalize None / negative values before calling this.
    """
    tracker = _current_tracker.get()
    if tracker is None:
        return
    tags = _current_tags.get()
    tracker.record(
        model,
        provider,
        input_tokens,
        output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
        input_token_details=input_token_details,
        output_token_details=output_token_details,
        component=tags.component,
        step=tags.step,
        conversation_id=tags.conversation_id,
        turn_id=tags.turn_id,
    )
