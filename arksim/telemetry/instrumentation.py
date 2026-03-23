# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from .provider import get_meter, get_tracer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy SpanKind helper (avoids top-level OTel import)
# ---------------------------------------------------------------------------


def _span_kind(kind_name: str) -> object:
    """Return an OTel SpanKind constant, or None if SDK is absent."""
    try:
        from opentelemetry.trace import SpanKind

        return getattr(SpanKind, kind_name)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Metrics (lazily created on first use)
# ---------------------------------------------------------------------------

_simulation_duration = None
_conversation_duration = None
_turn_duration = None
_agent_execute_duration = None
_evaluation_duration = None
_conversations_counter = None
_turns_counter = None
_errors_counter = None


def _ensure_metrics() -> None:
    """Create metric instruments on first call."""
    global _simulation_duration, _conversation_duration, _turn_duration  # noqa: PLW0603
    global _agent_execute_duration, _evaluation_duration  # noqa: PLW0603
    global _conversations_counter, _turns_counter, _errors_counter  # noqa: PLW0603

    if _simulation_duration is not None:
        return

    meter = get_meter()

    _simulation_duration = meter.create_histogram(
        name="arksim.simulation.duration",
        description="Duration of a full simulation run",
        unit="s",
    )
    _conversation_duration = meter.create_histogram(
        name="arksim.simulation.conversation.duration",
        description="Duration of a single simulated conversation",
        unit="s",
    )
    _turn_duration = meter.create_histogram(
        name="arksim.simulation.turn.duration",
        description="Duration of a single conversation turn",
        unit="s",
    )
    _agent_execute_duration = meter.create_histogram(
        name="arksim.agent.execute.duration",
        description="Duration of an agent.execute() call",
        unit="s",
    )
    _evaluation_duration = meter.create_histogram(
        name="arksim.evaluation.duration",
        description="Duration of a full evaluation run",
        unit="s",
    )
    _conversations_counter = meter.create_counter(
        name="arksim.simulation.conversations.total",
        description="Total number of simulated conversations completed",
    )
    _turns_counter = meter.create_counter(
        name="arksim.simulation.turns.total",
        description="Total number of simulation turns completed",
    )
    _errors_counter = meter.create_counter(
        name="arksim.errors.total",
        description="Total number of errors across simulation and evaluation",
    )


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------


def _set_error(span: object, exc: BaseException) -> None:
    """Record an exception on a span, setting its status to ERROR."""
    try:
        from opentelemetry.trace import StatusCode

        span.set_status(StatusCode.ERROR, str(exc))
    except ImportError:
        span.set_status("ERROR", str(exc))

    span.record_exception(exc)
    span.set_attribute("error.type", type(exc).__qualname__)

    _ensure_metrics()
    if _errors_counter is not None:
        _errors_counter.add(1, {"error.type": type(exc).__qualname__})


# ---------------------------------------------------------------------------
# Simulation spans
# ---------------------------------------------------------------------------


@asynccontextmanager
async def simulation_span(
    *,
    simulation_id: str,
    num_scenarios: int,
    num_conversations: int,
    model: str,
    provider: str,
) -> AsyncIterator[Any]:
    """Root span for an entire simulation run."""
    _ensure_metrics()
    tracer = get_tracer()
    kind = _span_kind("INTERNAL")
    kwargs: dict[str, Any] = {}
    if kind is not None:
        kwargs["kind"] = kind

    with tracer.start_as_current_span("arksim.simulation", **kwargs) as span:
        span.set_attribute("arksim.simulation.id", simulation_id)
        span.set_attribute("arksim.simulation.num_scenarios", num_scenarios)
        span.set_attribute("arksim.simulation.num_conversations", num_conversations)
        span.set_attribute("arksim.simulation.model", model)
        span.set_attribute("arksim.simulation.provider", provider)
        # GenAI semconv attributes
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.provider.name", provider)
        t0 = time.monotonic()
        try:
            yield span
        except Exception as exc:
            _set_error(span, exc)
            raise
        finally:
            elapsed = time.monotonic() - t0
            if _simulation_duration is not None:
                _simulation_duration.record(
                    elapsed,
                    {"gen_ai.request.model": model, "gen_ai.provider.name": provider},
                )


@asynccontextmanager
async def conversation_span(
    *,
    conversation_id: str,
    scenario_id: str,
    goal: str,
) -> AsyncIterator[Any]:
    """Span for a single simulated conversation."""
    _ensure_metrics()
    tracer = get_tracer()
    kind = _span_kind("INTERNAL")
    kwargs: dict[str, Any] = {}
    if kind is not None:
        kwargs["kind"] = kind

    with tracer.start_as_current_span(
        "arksim.simulation.conversation", **kwargs
    ) as span:
        span.set_attribute("arksim.conversation.id", conversation_id)
        span.set_attribute("arksim.scenario.id", scenario_id)
        span.set_attribute("arksim.scenario.goal", goal[:500] if goal else "")
        t0 = time.monotonic()
        try:
            yield span
        except Exception as exc:
            _set_error(span, exc)
            raise
        finally:
            elapsed = time.monotonic() - t0
            if _conversation_duration is not None:
                _conversation_duration.record(
                    elapsed, {"arksim.scenario.id": scenario_id}
                )
            if _conversations_counter is not None:
                _conversations_counter.add(1, {"arksim.scenario.id": scenario_id})


@asynccontextmanager
async def turn_span(*, turn_id: int) -> AsyncIterator[Any]:
    """Span for a single conversation turn."""
    _ensure_metrics()
    tracer = get_tracer()
    kind = _span_kind("INTERNAL")
    kwargs: dict[str, Any] = {}
    if kind is not None:
        kwargs["kind"] = kind

    with tracer.start_as_current_span("arksim.simulation.turn", **kwargs) as span:
        span.set_attribute("arksim.turn.id", turn_id)
        t0 = time.monotonic()
        try:
            yield span
        except Exception as exc:
            _set_error(span, exc)
            raise
        finally:
            elapsed = time.monotonic() - t0
            if _turn_duration is not None:
                _turn_duration.record(elapsed)
            if _turns_counter is not None:
                _turns_counter.add(1)


@asynccontextmanager
async def agent_execute_span(
    *,
    agent_name: str,
    agent_type: str,
    model: str = "",
    provider: str = "",
) -> AsyncIterator[Any]:
    """Span for an agent.execute() call.

    Uses SpanKind.CLIENT per OTel GenAI conventions (calling an external
    AI service).
    """
    _ensure_metrics()
    tracer = get_tracer()
    kind = _span_kind("CLIENT")
    kwargs: dict[str, Any] = {}
    if kind is not None:
        kwargs["kind"] = kind

    with tracer.start_as_current_span("arksim.agent.execute", **kwargs) as span:
        # arksim domain attributes
        span.set_attribute("arksim.agent.name", agent_name)
        span.set_attribute("arksim.agent.type", agent_type)
        # OTel GenAI semantic convention attributes
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.agent.name", agent_name)
        if model:
            span.set_attribute("gen_ai.request.model", model)
        if provider:
            span.set_attribute("gen_ai.provider.name", provider)
        t0 = time.monotonic()
        try:
            yield span
        except Exception as exc:
            _set_error(span, exc)
            raise
        finally:
            elapsed = time.monotonic() - t0
            if _agent_execute_duration is not None:
                _agent_execute_duration.record(
                    elapsed,
                    {
                        "arksim.agent.name": agent_name,
                        "arksim.agent.type": agent_type,
                        **({"gen_ai.request.model": model} if model else {}),
                    },
                )


# ---------------------------------------------------------------------------
# Evaluation spans
# ---------------------------------------------------------------------------


@contextmanager
def evaluation_span(
    *,
    evaluation_id: str,
    simulation_id: str,
    num_conversations: int,
) -> Iterator[Any]:
    """Root span for an evaluation run."""
    _ensure_metrics()
    tracer = get_tracer()
    kind = _span_kind("INTERNAL")
    kwargs: dict[str, Any] = {}
    if kind is not None:
        kwargs["kind"] = kind

    with tracer.start_as_current_span("arksim.evaluation", **kwargs) as span:
        span.set_attribute("arksim.evaluation.id", evaluation_id)
        span.set_attribute("arksim.evaluation.simulation_id", simulation_id)
        span.set_attribute("arksim.evaluation.num_conversations", num_conversations)
        t0 = time.monotonic()
        try:
            yield span
        except Exception as exc:
            _set_error(span, exc)
            raise
        finally:
            elapsed = time.monotonic() - t0
            if _evaluation_duration is not None:
                _evaluation_duration.record(elapsed)


@contextmanager
def eval_conversation_span(
    *,
    conversation_id: str,
) -> Iterator[Any]:
    """Span for evaluating a single conversation."""
    tracer = get_tracer()
    kind = _span_kind("INTERNAL")
    kwargs: dict[str, Any] = {}
    if kind is not None:
        kwargs["kind"] = kind

    with tracer.start_as_current_span(
        "arksim.evaluation.conversation", **kwargs
    ) as span:
        span.set_attribute("arksim.conversation.id", conversation_id)
        try:
            yield span
        except Exception as exc:
            _set_error(span, exc)
            raise


# ---------------------------------------------------------------------------
# Evaluation result events (gen_ai.evaluation.result semconv)
# ---------------------------------------------------------------------------


def record_evaluation_result(
    span: object,
    *,
    metric_name: str,
    score_value: float,
    score_label: str = "",
) -> None:
    """Record a gen_ai.evaluation.result event on the given span.

    Follows the OTel GenAI semantic convention for evaluation results.
    """
    attrs: dict[str, Any] = {
        "gen_ai.evaluation.name": metric_name,
        "gen_ai.evaluation.score.value": score_value,
    }
    if score_label:
        attrs["gen_ai.evaluation.score.label"] = score_label
    span.add_event("gen_ai.evaluation.result", attributes=attrs)
