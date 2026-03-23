# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from .provider import get_tracer

logger = logging.getLogger(__name__)


def _set_error(span: object, exc: BaseException) -> None:
    """Record an exception on a span, setting its status to ERROR."""
    try:
        from opentelemetry.trace import StatusCode

        span.set_status(StatusCode.ERROR, str(exc))
        span.record_exception(exc)
    except ImportError:
        # OTel SDK not installed; span is a no-op anyway.
        span.set_status("ERROR", str(exc))
        span.record_exception(exc)


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
    tracer = get_tracer()
    with tracer.start_as_current_span("arksim.simulation") as span:
        span.set_attribute("arksim.simulation.id", simulation_id)
        span.set_attribute("arksim.simulation.num_scenarios", num_scenarios)
        span.set_attribute("arksim.simulation.num_conversations", num_conversations)
        span.set_attribute("arksim.simulation.model", model)
        span.set_attribute("arksim.simulation.provider", provider)
        try:
            yield span
        except Exception as exc:
            _set_error(span, exc)
            raise


@asynccontextmanager
async def conversation_span(
    *,
    conversation_id: str,
    scenario_id: str,
    goal: str,
) -> AsyncIterator[Any]:
    """Span for a single simulated conversation."""
    tracer = get_tracer()
    with tracer.start_as_current_span("arksim.simulation.conversation") as span:
        span.set_attribute("arksim.conversation.id", conversation_id)
        span.set_attribute("arksim.scenario.id", scenario_id)
        span.set_attribute("arksim.scenario.goal", goal[:500] if goal else "")
        try:
            yield span
        except Exception as exc:
            _set_error(span, exc)
            raise


@asynccontextmanager
async def turn_span(*, turn_id: int) -> AsyncIterator[Any]:
    """Span for a single conversation turn."""
    tracer = get_tracer()
    with tracer.start_as_current_span("arksim.simulation.turn") as span:
        span.set_attribute("arksim.turn.id", turn_id)
        try:
            yield span
        except Exception as exc:
            _set_error(span, exc)
            raise


@asynccontextmanager
async def agent_execute_span(
    *,
    agent_name: str,
    agent_type: str,
) -> AsyncIterator[Any]:
    """Span for an agent.execute() call."""
    tracer = get_tracer()
    with tracer.start_as_current_span("arksim.agent.execute") as span:
        span.set_attribute("arksim.agent.name", agent_name)
        span.set_attribute("arksim.agent.type", agent_type)
        try:
            yield span
        except Exception as exc:
            _set_error(span, exc)
            raise


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
    tracer = get_tracer()
    with tracer.start_as_current_span("arksim.evaluation") as span:
        span.set_attribute("arksim.evaluation.id", evaluation_id)
        span.set_attribute("arksim.evaluation.simulation_id", simulation_id)
        span.set_attribute("arksim.evaluation.num_conversations", num_conversations)
        try:
            yield span
        except Exception as exc:
            _set_error(span, exc)
            raise


@contextmanager
def eval_conversation_span(
    *,
    conversation_id: str,
) -> Iterator[Any]:
    """Span for evaluating a single conversation."""
    tracer = get_tracer()
    with tracer.start_as_current_span("arksim.evaluation.conversation") as span:
        span.set_attribute("arksim.conversation.id", conversation_id)
        try:
            yield span
        except Exception as exc:
            _set_error(span, exc)
            raise
