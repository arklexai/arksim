# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry", reason="opentelemetry SDK not installed")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind

from arksim.telemetry.instrumentation import (
    agent_execute_span,
    conversation_span,
    eval_conversation_span,
    evaluation_span,
    record_evaluation_result,
    simulation_span,
    turn_span,
)


@pytest.fixture(autouse=True)
def _setup_in_memory_exporter() -> InMemorySpanExporter:
    """Configure an in-memory span exporter for all tests in this module."""
    import opentelemetry.trace as trace_api
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    # Reset metrics singleton so each test starts fresh
    import arksim.telemetry.instrumentation as instr_mod

    instr_mod._simulation_duration = None

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    trace_api.set_tracer_provider(provider)

    yield exporter
    provider.shutdown()
    trace_api._TRACER_PROVIDER = None  # noqa: SLF001
    trace_api._TRACER_PROVIDER_SET_ONCE._done = False  # noqa: SLF001


def _get_spans(exporter: InMemorySpanExporter) -> list:
    return list(exporter.get_finished_spans())


# ---------------------------------------------------------------------------
# Simulation spans
# ---------------------------------------------------------------------------


class TestSimulationSpan:
    @pytest.mark.asyncio
    async def test_creates_span_with_attributes(
        self, _setup_in_memory_exporter: InMemorySpanExporter
    ) -> None:
        async with simulation_span(
            simulation_id="sim-123",
            num_scenarios=2,
            num_conversations=10,
            model="gpt-4",
            provider="openai",
        ):
            pass

        spans = _get_spans(_setup_in_memory_exporter)
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "arksim.simulation"
        attrs = dict(span.attributes)
        # arksim domain attrs
        assert attrs["arksim.simulation.id"] == "sim-123"
        assert attrs["arksim.simulation.num_scenarios"] == 2
        assert attrs["arksim.simulation.num_conversations"] == 10
        assert attrs["arksim.simulation.model"] == "gpt-4"
        assert attrs["arksim.simulation.provider"] == "openai"
        # GenAI semconv attrs
        assert attrs["gen_ai.request.model"] == "gpt-4"
        assert attrs["gen_ai.provider.name"] == "openai"

    @pytest.mark.asyncio
    async def test_records_error_with_error_type(
        self, _setup_in_memory_exporter: InMemorySpanExporter
    ) -> None:
        with pytest.raises(ValueError, match="boom"):
            async with simulation_span(
                simulation_id="sim-err",
                num_scenarios=1,
                num_conversations=1,
                model="m",
                provider="p",
            ):
                raise ValueError("boom")

        spans = _get_spans(_setup_in_memory_exporter)
        assert len(spans) == 1
        assert spans[0].status.status_code.name == "ERROR"
        assert spans[0].attributes["error.type"] == "ValueError"


class TestConversationSpan:
    @pytest.mark.asyncio
    async def test_creates_span_with_attributes(
        self, _setup_in_memory_exporter: InMemorySpanExporter
    ) -> None:
        async with conversation_span(
            conversation_id="conv-1",
            scenario_id="sc-1",
            goal="Buy a product",
        ):
            pass

        spans = _get_spans(_setup_in_memory_exporter)
        assert len(spans) == 1
        assert spans[0].name == "arksim.simulation.conversation"
        attrs = dict(spans[0].attributes)
        assert attrs["arksim.conversation.id"] == "conv-1"
        assert attrs["arksim.scenario.id"] == "sc-1"
        assert attrs["arksim.scenario.goal"] == "Buy a product"


class TestTurnSpan:
    @pytest.mark.asyncio
    async def test_creates_span_with_turn_id(
        self, _setup_in_memory_exporter: InMemorySpanExporter
    ) -> None:
        async with turn_span(turn_id=3):
            pass

        spans = _get_spans(_setup_in_memory_exporter)
        assert len(spans) == 1
        assert spans[0].name == "arksim.simulation.turn"
        assert spans[0].attributes["arksim.turn.id"] == 3


class TestAgentExecuteSpan:
    @pytest.mark.asyncio
    async def test_creates_span_with_genai_attributes(
        self, _setup_in_memory_exporter: InMemorySpanExporter
    ) -> None:
        async with agent_execute_span(
            agent_name="my-agent",
            agent_type="chat_completions",
            model="gpt-4",
            provider="openai",
        ):
            pass

        spans = _get_spans(_setup_in_memory_exporter)
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "arksim.agent.execute"
        attrs = dict(span.attributes)
        # arksim domain attrs
        assert attrs["arksim.agent.name"] == "my-agent"
        assert attrs["arksim.agent.type"] == "chat_completions"
        # GenAI semconv attrs
        assert attrs["gen_ai.operation.name"] == "chat"
        assert attrs["gen_ai.agent.name"] == "my-agent"
        assert attrs["gen_ai.request.model"] == "gpt-4"
        assert attrs["gen_ai.provider.name"] == "openai"

    @pytest.mark.asyncio
    async def test_span_kind_is_client(
        self, _setup_in_memory_exporter: InMemorySpanExporter
    ) -> None:
        async with agent_execute_span(agent_name="a", agent_type="t"):
            pass

        spans = _get_spans(_setup_in_memory_exporter)
        assert spans[0].kind == SpanKind.CLIENT


class TestSpanKind:
    @pytest.mark.asyncio
    async def test_simulation_span_is_internal(
        self, _setup_in_memory_exporter: InMemorySpanExporter
    ) -> None:
        async with simulation_span(
            simulation_id="s",
            num_scenarios=1,
            num_conversations=1,
            model="m",
            provider="p",
        ):
            pass

        spans = _get_spans(_setup_in_memory_exporter)
        assert spans[0].kind == SpanKind.INTERNAL

    def test_evaluation_span_is_internal(
        self, _setup_in_memory_exporter: InMemorySpanExporter
    ) -> None:
        with evaluation_span(
            evaluation_id="e",
            simulation_id="s",
            num_conversations=1,
        ):
            pass

        spans = _get_spans(_setup_in_memory_exporter)
        assert spans[0].kind == SpanKind.INTERNAL


class TestParentChildRelationships:
    @pytest.mark.asyncio
    async def test_nested_spans_have_parent(
        self, _setup_in_memory_exporter: InMemorySpanExporter
    ) -> None:
        async with (
            simulation_span(
                simulation_id="s1",
                num_scenarios=1,
                num_conversations=1,
                model="m",
                provider="p",
            ),
            conversation_span(
                conversation_id="c1",
                scenario_id="sc1",
                goal="g",
            ),
            turn_span(turn_id=0),
        ):
            pass

        spans = _get_spans(_setup_in_memory_exporter)
        assert len(spans) == 3

        by_name = {s.name: s for s in spans}
        turn = by_name["arksim.simulation.turn"]
        conv = by_name["arksim.simulation.conversation"]
        sim = by_name["arksim.simulation"]

        assert turn.parent.span_id == conv.context.span_id
        assert conv.parent.span_id == sim.context.span_id
        assert sim.parent is None


# ---------------------------------------------------------------------------
# Evaluation spans
# ---------------------------------------------------------------------------


class TestEvaluationSpan:
    def test_creates_span_with_attributes(
        self, _setup_in_memory_exporter: InMemorySpanExporter
    ) -> None:
        with evaluation_span(
            evaluation_id="eval-1",
            simulation_id="sim-1",
            num_conversations=5,
        ):
            pass

        spans = _get_spans(_setup_in_memory_exporter)
        assert len(spans) == 1
        assert spans[0].name == "arksim.evaluation"
        attrs = dict(spans[0].attributes)
        assert attrs["arksim.evaluation.id"] == "eval-1"
        assert attrs["arksim.evaluation.simulation_id"] == "sim-1"
        assert attrs["arksim.evaluation.num_conversations"] == 5


class TestEvalConversationSpan:
    def test_creates_span_with_conversation_id(
        self, _setup_in_memory_exporter: InMemorySpanExporter
    ) -> None:
        with eval_conversation_span(conversation_id="conv-42"):
            pass

        spans = _get_spans(_setup_in_memory_exporter)
        assert len(spans) == 1
        assert spans[0].name == "arksim.evaluation.conversation"
        assert spans[0].attributes["arksim.conversation.id"] == "conv-42"


# ---------------------------------------------------------------------------
# Evaluation result events
# ---------------------------------------------------------------------------


class TestEvaluationResultEvents:
    def test_record_evaluation_result_adds_event(
        self, _setup_in_memory_exporter: InMemorySpanExporter
    ) -> None:
        with eval_conversation_span(conversation_id="conv-1") as span:
            record_evaluation_result(
                span,
                metric_name="faithfulness",
                score_value=4.5,
                score_label="good",
            )
            record_evaluation_result(
                span,
                metric_name="goal_completion",
                score_value=0.9,
            )

        spans = _get_spans(_setup_in_memory_exporter)
        assert len(spans) == 1
        events = spans[0].events
        assert len(events) == 2

        e0 = events[0]
        assert e0.name == "gen_ai.evaluation.result"
        assert e0.attributes["gen_ai.evaluation.name"] == "faithfulness"
        assert e0.attributes["gen_ai.evaluation.score.value"] == 4.5
        assert e0.attributes["gen_ai.evaluation.score.label"] == "good"

        e1 = events[1]
        assert e1.name == "gen_ai.evaluation.result"
        assert e1.attributes["gen_ai.evaluation.name"] == "goal_completion"
        assert e1.attributes["gen_ai.evaluation.score.value"] == 0.9
        assert "gen_ai.evaluation.score.label" not in e1.attributes
