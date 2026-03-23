# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from arksim.telemetry.instrumentation import (
    agent_execute_span,
    conversation_span,
    eval_conversation_span,
    evaluation_span,
    simulation_span,
    turn_span,
)


@pytest.fixture(autouse=True)
def _setup_in_memory_exporter() -> InMemorySpanExporter:
    """Configure an in-memory span exporter for all tests in this module."""
    import opentelemetry.trace as trace_api
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Reset global tracer provider so each test gets a fresh one.
    # The SDK prevents overriding via set_tracer_provider after the first
    # call, so we directly replace the internal proxy's underlying provider.
    trace_api.set_tracer_provider(provider)

    yield exporter
    provider.shutdown()
    # Reset the global state so the next test can set a new provider.
    # This reaches into the proxy tracer provider used by the OTel API.
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
        assert attrs["arksim.simulation.id"] == "sim-123"
        assert attrs["arksim.simulation.num_scenarios"] == 2
        assert attrs["arksim.simulation.num_conversations"] == 10
        assert attrs["arksim.simulation.model"] == "gpt-4"
        assert attrs["arksim.simulation.provider"] == "openai"

    @pytest.mark.asyncio
    async def test_records_error_on_exception(
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
    async def test_creates_span_with_agent_info(
        self, _setup_in_memory_exporter: InMemorySpanExporter
    ) -> None:
        async with agent_execute_span(
            agent_name="my-agent",
            agent_type="chat_completions",
        ):
            pass

        spans = _get_spans(_setup_in_memory_exporter)
        assert len(spans) == 1
        assert spans[0].name == "arksim.agent.execute"
        attrs = dict(spans[0].attributes)
        assert attrs["arksim.agent.name"] == "my-agent"
        assert attrs["arksim.agent.type"] == "chat_completions"


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

        # turn's parent should be conversation
        assert turn.parent.span_id == conv.context.span_id
        # conversation's parent should be simulation
        assert conv.parent.span_id == sim.context.span_id
        # simulation should be root
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
