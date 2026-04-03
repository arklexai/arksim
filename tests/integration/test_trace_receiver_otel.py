# SPDX-License-Identifier: Apache-2.0
"""Integration tests using the real OpenTelemetry SDK.

These tests create spans with the actual ``opentelemetry-sdk`` and export
them via ``OTLPSpanExporter`` (protobuf over HTTP) to our TraceReceiver.
This validates that arksim works with the real OTel wire format that
instrumented agents produce.

Requires: pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
"""

from __future__ import annotations

import socket

import pytest

pytest.importorskip("opentelemetry", reason="opentelemetry SDK not installed")

from opentelemetry import trace  # noqa: E402
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,  # noqa: E402
)
from opentelemetry.sdk.resources import Resource  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402

from arksim.tracing.receiver import TraceReceiver  # noqa: E402


@pytest.fixture
def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _create_otel_provider(port: int, conv_id: str) -> TracerProvider:
    """Create a TracerProvider that exports to our TraceReceiver."""
    resource = Resource.create(
        {
            "service.name": "test-agent",
            "arksim.conversation_id": conv_id,
        }
    )
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(
        endpoint=f"http://127.0.0.1:{port}/v1/traces",
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_otel_single_tool_call(_unused_port: int) -> None:
    """A single tool call span created with the real OTel SDK is captured."""
    port = _unused_port
    async with TraceReceiver(port=port, wait_timeout=1.0) as receiver:
        provider = _create_otel_provider(port, "otel-conv-1")
        tracer = provider.get_tracer("test-agent")

        with tracer.start_as_current_span("execute_tool lookup_customer") as span:
            span.set_attribute("arksim.turn_id", 0)
            span.set_attribute("gen_ai.tool.name", "lookup_customer")
            span.set_attribute(
                "gen_ai.tool.call.arguments", '{"email": "alice@example.com"}'
            )
            span.set_attribute(
                "gen_ai.tool.call.result",
                '{"id": "C-001", "name": "Alice Johnson"}',
            )

        provider.force_flush()
        provider.shutdown()

        tool_calls = await receiver.wait_for_traces("otel-conv-1", 0)

    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc.name == "lookup_customer"
    assert tc.arguments == {"email": "alice@example.com"}
    assert tc.result == '{"id": "C-001", "name": "Alice Johnson"}'
    assert tc.error is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_otel_multiple_tool_calls(_unused_port: int) -> None:
    """Multiple tool call spans in one trace are all captured."""
    port = _unused_port
    async with TraceReceiver(port=port, wait_timeout=1.0) as receiver:
        provider = _create_otel_provider(port, "otel-conv-2")
        tracer = provider.get_tracer("test-agent")

        with tracer.start_as_current_span("execute_tool search_products") as span:
            span.set_attribute("arksim.turn_id", 0)
            span.set_attribute("gen_ai.tool.name", "search_products")
            span.set_attribute(
                "gen_ai.tool.call.arguments", '{"query": "laptop", "max_price": 1000}'
            )

        with tracer.start_as_current_span("execute_tool get_order") as span:
            span.set_attribute("arksim.turn_id", 0)
            span.set_attribute("gen_ai.tool.name", "get_order")
            span.set_attribute("gen_ai.tool.call.arguments", '{"order_id": "ORD-1001"}')

        provider.force_flush()
        provider.shutdown()

        tool_calls = await receiver.wait_for_traces("otel-conv-2", 0)

    assert len(tool_calls) == 2
    names = {tc.name for tc in tool_calls}
    assert names == {"search_products", "get_order"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_otel_error_span(_unused_port: int) -> None:
    """A tool call that errors is captured with the error message."""
    port = _unused_port
    async with TraceReceiver(port=port, wait_timeout=1.0) as receiver:
        provider = _create_otel_provider(port, "otel-conv-3")
        tracer = provider.get_tracer("test-agent")

        with tracer.start_as_current_span("execute_tool cancel_order") as span:
            span.set_attribute("arksim.turn_id", 0)
            span.set_attribute("gen_ai.tool.name", "cancel_order")
            span.set_attribute("gen_ai.tool.call.arguments", '{"order_id": "ORD-9999"}')
            span.set_status(trace.StatusCode.ERROR, "Order not found")

        provider.force_flush()
        provider.shutdown()

        tool_calls = await receiver.wait_for_traces("otel-conv-3", 0)

    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc.name == "cancel_order"
    assert tc.error == "Order not found"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_otel_resource_level_routing(_unused_port: int) -> None:
    """conversation_id set on the OTel Resource (not per-span) still routes correctly."""
    port = _unused_port
    async with TraceReceiver(port=port, wait_timeout=1.0) as receiver:
        # conv_id is on the resource, turn_id on the span
        provider = _create_otel_provider(port, "otel-conv-4")
        tracer = provider.get_tracer("test-agent")

        with tracer.start_as_current_span("execute_tool ping") as span:
            span.set_attribute("arksim.turn_id", 2)
            span.set_attribute("gen_ai.tool.name", "ping")

        provider.force_flush()
        provider.shutdown()

        tool_calls = await receiver.wait_for_traces("otel-conv-4", 2)

    assert len(tool_calls) == 1
    assert tool_calls[0].name == "ping"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_otel_tool_call_id_attribute(_unused_port: int) -> None:
    """gen_ai.tool.call.id attribute is used as the ToolCall.id."""
    port = _unused_port
    async with TraceReceiver(port=port, wait_timeout=1.0) as receiver:
        provider = _create_otel_provider(port, "otel-conv-5")
        tracer = provider.get_tracer("test-agent")

        with tracer.start_as_current_span("execute_tool search") as span:
            span.set_attribute("arksim.turn_id", 0)
            span.set_attribute("gen_ai.tool.name", "search")
            span.set_attribute("gen_ai.tool.call.id", "call_abc123")

        provider.force_flush()
        provider.shutdown()

        tool_calls = await receiver.wait_for_traces("otel-conv-5", 0)

    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call_abc123"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_traceparent_routing(_unused_port: int) -> None:
    """W3C traceparent routing: spans route by trace_id when arksim attributes are absent."""
    import os

    from opentelemetry import context as otel_context
    from opentelemetry.trace import (
        NonRecordingSpan,
        SpanContext,
        TraceFlags,
        set_span_in_context,
    )

    from arksim.tracing.propagation import generate_traceparent

    port = _unused_port
    async with TraceReceiver(port=port, wait_timeout=2.0) as receiver:
        # Simulator generates traceparent and registers the trace_id mapping.
        traceparent = generate_traceparent(receiver, "e2e-conv", 0)
        trace_id = traceparent.split("-")[1]

        # Create a provider with no arksim resource attributes.
        resource = Resource.create({"service.name": "external-agent"})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(
            endpoint=f"http://127.0.0.1:{port}/v1/traces",
        )
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        # Override the OTel context so spans inherit arksim's trace_id.
        parent_ctx = SpanContext(
            trace_id=int(trace_id, 16),
            span_id=int(os.urandom(8).hex(), 16),
            is_remote=True,
            trace_flags=TraceFlags(0x01),
        )
        ctx = set_span_in_context(NonRecordingSpan(parent_ctx))
        token = otel_context.attach(ctx)

        try:
            tracer = provider.get_tracer("external-agent")
            with tracer.start_as_current_span("execute_tool search") as span:
                span.set_attribute("gen_ai.tool.name", "search")
                span.set_attribute("gen_ai.tool.call.arguments", '{"q": "laptop"}')
        finally:
            otel_context.detach(token)

        provider.force_flush()
        provider.shutdown()

        # Receiver routes by trace_id because no arksim.* attributes are present.
        tool_calls = await receiver.wait_for_traces("e2e-conv", 0)

    assert len(tool_calls) == 1
    assert tool_calls[0].name == "search"
    assert tool_calls[0].arguments == {"q": "laptop"}
