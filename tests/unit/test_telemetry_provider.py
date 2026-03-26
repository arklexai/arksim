# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry", reason="opentelemetry SDK not installed")

_has_grpc = True
try:
    import opentelemetry.exporter.otlp.proto.grpc.trace_exporter  # noqa: F401
except ImportError:
    _has_grpc = False

requires_grpc = pytest.mark.skipif(not _has_grpc, reason="gRPC exporter not installed")

from arksim.telemetry.config import TelemetryConfig  # noqa: E402
from arksim.telemetry.provider import (  # noqa: E402
    _NoOpSpan,
    _NoOpTracer,
    get_tracer,
    setup_telemetry,
    shutdown_telemetry,
)


@pytest.fixture(autouse=True)
def _reset_otel_provider() -> None:
    """Reset the global OTel tracer provider after each test."""
    yield
    shutdown_telemetry()
    import opentelemetry.trace as trace_api

    trace_api._TRACER_PROVIDER = None  # noqa: SLF001
    trace_api._TRACER_PROVIDER_SET_ONCE._done = False  # noqa: SLF001


class TestSetupTelemetry:
    @requires_grpc
    def test_setup_creates_provider(self) -> None:
        """setup_telemetry should configure the global tracer provider."""
        from opentelemetry import trace

        cfg = TelemetryConfig(
            enabled=True,
            endpoint="http://localhost:4317",
            protocol="grpc",
            insecure=True,
        )
        setup_telemetry(cfg)
        try:
            provider = trace.get_tracer_provider()
            assert hasattr(provider, "shutdown")
            tracer = trace.get_tracer("test")
            assert tracer is not None
        finally:
            shutdown_telemetry()

    def test_setup_http_protocol(self) -> None:
        """setup_telemetry should work with http protocol."""
        cfg = TelemetryConfig(
            enabled=True,
            endpoint="http://localhost:4318",
            protocol="http",
        )
        setup_telemetry(cfg)
        try:
            tracer = get_tracer("test")
            assert tracer is not None
        finally:
            shutdown_telemetry()

    @requires_grpc
    def test_setup_with_headers(self) -> None:
        """setup_telemetry should resolve and pass headers."""
        cfg = TelemetryConfig(
            enabled=True,
            endpoint="http://localhost:4317",
            protocol="grpc",
            headers={"x-custom": "value"},
        )
        setup_telemetry(cfg)
        shutdown_telemetry()


class TestShutdownTelemetry:
    def test_shutdown_without_setup_is_noop(self) -> None:
        """shutdown_telemetry should be safe to call when not set up."""
        shutdown_telemetry()  # Should not raise

    @requires_grpc
    def test_double_shutdown_is_safe(self) -> None:
        """Calling shutdown twice should be safe."""
        cfg = TelemetryConfig(
            enabled=True,
            endpoint="http://localhost:4317",
            protocol="grpc",
            insecure=True,
        )
        setup_telemetry(cfg)
        shutdown_telemetry()
        shutdown_telemetry()  # Should not raise


class TestGetTracer:
    @requires_grpc
    def test_returns_tracer_after_setup(self) -> None:
        cfg = TelemetryConfig(
            enabled=True,
            endpoint="http://localhost:4317",
            protocol="grpc",
            insecure=True,
        )
        setup_telemetry(cfg)
        try:
            tracer = get_tracer("arksim")
            assert not isinstance(tracer, _NoOpTracer)
        finally:
            shutdown_telemetry()

    def test_returns_tracer_without_setup(self) -> None:
        """get_tracer should return a real or no-op tracer even without setup."""
        tracer = get_tracer("test")
        assert tracer is not None


class TestNoOpFallbacks:
    def test_noop_tracer_start_as_current_span(self) -> None:
        tracer = _NoOpTracer()
        cm = tracer.start_as_current_span("test")
        with cm as span:
            assert isinstance(span, _NoOpSpan)

    def test_noop_span_methods(self) -> None:
        span = _NoOpSpan()
        span.set_attribute("key", "value")
        span.set_status("OK")
        span.record_exception(ValueError("test"))
        span.add_event("event", {"key": "value"})

    @pytest.mark.asyncio
    async def test_noop_context_manager_async(self) -> None:
        tracer = _NoOpTracer()
        cm = tracer.start_as_current_span("test")
        async with cm as span:
            assert isinstance(span, _NoOpSpan)
