# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import TelemetryConfig

logger = logging.getLogger(__name__)

_TELEMETRY_ACTIVE = False


def _get_service_version() -> str:
    """Return the arksim package version, or 'unknown' if unavailable."""
    try:
        from arksim import __version__

        return __version__
    except Exception:
        return "unknown"


def setup_telemetry(config: TelemetryConfig) -> None:
    """Configure the global OTel TracerProvider and MeterProvider with OTLP exporters.

    Raises ``RuntimeError`` if the OpenTelemetry SDK is not installed.
    """
    global _TELEMETRY_ACTIVE  # noqa: PLW0603

    try:
        from opentelemetry import metrics, trace
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        raise RuntimeError(
            "Telemetry is enabled but opentelemetry-sdk is not installed. "
            "Install it with: pip install arksim[telemetry]"
        ) from None

    from arksim.config.utils import resolve_env_vars

    resolved_headers = resolve_env_vars(dict(config.headers))

    resource = Resource.create(
        {
            "service.name": config.service_name,
            "service.version": _get_service_version(),
        }
    )

    # --- Trace exporter ---
    if config.protocol == "grpc":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
        except ImportError:
            raise RuntimeError(
                "Telemetry protocol is 'grpc' but the gRPC exporter is not installed. "
                "Install it with: pip install opentelemetry-exporter-otlp-proto-grpc"
            ) from None

        span_exporter = OTLPSpanExporter(
            endpoint=config.endpoint,
            headers=tuple(resolved_headers.items()) or None,
            insecure=config.insecure,
        )
    else:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
        except ImportError:
            raise RuntimeError(
                "Telemetry protocol is 'http' but the HTTP exporter is not installed. "
                "Install it with: pip install opentelemetry-exporter-otlp-proto-http"
            ) from None

        span_exporter = OTLPSpanExporter(
            endpoint=config.endpoint,
            headers=resolved_headers or None,
        )

    # --- Metric exporter ---
    try:
        if config.protocol == "grpc":
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )

            metric_exporter = OTLPMetricExporter(
                endpoint=config.endpoint,
                headers=tuple(resolved_headers.items()) or None,
                insecure=config.insecure,
            )
        else:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter,
            )

            metric_exporter = OTLPMetricExporter(
                endpoint=config.endpoint,
                headers=resolved_headers or None,
            )

        metric_reader = PeriodicExportingMetricReader(metric_exporter)
        meter_provider = MeterProvider(
            resource=resource, metric_readers=[metric_reader]
        )
        metrics.set_meter_provider(meter_provider)
    except ImportError:
        logger.warning("Metric exporter not available; only traces will be exported")

    # --- Trace provider ---
    trace_provider = TracerProvider(resource=resource)
    trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(trace_provider)
    _TELEMETRY_ACTIVE = True

    logger.info(
        "Telemetry enabled: exporting to %s via %s",
        config.endpoint,
        config.protocol,
    )


def shutdown_telemetry() -> None:
    """Flush pending spans/metrics and shut down global providers."""
    global _TELEMETRY_ACTIVE  # noqa: PLW0603

    if not _TELEMETRY_ACTIVE:
        return

    try:
        from opentelemetry import metrics, trace

        provider = trace.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()

        meter_provider = metrics.get_meter_provider()
        if hasattr(meter_provider, "shutdown"):
            meter_provider.shutdown()
    except Exception:
        logger.warning("Failed to shut down telemetry providers", exc_info=True)
    finally:
        _TELEMETRY_ACTIVE = False


def get_tracer(name: str = "arksim") -> object:
    """Return a tracer from the global provider.

    If the OTel SDK is not installed or telemetry has not been set up,
    returns the default no-op tracer so callers never need to guard
    against ``None``.
    """
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except ImportError:
        return _NoOpTracer()


def get_meter(name: str = "arksim") -> object:
    """Return a meter from the global provider.

    Returns a no-op meter if the OTel SDK is not installed.
    """
    try:
        from opentelemetry import metrics

        return metrics.get_meter(name)
    except ImportError:
        return _NoOpMeter()


class _NoOpTracer:
    """Minimal stand-in when the OTel SDK is not available."""

    def start_as_current_span(self, name: str, **kwargs: object) -> _NoOpContextManager:
        return _NoOpContextManager()


class _NoOpContextManager:
    """Context manager that yields a no-op span."""

    def __enter__(self) -> _NoOpSpan:
        return _NoOpSpan()

    def __exit__(self, *args: object) -> None:
        pass

    async def __aenter__(self) -> _NoOpSpan:
        return _NoOpSpan()

    async def __aexit__(self, *args: object) -> None:
        pass


class _NoOpSpan:
    """No-op span that silently accepts any attribute/event calls."""

    def set_attribute(self, key: str, value: object) -> None:
        pass

    def set_status(self, status: object, description: str | None = None) -> None:
        pass

    def record_exception(self, exception: BaseException) -> None:
        pass

    def add_event(self, name: str, attributes: dict[str, object] | None = None) -> None:
        pass


class _NoOpMeter:
    """No-op meter stand-in."""

    def create_histogram(self, name: str, **kwargs: object) -> _NoOpInstrument:
        return _NoOpInstrument()

    def create_counter(self, name: str, **kwargs: object) -> _NoOpInstrument:
        return _NoOpInstrument()


class _NoOpInstrument:
    """No-op metric instrument."""

    def record(self, value: float, attributes: dict[str, object] | None = None) -> None:
        pass

    def add(self, value: int, attributes: dict[str, object] | None = None) -> None:
        pass
