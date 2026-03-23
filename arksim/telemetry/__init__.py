# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from .config import TelemetryConfig
from .provider import get_meter, get_tracer, setup_telemetry, shutdown_telemetry

__all__ = [
    "TelemetryConfig",
    "get_meter",
    "get_tracer",
    "setup_telemetry",
    "shutdown_telemetry",
]
