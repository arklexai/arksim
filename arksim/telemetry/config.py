# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TelemetryConfig(BaseModel):
    """Configuration for OpenTelemetry telemetry export."""

    enabled: bool = Field(default=False, description="Enable OTel telemetry export")
    service_name: str = Field(default="arksim", description="OTel service name")
    endpoint: str = Field(
        default="http://localhost:4317",
        description="OTLP endpoint URL",
    )
    protocol: Literal["grpc", "http"] = Field(
        default="grpc",
        description="OTLP export protocol",
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="OTLP exporter headers. Values support ${ENV_VAR} syntax.",
    )
    insecure: bool = Field(
        default=True,
        description="Use insecure connection (no TLS) for gRPC",
    )
