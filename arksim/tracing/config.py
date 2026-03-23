# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pydantic import BaseModel, Field


class TraceReceiverConfig(BaseModel):
    """Configuration for the OTel trace receiver."""

    enabled: bool = Field(default=False, description="Enable the trace receiver")
    host: str = Field(
        default="127.0.0.1",
        min_length=1,
        description="Bind address for the receiver",
    )
    port: int = Field(
        default=4318, ge=1, le=65535, description="Port for the OTLP/HTTP receiver"
    )
    wait_timeout: float = Field(
        default=5.0,
        gt=0,
        description="Seconds to wait for traces after each agent response",
    )
