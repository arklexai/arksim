# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pydantic import BaseModel, Field


class TraceReceiverConfig(BaseModel):
    """Configuration for the OTel trace receiver."""

    enabled: bool = Field(default=False, description="Enable the trace receiver")
    port: int = Field(default=4318, description="Port for the OTLP/HTTP receiver")
    wait_timeout: float = Field(
        default=5.0,
        description="Seconds to wait for traces after each agent response",
    )
