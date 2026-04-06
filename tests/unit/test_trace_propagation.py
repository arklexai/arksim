# SPDX-License-Identifier: Apache-2.0
"""Tests for W3C Trace Context propagation."""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import httpx
import pytest

from arksim.tracing.context import trace_traceparent
from arksim.tracing.propagation import (
    generate_traceparent,
    inject_trace_context,
)

W3C_TRACEPARENT_RE = re.compile(r"^00-[0-9a-f]{32}-[0-9a-f]{16}-01$")


class TestGenerateTraceparent:
    def test_format_matches_w3c(self) -> None:
        receiver = MagicMock()
        tp = generate_traceparent(receiver, "conv-1", 0)
        assert W3C_TRACEPARENT_RE.match(tp), f"Bad format: {tp}"

    def test_registers_trace_id_with_receiver(self) -> None:
        receiver = MagicMock()
        tp = generate_traceparent(receiver, "conv-1", 0)
        trace_id = tp.split("-")[1]
        receiver.register_trace_id.assert_called_once_with(trace_id, "conv-1", 0)

    def test_unique_trace_ids(self) -> None:
        receiver = MagicMock()
        tp1 = generate_traceparent(receiver, "conv-1", 0)
        tp2 = generate_traceparent(receiver, "conv-1", 1)
        assert tp1.split("-")[1] != tp2.split("-")[1]


class TestInjectTraceContext:
    @pytest.mark.asyncio
    async def test_injects_header_when_set(self) -> None:
        token = trace_traceparent.set("00-abc-def-01")
        try:
            request = httpx.Request("POST", "http://example.com")
            await inject_trace_context(request)
            assert request.headers["traceparent"] == "00-abc-def-01"
        finally:
            trace_traceparent.reset(token)

    @pytest.mark.asyncio
    async def test_noop_when_none(self) -> None:
        token = trace_traceparent.set(None)
        try:
            request = httpx.Request("POST", "http://example.com")
            await inject_trace_context(request)
            assert "traceparent" not in request.headers
        finally:
            trace_traceparent.reset(token)
