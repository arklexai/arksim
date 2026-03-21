# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.tracing.receiver."""

from __future__ import annotations

import asyncio
import json

import pytest

from arksim.tracing.receiver import TraceReceiver, _extract_spans_with_routing


def _make_otlp_payload(
    conversation_id: str,
    turn_id: int,
    spans: list[dict],
    *,
    routing_in_resource: bool = False,
) -> dict:
    """Build a minimal OTLP/HTTP JSON payload."""
    routing_attrs = [
        {"key": "arksim.conversation_id", "value": {"stringValue": conversation_id}},
        {"key": "arksim.turn_id", "value": {"intValue": str(turn_id)}},
    ]

    resource_attrs = routing_attrs if routing_in_resource else []

    for span in spans:
        if not routing_in_resource:
            span.setdefault("attributes", []).extend(routing_attrs)

    return {
        "resourceSpans": [
            {
                "resource": {"attributes": resource_attrs},
                "scopeSpans": [{"spans": spans}],
            }
        ]
    }


async def _push_traces(port: int, payload: dict) -> bytes:
    """Push an OTLP payload to the receiver and return the HTTP response."""
    body = json.dumps(payload).encode()
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    request = (
        f"POST /v1/traces HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{port}\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Content-Type: application/json\r\n"
        f"\r\n"
    ).encode() + body
    writer.write(request)
    await writer.drain()
    response = await asyncio.wait_for(reader.read(4096), timeout=5)
    writer.close()
    await writer.wait_closed()
    return response


# ── _extract_spans_with_routing ──


class TestExtractSpansWithRouting:
    def test_span_level_routing(self) -> None:
        payload = _make_otlp_payload(
            "c1",
            0,
            [{"name": "tool", "spanId": "s1", "attributes": [], "status": {}}],
        )
        grouped = _extract_spans_with_routing(payload)
        assert ("c1", 0) in grouped
        assert len(grouped[("c1", 0)]) == 1

    def test_resource_level_routing(self) -> None:
        payload = _make_otlp_payload(
            "c2",
            1,
            [{"name": "tool", "spanId": "s2", "attributes": [], "status": {}}],
            routing_in_resource=True,
        )
        grouped = _extract_spans_with_routing(payload)
        assert ("c2", 1) in grouped

    def test_missing_routing_attrs_skips_span(self) -> None:
        payload = {
            "resourceSpans": [
                {
                    "resource": {"attributes": []},
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "name": "orphan",
                                    "spanId": "o1",
                                    "attributes": [],
                                    "status": {},
                                }
                            ]
                        }
                    ],
                }
            ]
        }
        grouped = _extract_spans_with_routing(payload)
        assert len(grouped) == 0

    def test_invalid_turn_id_skips_span(self) -> None:
        payload = {
            "resourceSpans": [
                {
                    "resource": {"attributes": []},
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "name": "bad_turn",
                                    "spanId": "b1",
                                    "attributes": [
                                        {
                                            "key": "arksim.conversation_id",
                                            "value": {"stringValue": "c1"},
                                        },
                                        {
                                            "key": "arksim.turn_id",
                                            "value": {"stringValue": "not_a_number"},
                                        },
                                    ],
                                    "status": {},
                                }
                            ]
                        }
                    ],
                }
            ]
        }
        grouped = _extract_spans_with_routing(payload)
        assert len(grouped) == 0

    def test_multiple_keys(self) -> None:
        spans_c1 = [{"name": "a", "spanId": "a1", "attributes": [], "status": {}}]
        spans_c2 = [{"name": "b", "spanId": "b1", "attributes": [], "status": {}}]
        p1 = _make_otlp_payload("c1", 0, spans_c1)
        p2 = _make_otlp_payload("c2", 0, spans_c2)
        # Merge resource spans
        payload = {"resourceSpans": p1["resourceSpans"] + p2["resourceSpans"]}
        grouped = _extract_spans_with_routing(payload)
        assert ("c1", 0) in grouped
        assert ("c2", 0) in grouped


# ── TraceReceiver ──


@pytest.fixture
def _unused_port() -> int:
    """Find a free port for testing."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.asyncio
async def test_receiver_lifecycle(_unused_port: int) -> None:
    """Receiver starts and stops cleanly."""
    receiver = TraceReceiver(port=_unused_port, wait_timeout=0.1)
    async with receiver:
        assert receiver._server is not None
    assert receiver._server is None


@pytest.mark.asyncio
async def test_receiver_accepts_traces(_unused_port: int) -> None:
    """Push OTLP traces and retrieve tool calls."""
    port = _unused_port
    async with TraceReceiver(port=port, wait_timeout=0.2) as receiver:
        payload = _make_otlp_payload(
            "conv-1",
            0,
            [
                {
                    "name": "execute_tool search",
                    "spanId": "s1",
                    "attributes": [
                        {
                            "key": "gen_ai.tool.name",
                            "value": {"stringValue": "search"},
                        },
                        {
                            "key": "gen_ai.tool.call.arguments",
                            "value": {"stringValue": '{"q": "test"}'},
                        },
                    ],
                    "status": {"code": 1},
                }
            ],
        )
        response = await _push_traces(port, payload)
        assert b"200" in response

        tool_calls = await receiver.wait_for_traces("conv-1", 0)
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "search"
        assert tool_calls[0].arguments == {"q": "test"}


@pytest.mark.asyncio
async def test_receiver_timeout_returns_empty(_unused_port: int) -> None:
    """If no traces arrive, wait_for_traces returns empty after timeout."""
    async with TraceReceiver(port=_unused_port, wait_timeout=0.1) as receiver:
        tool_calls = await receiver.wait_for_traces("nonexistent", 0)
        assert tool_calls == []


@pytest.mark.asyncio
async def test_receiver_wrong_path_returns_404(_unused_port: int) -> None:
    """Requests to paths other than /v1/traces get 404."""
    port = _unused_port
    async with TraceReceiver(port=port, wait_timeout=0.1):
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        request = (f"GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n").encode()
        writer.write(request)
        await writer.drain()
        response = await asyncio.wait_for(reader.read(4096), timeout=5)
        assert b"404" in response
        writer.close()
        await writer.wait_closed()


@pytest.mark.asyncio
async def test_receiver_traces_before_wait(_unused_port: int) -> None:
    """Traces pushed before wait_for_traces is called are still collected."""
    port = _unused_port
    async with TraceReceiver(port=port, wait_timeout=0.2) as receiver:
        payload = _make_otlp_payload(
            "conv-2",
            1,
            [
                {
                    "name": "execute_tool tool_a",
                    "spanId": "ta1",
                    "attributes": [
                        {"key": "gen_ai.tool.name", "value": {"stringValue": "tool_a"}},
                    ],
                    "status": {},
                }
            ],
        )
        response = await _push_traces(port, payload)
        assert b"200" in response

        tool_calls = await receiver.wait_for_traces("conv-2", 1)
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "tool_a"


@pytest.mark.asyncio
async def test_receiver_deduplication_across_keys(_unused_port: int) -> None:
    """Spans for different (conv, turn) keys are kept separate."""
    port = _unused_port
    async with TraceReceiver(port=port, wait_timeout=0.2) as receiver:
        for conv_id, turn in [("c1", 0), ("c1", 1)]:
            payload = _make_otlp_payload(
                conv_id,
                turn,
                [
                    {
                        "name": f"tool_turn_{turn}",
                        "spanId": f"s-{turn}",
                        "attributes": [
                            {
                                "key": "gen_ai.tool.name",
                                "value": {"stringValue": f"tool_turn_{turn}"},
                            },
                        ],
                        "status": {},
                    }
                ],
            )
            response = await _push_traces(port, payload)
            assert b"200" in response

        tc0 = await receiver.wait_for_traces("c1", 0)
        tc1 = await receiver.wait_for_traces("c1", 1)
        assert len(tc0) == 1
        assert tc0[0].name == "tool_turn_0"
        assert len(tc1) == 1
        assert tc1[0].name == "tool_turn_1"


@pytest.mark.asyncio
async def test_receiver_invalid_json(_unused_port: int) -> None:
    """Invalid JSON body is handled gracefully."""
    port = _unused_port
    async with TraceReceiver(port=port, wait_timeout=0.1):
        body = b"not json at all"
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        request = (
            f"POST /v1/traces HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Content-Type: application/json\r\n"
            f"\r\n"
        ).encode() + body
        writer.write(request)
        await writer.drain()
        response = await asyncio.wait_for(reader.read(4096), timeout=5)
        assert b"200" in response
        writer.close()
        await writer.wait_closed()


@pytest.mark.asyncio
async def test_receiver_multi_batch_collection(_unused_port: int) -> None:
    """Traces pushed in multiple batches for the same turn are all collected."""
    port = _unused_port
    async with TraceReceiver(port=port, wait_timeout=0.5) as receiver:
        # Push first batch
        payload1 = _make_otlp_payload(
            "conv-mb",
            0,
            [
                {
                    "name": "execute_tool tool_a",
                    "spanId": "mb-a",
                    "attributes": [
                        {"key": "gen_ai.tool.name", "value": {"stringValue": "tool_a"}},
                    ],
                    "status": {},
                }
            ],
        )
        response = await _push_traces(port, payload1)
        assert b"200" in response

        # Push second batch (simulating parallel tool execution)
        payload2 = _make_otlp_payload(
            "conv-mb",
            0,
            [
                {
                    "name": "execute_tool tool_b",
                    "spanId": "mb-b",
                    "attributes": [
                        {"key": "gen_ai.tool.name", "value": {"stringValue": "tool_b"}},
                    ],
                    "status": {},
                }
            ],
        )
        response = await _push_traces(port, payload2)
        assert b"200" in response

        # Both batches should be collected
        tool_calls = await receiver.wait_for_traces("conv-mb", 0)
        assert len(tool_calls) == 2
        names = {tc.name for tc in tool_calls}
        assert names == {"tool_a", "tool_b"}


@pytest.mark.asyncio
async def test_receiver_rejects_oversized_payload(_unused_port: int) -> None:
    """Payloads exceeding the size limit are rejected with 413."""
    port = _unused_port
    async with TraceReceiver(port=port, wait_timeout=0.1):
        # Claim a content-length larger than the 10MB limit
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        fake_length = 11 * 1024 * 1024  # 11 MB
        request = (
            f"POST /v1/traces HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            f"Content-Length: {fake_length}\r\n"
            f"Content-Type: application/json\r\n"
            f"\r\n"
        ).encode()
        writer.write(request)
        await writer.drain()
        response = await asyncio.wait_for(reader.read(4096), timeout=5)
        assert b"413" in response
        writer.close()
        await writer.wait_closed()
