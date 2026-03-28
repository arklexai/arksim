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
    """Invalid JSON body returns 400 so OTel exporters can retry."""
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
        assert b"400" in response
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


@pytest.mark.asyncio
async def test_receiver_concurrent_wait_no_cross_contamination(
    _unused_port: int,
) -> None:
    """Two concurrent wait_for_traces calls get only their own traces."""
    port = _unused_port
    async with TraceReceiver(port=port, wait_timeout=2.0) as receiver:
        payload_a = _make_otlp_payload(
            "conv-a",
            0,
            [
                {
                    "name": "tool_a",
                    "spanId": "sa",
                    "attributes": [
                        {
                            "key": "gen_ai.tool.name",
                            "value": {"stringValue": "tool_a"},
                        },
                    ],
                    "status": {},
                }
            ],
        )
        payload_b = _make_otlp_payload(
            "conv-b",
            0,
            [
                {
                    "name": "tool_b",
                    "spanId": "sb",
                    "attributes": [
                        {
                            "key": "gen_ai.tool.name",
                            "value": {"stringValue": "tool_b"},
                        },
                    ],
                    "status": {},
                }
            ],
        )
        await _push_traces(port, payload_a)
        await _push_traces(port, payload_b)

        results = await asyncio.gather(
            receiver.wait_for_traces("conv-a", 0),
            receiver.wait_for_traces("conv-b", 0),
        )

        tc_a, tc_b = results
        assert len(tc_a) == 1
        assert tc_a[0].name == "tool_a"
        assert len(tc_b) == 1
        assert tc_b[0].name == "tool_b"


class TestAttributePrecedence:
    def test_span_level_overrides_resource_level(self) -> None:
        """When both resource and span have arksim.conversation_id, span wins."""
        payload = {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {
                                "key": "arksim.conversation_id",
                                "value": {"stringValue": "resource-conv"},
                            },
                            {
                                "key": "arksim.turn_id",
                                "value": {"intValue": "0"},
                            },
                        ]
                    },
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "name": "tool",
                                    "spanId": "s1",
                                    "attributes": [
                                        {
                                            "key": "arksim.conversation_id",
                                            "value": {"stringValue": "span-conv"},
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
        assert ("span-conv", 0) in grouped
        assert ("resource-conv", 0) not in grouped


@pytest.mark.asyncio
async def test_receiver_415_when_protobuf_unavailable(_unused_port: int) -> None:
    """Protobuf payload returns 415 when opentelemetry-proto is not installed."""
    import arksim.tracing.receiver as recv_module

    port = _unused_port
    original = recv_module._HAS_PROTOBUF
    recv_module._HAS_PROTOBUF = False
    try:
        async with TraceReceiver(port=port, wait_timeout=0.1):
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            body = b"\x00\x01\x02"
            request = (
                f"POST /v1/traces HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"Content-Type: application/x-protobuf\r\n"
                f"\r\n"
            ).encode() + body
            writer.write(request)
            await writer.drain()
            response = await asyncio.wait_for(reader.read(4096), timeout=5)
            assert b"415" in response
            assert b"arksim[otel]" in response
            writer.close()
            await writer.wait_closed()
    finally:
        recv_module._HAS_PROTOBUF = original


# ── Direct injection (submit_tool_calls) ──


@pytest.mark.asyncio
async def test_submit_tool_calls_direct_injection(_unused_port: int) -> None:
    """submit_tool_calls injects ToolCalls directly, bypassing HTTP."""
    from arksim.simulation_engine.tool_types import ToolCall

    port = _unused_port
    async with TraceReceiver(port=port, wait_timeout=2.0) as receiver:
        tc = ToolCall(id="direct-1", name="search", arguments={"q": "test"})
        receiver.submit_tool_calls("conv-1", 0, [tc])

        result = await receiver.wait_for_traces("conv-1", 0)
        assert len(result) == 1
        assert result[0].name == "search"
        assert result[0].id == "direct-1"
        assert result[0].arguments == {"q": "test"}


@pytest.mark.asyncio
async def test_submit_tool_calls_multiple_turns(_unused_port: int) -> None:
    """Direct injection routes to the correct turn."""
    from arksim.simulation_engine.tool_types import ToolCall

    port = _unused_port
    async with TraceReceiver(port=port, wait_timeout=2.0) as receiver:
        receiver.submit_tool_calls(
            "conv-1", 0, [ToolCall(id="t0", name="tool_a", arguments={})]
        )
        receiver.submit_tool_calls(
            "conv-1", 1, [ToolCall(id="t1", name="tool_b", arguments={})]
        )

        tc0 = await receiver.wait_for_traces("conv-1", 0)
        tc1 = await receiver.wait_for_traces("conv-1", 1)
        assert len(tc0) == 1
        assert tc0[0].name == "tool_a"
        assert len(tc1) == 1
        assert tc1[0].name == "tool_b"


@pytest.mark.asyncio
async def test_submit_tool_calls_mixed_with_http(_unused_port: int) -> None:
    """Direct injection and HTTP spans are merged for the same turn."""
    from arksim.simulation_engine.tool_types import ToolCall

    port = _unused_port
    async with TraceReceiver(port=port, wait_timeout=2.0) as receiver:
        # Direct injection
        receiver.submit_tool_calls(
            "conv-mix", 0, [ToolCall(id="direct-1", name="lookup", arguments={})]
        )

        # HTTP span
        payload = _make_otlp_payload(
            "conv-mix",
            0,
            [
                {
                    "name": "execute_tool search",
                    "spanId": "http-1",
                    "attributes": [
                        {
                            "key": "gen_ai.tool.name",
                            "value": {"stringValue": "search"},
                        },
                    ],
                    "status": {},
                }
            ],
        )
        await _push_traces(port, payload)

        result = await receiver.wait_for_traces("conv-mix", 0)
        names = {tc.name for tc in result}
        assert names == {"lookup", "search"}
        assert len(result) == 2


@pytest.mark.asyncio
async def test_submit_tool_calls_after_stop_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """submit_tool_calls after stop() should warn, not silently buffer."""
    from arksim.simulation_engine.tool_types import ToolCall

    receiver = TraceReceiver(host="127.0.0.1", port=0, wait_timeout=1.0)
    await receiver.start()
    await receiver.stop()

    tc = ToolCall(id="tc-1", name="tool", arguments={})
    with caplog.at_level("WARNING", logger="arksim.tracing.receiver"):
        receiver.submit_tool_calls("conv-1", 0, [tc])

    assert "receiver is not running" in caplog.text
    assert len(receiver._direct_tool_calls) == 0


@pytest.mark.asyncio
async def test_negative_content_length_returns_400(_unused_port: int) -> None:
    """Negative Content-Length should return 400 Bad Request, not 413."""
    port = _unused_port
    receiver = TraceReceiver(host="127.0.0.1", port=port, wait_timeout=1.0)
    await receiver.start()

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        request = "POST /v1/traces HTTP/1.1\r\nContent-Length: -1\r\n\r\n"
        writer.write(request.encode())
        await writer.drain()

        response = await asyncio.wait_for(reader.read(4096), timeout=5)
        response_str = response.decode()
        assert "400" in response_str, f"Expected 400, got: {response_str}"

        writer.close()
        await writer.wait_closed()
    finally:
        await receiver.stop()
