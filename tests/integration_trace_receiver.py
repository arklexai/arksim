#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Manual integration test for the trace receiver.

Starts a real TraceReceiver, pushes OTLP traces over HTTP (simulating
what an instrumented agent would do), and verifies the tool calls are
collected correctly. This exercises the full network path.

Usage:
    python tests/integration_trace_receiver.py
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_otlp_payload(conv_id: str, turn_id: int, tools: list[dict]) -> dict:
    """Build an OTLP/HTTP JSON payload with tool call spans."""
    spans = []
    for tool in tools:
        span = {
            "name": f"execute_tool {tool['name']}",
            "spanId": tool.get("id", tool["name"]),
            "startTimeUnixNano": "1679000000000000000",
            "endTimeUnixNano": "1679000001000000000",
            "attributes": [
                {"key": "arksim.conversation_id", "value": {"stringValue": conv_id}},
                {"key": "arksim.turn_id", "value": {"intValue": str(turn_id)}},
                {"key": "gen_ai.tool.name", "value": {"stringValue": tool["name"]}},
            ],
            "status": {"code": 1},
        }
        if "arguments" in tool:
            span["attributes"].append(
                {
                    "key": "gen_ai.tool.call.arguments",
                    "value": {"stringValue": json.dumps(tool["arguments"])},
                }
            )
        if "result" in tool:
            span["attributes"].append(
                {
                    "key": "gen_ai.tool.call.result",
                    "value": {"stringValue": tool["result"]},
                }
            )
        if tool.get("error"):
            span["status"] = {"code": 2, "message": tool["error"]}
        spans.append(span)

    return {
        "resourceSpans": [
            {
                "resource": {"attributes": []},
                "scopeSpans": [{"spans": spans}],
            }
        ]
    }


async def _push_traces(port: int, payload: dict) -> int:
    """Send an OTLP payload and return the HTTP status code."""
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
    # Parse status code from "HTTP/1.1 200 OK"
    status_line = response.split(b"\r\n")[0].decode()
    return int(status_line.split()[1])


async def main() -> None:
    from arksim.tracing.receiver import TraceReceiver

    port = _find_free_port()
    print(f"[1] Starting TraceReceiver on port {port}...")

    async with TraceReceiver(port=port, wait_timeout=1.0) as receiver:
        print("    OK - receiver started")

        # ── Test 1: Single tool call ──
        print("\n[2] Pushing single tool call (lookup_customer)...")
        payload = _make_otlp_payload(
            "conv-test-1",
            0,
            [
                {
                    "name": "lookup_customer",
                    "id": "tc-001",
                    "arguments": {"email": "alice@example.com"},
                    "result": '{"id": "C-001", "name": "Alice Johnson"}',
                }
            ],
        )
        status = await _push_traces(port, payload)
        assert status == 200, f"Expected 200, got {status}"
        print(f"    HTTP {status} OK")

        print("    Waiting for traces...")
        tool_calls = await receiver.wait_for_traces("conv-test-1", 0)
        assert len(tool_calls) == 1, f"Expected 1 tool call, got {len(tool_calls)}"
        tc = tool_calls[0]
        assert tc.name == "lookup_customer"
        assert tc.arguments == {"email": "alice@example.com"}
        assert tc.result == '{"id": "C-001", "name": "Alice Johnson"}'
        assert tc.error is None
        print(f"    OK - got ToolCall(name={tc.name!r}, args={tc.arguments})")

        # ── Test 2: Multiple tool calls in one batch ──
        print("\n[3] Pushing 3 tool calls in one batch...")
        payload = _make_otlp_payload(
            "conv-test-2",
            0,
            [
                {
                    "name": "search_products",
                    "id": "tc-010",
                    "arguments": {"query": "laptop"},
                },
                {
                    "name": "get_order",
                    "id": "tc-011",
                    "arguments": {"order_id": "ORD-1001"},
                },
                {
                    "name": "cancel_order",
                    "id": "tc-012",
                    "arguments": {"order_id": "ORD-1002"},
                    "error": "denied",
                },
            ],
        )
        status = await _push_traces(port, payload)
        assert status == 200
        print(f"    HTTP {status} OK")

        tool_calls = await receiver.wait_for_traces("conv-test-2", 0)
        assert len(tool_calls) == 3, f"Expected 3 tool calls, got {len(tool_calls)}"
        names = [tc.name for tc in tool_calls]
        assert "search_products" in names
        assert "get_order" in names
        assert "cancel_order" in names
        errored = [tc for tc in tool_calls if tc.error]
        assert len(errored) == 1
        assert errored[0].name == "cancel_order"
        assert errored[0].error == "denied"
        print(f"    OK - got {len(tool_calls)} tool calls: {names}")
        print(f"    Error tool call: {errored[0].name} -> {errored[0].error!r}")

        # ── Test 3: Multi-batch (two separate HTTP pushes for same turn) ──
        print("\n[4] Pushing 2 separate batches for the same turn...")
        p1 = _make_otlp_payload(
            "conv-test-3",
            0,
            [{"name": "tool_a", "id": "tc-a"}],
        )
        p2 = _make_otlp_payload(
            "conv-test-3",
            0,
            [{"name": "tool_b", "id": "tc-b"}],
        )
        s1 = await _push_traces(port, p1)
        s2 = await _push_traces(port, p2)
        assert s1 == 200 and s2 == 200
        print(f"    HTTP {s1}, {s2} OK")

        tool_calls = await receiver.wait_for_traces("conv-test-3", 0)
        assert len(tool_calls) == 2, f"Expected 2 tool calls, got {len(tool_calls)}"
        names = {tc.name for tc in tool_calls}
        assert names == {"tool_a", "tool_b"}
        print(f"    OK - both batches collected: {names}")

        # ── Test 4: Timeout with no traces ──
        print("\n[5] Testing timeout (no traces pushed)...")
        tool_calls = await receiver.wait_for_traces("conv-nonexistent", 99)
        assert tool_calls == []
        print("    OK - empty list returned after timeout")

        # ── Test 5: 404 for wrong path ──
        print("\n[6] Testing 404 for wrong path...")
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(f"GET /wrong HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode())
        await writer.drain()
        resp = await asyncio.wait_for(reader.read(4096), timeout=5)
        writer.close()
        await writer.wait_closed()
        assert b"404" in resp
        print("    OK - got 404")

    print("\n    Receiver stopped cleanly")
    print("\n=== ALL INTEGRATION TESTS PASSED ===")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        sys.exit(1)
