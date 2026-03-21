# SPDX-License-Identifier: Apache-2.0
"""Lightweight OTLP/HTTP trace receiver for capturing agent tool call spans.

Accepts ``POST /v1/traces`` with OTLP JSON payloads, extracts tool call spans,
and buffers them keyed by ``(conversation_id, turn_id)`` so the simulator can
collect them after each agent turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import defaultdict
from http import HTTPStatus
from typing import Any

from arksim.simulation_engine.tool_types import ToolCall

from .span_converter import spans_to_tool_calls

logger = logging.getLogger(__name__)

# Maximum accepted payload size (10 MB). Requests exceeding this are rejected.
_MAX_PAYLOAD_BYTES = 10 * 1024 * 1024


def _extract_spans_with_routing(
    payload: dict[str, Any],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Parse an OTLP/HTTP JSON payload and group spans by (conversation_id, turn_id).

    Routing attributes (``arksim.conversation_id``, ``arksim.turn_id``) are
    looked up first in span attributes, then in resource attributes.
    """
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)

    for resource_span in payload.get("resourceSpans", []):
        resource_attrs = resource_span.get("resource", {}).get("attributes", [])

        for scope_span in resource_span.get("scopeSpans", []):
            for span in scope_span.get("spans", []):
                span_attrs = span.get("attributes", [])

                # Look up routing keys: span attrs take precedence
                conv_id = _find_attr(
                    span_attrs, "arksim.conversation_id"
                ) or _find_attr(resource_attrs, "arksim.conversation_id")
                raw_turn = _find_attr(span_attrs, "arksim.turn_id") or _find_attr(
                    resource_attrs, "arksim.turn_id"
                )

                if conv_id is None or raw_turn is None:
                    logger.debug(
                        "Span %s missing arksim routing attributes, skipping",
                        span.get("spanId", "?"),
                    )
                    continue

                try:
                    turn_id = int(raw_turn)
                except (ValueError, TypeError):
                    logger.warning(
                        "Invalid arksim.turn_id value %r in span %s",
                        raw_turn,
                        span.get("spanId", "?"),
                    )
                    continue

                grouped[(conv_id, turn_id)].append(span)

    return grouped


def _find_attr(attrs: list[dict[str, Any]], key: str) -> str | None:
    """Find an attribute value by key in an OTLP attribute list."""
    for attr in attrs:
        if attr.get("key") == key:
            value = attr.get("value", {})
            str_val = value.get("stringValue")
            if str_val is not None:
                return str_val
            int_val = value.get("intValue")
            if int_val is not None:
                return str(int_val)
    return None


class TraceReceiver:
    """Async context manager that runs an OTLP/HTTP receiver.

    Usage::

        async with TraceReceiver(port=9712, wait_timeout=5.0) as receiver:
            # ... run simulation turns ...
            tool_calls = await receiver.wait_for_traces("conv-1", 0)
    """

    def __init__(self, port: int = 9712, wait_timeout: float = 5.0) -> None:
        self.port = port
        self.wait_timeout = wait_timeout
        self._spans: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        self._events: dict[tuple[str, int], asyncio.Event] = {}
        self._server: asyncio.Server | None = None
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> TraceReceiver:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    async def start(self) -> None:
        """Start the HTTP server."""
        self._server = await asyncio.start_server(
            self._handle_connection, "127.0.0.1", self.port
        )
        logger.info("Trace receiver listening on 127.0.0.1:%d", self.port)

    async def stop(self) -> None:
        """Stop the HTTP server and clean up."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            logger.info("Trace receiver stopped")

    async def wait_for_traces(
        self, conversation_id: str, turn_id: int
    ) -> list[ToolCall]:
        """Wait for tool call spans for the given conversation turn.

        Always waits the full ``wait_timeout`` to allow multi-batch trace
        pushes to arrive. Returns collected ToolCall objects. If no traces
        arrive within the timeout, logs a warning and returns an empty list.
        """
        key = (conversation_id, turn_id)

        # Wait the full timeout so that agents pushing traces in multiple
        # batches (e.g. parallel tool execution) have time to deliver all
        # spans before we drain the buffer.
        await asyncio.sleep(self.wait_timeout)

        async with self._lock:
            spans = self._spans.pop(key, [])
            self._events.pop(key, None)

        tool_calls = spans_to_tool_calls(spans)
        if tool_calls:
            logger.debug(
                "Collected %d tool calls for (%s, %d)",
                len(tool_calls),
                conversation_id,
                turn_id,
            )
        else:
            logger.warning(
                "No tool call traces received for (%s, %d) within %.1fs timeout",
                conversation_id,
                turn_id,
                self.wait_timeout,
            )
        return tool_calls

    # ── HTTP handling ──

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a raw TCP connection as a minimal HTTP request."""
        try:
            # Read request line
            request_line = await asyncio.wait_for(reader.readline(), timeout=10)
            if not request_line:
                return

            parts = request_line.decode("utf-8", errors="replace").strip().split()
            if len(parts) < 2:
                self._send_response(writer, HTTPStatus.BAD_REQUEST)
                return

            method, path = parts[0], parts[1]

            # Read headers
            content_length = 0
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=10)
                if line in (b"\r\n", b"\n", b""):
                    break
                header = line.decode("utf-8", errors="replace").strip().lower()
                if header.startswith("content-length:"):
                    content_length = int(header.split(":", 1)[1].strip())

            # Only accept POST /v1/traces
            if method != "POST" or path != "/v1/traces":
                self._send_response(writer, HTTPStatus.NOT_FOUND)
                return

            # Reject oversized payloads
            if content_length > _MAX_PAYLOAD_BYTES:
                logger.warning(
                    "Trace payload too large (%d bytes), rejecting",
                    content_length,
                )
                self._send_response(writer, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                return

            # Read body
            body = b""
            if content_length > 0:
                body = await asyncio.wait_for(
                    reader.readexactly(content_length), timeout=30
                )

            await self._handle_traces(body)
            self._send_response(writer, HTTPStatus.OK, b"{}")

        except (asyncio.TimeoutError, ConnectionError, asyncio.IncompleteReadError):
            logger.debug("Connection error in trace receiver")
        except Exception:
            logger.exception("Unexpected error in trace receiver")
            with contextlib.suppress(Exception):
                self._send_response(writer, HTTPStatus.INTERNAL_SERVER_ERROR)
        finally:
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    @staticmethod
    def _send_response(
        writer: asyncio.StreamWriter,
        status: HTTPStatus,
        body: bytes = b"",
    ) -> None:
        """Write a minimal HTTP response."""
        response = (
            f"HTTP/1.1 {status.value} {status.phrase}\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Content-Type: application/json\r\n"
            f"\r\n"
        ).encode() + body
        writer.write(response)

    async def _handle_traces(self, body: bytes) -> None:
        """Parse OTLP JSON body and buffer spans by routing key."""
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Invalid JSON in trace payload")
            return

        grouped = _extract_spans_with_routing(payload)
        if not grouped:
            logger.debug("No routable spans in trace payload")
            return

        async with self._lock:
            for key, spans in grouped.items():
                self._spans[key].extend(spans)
                event = self._events.setdefault(key, asyncio.Event())
                event.set()
