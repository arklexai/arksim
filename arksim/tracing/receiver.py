# SPDX-License-Identifier: Apache-2.0
"""Lightweight OTLP/HTTP trace receiver for capturing agent tool call spans.

Accepts ``POST /v1/traces`` with OTLP payloads (protobuf or JSON), extracts
tool call spans, and buffers them keyed by ``(conversation_id, turn_id)`` so
the simulator can collect them after each agent turn.

Protobuf support requires ``opentelemetry-proto`` (installed automatically
with any OTel exporter package, or via ``pip install arksim[otel]``).
Falls back to JSON-only when unavailable; protobuf payloads get HTTP 415.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
from collections import defaultdict
from http import HTTPStatus
from typing import Any

from arksim.simulation_engine.tool_types import ToolCall
from arksim.tracing._attrs import get_attr
from arksim.tracing.span_converter import spans_to_tool_calls

logger = logging.getLogger(__name__)

# Maximum accepted payload size (10 MB). Requests exceeding this are rejected.
_MAX_PAYLOAD_BYTES = 10 * 1024 * 1024

# Short delay after the first trace arrives to catch trailing spans
# from multi-batch pushes (e.g. parallel tool execution).
_SETTLE_SECONDS = 0.1

# Try to import protobuf support for parsing real OTel SDK payloads.
try:
    from google.protobuf.json_format import MessageToDict
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )

    _HAS_PROTOBUF = True
except ImportError:
    _HAS_PROTOBUF = False


def _parse_protobuf_payload(body: bytes) -> dict[str, Any]:
    """Deserialize an OTLP protobuf payload into the same dict structure as OTLP JSON.

    Preserves zero-valued fields (e.g. ``int_value=0`` for ``arksim.turn_id=0``)
    so they are not silently dropped. Uses ``always_print_fields_with_no_presence``
    on protobuf 4.24+ and falls back to ``including_default_value_fields`` on
    older versions.
    """
    request = ExportTraceServiceRequest()
    request.ParseFromString(body)
    try:
        return MessageToDict(
            request,
            preserving_proto_field_name=True,
            always_print_fields_with_no_presence=True,
        )
    except TypeError:
        # protobuf < 4.24: always_print_fields_with_no_presence not available
        return MessageToDict(
            request,
            preserving_proto_field_name=True,
            including_default_value_fields=True,
        )


def _extract_spans_with_routing(
    payload: dict[str, Any],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Parse an OTLP payload dict and group spans by (conversation_id, turn_id).

    Routing attributes (``arksim.conversation_id``, ``arksim.turn_id``) are
    looked up first in span attributes, then in resource attributes.
    """
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)

    for resource_span in payload.get(
        "resource_spans", payload.get("resourceSpans", [])
    ):
        resource = resource_span.get("resource", {})
        resource_attrs = resource.get("attributes", [])

        for scope_span in resource_span.get(
            "scope_spans", resource_span.get("scopeSpans", [])
        ):
            for span in scope_span.get("spans", []):
                span_attrs = span.get("attributes", [])

                # Look up routing keys: span attrs take precedence.
                # Use `is not None` (not `or`) to preserve falsy values
                # like empty strings, since get_attr returns None for missing.
                span_conv = get_attr(span_attrs, "arksim.conversation_id")
                conv_id = (
                    span_conv
                    if span_conv is not None
                    else get_attr(resource_attrs, "arksim.conversation_id")
                )
                span_turn = get_attr(span_attrs, "arksim.turn_id")
                raw_turn = (
                    span_turn
                    if span_turn is not None
                    else get_attr(resource_attrs, "arksim.turn_id")
                )

                if conv_id is None or raw_turn is None:
                    logger.debug(
                        "Span %s missing arksim routing attributes, skipping",
                        span.get("spanId", span.get("span_id", "?")),
                    )
                    continue

                try:
                    turn_id = int(raw_turn)
                except (ValueError, TypeError):
                    logger.warning(
                        "Invalid arksim.turn_id value %r in span %s",
                        raw_turn,
                        span.get("spanId", span.get("span_id", "?")),
                    )
                    continue

                grouped[(conv_id, turn_id)].append(span)

    return grouped


class TraceReceiver:
    """Async context manager that runs an OTLP/HTTP receiver.

    Accepts both protobuf and JSON payloads on the standard OTLP/HTTP port.
    Protobuf is the default format used by OTel SDK exporters.

    Usage::

        async with TraceReceiver(host="127.0.0.1", port=4318, wait_timeout=5.0) as receiver:
            # ... run simulation turns ...
            tool_calls = await receiver.wait_for_traces("conv-1", 0)
    """

    def __init__(
        self, host: str = "127.0.0.1", port: int = 4318, wait_timeout: float = 5.0
    ) -> None:
        self.host = host
        self.port = port
        self.wait_timeout = wait_timeout
        self._spans: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        self._direct_tool_calls: dict[tuple[str, int], list[ToolCall]] = defaultdict(
            list
        )
        self._events: dict[tuple[str, int], asyncio.Event] = {}
        # Threading events for direct injection signaling (works across threads)
        self._direct_events: dict[tuple[str, int], threading.Event] = {}
        self._server: asyncio.Server | None = None
        self._lock = asyncio.Lock()
        self._submit_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    async def __aenter__(self) -> TraceReceiver:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    async def start(self) -> None:
        """Start the HTTP server."""
        self._loop = asyncio.get_running_loop()
        self._server = await asyncio.start_server(
            self._handle_connection, self.host, self.port
        )
        proto_status = "protobuf+JSON" if _HAS_PROTOBUF else "JSON only"
        logger.info(
            "Trace receiver listening on %s:%d (%s)",
            self.host,
            self.port,
            proto_status,
        )

    async def stop(self) -> None:
        """Stop the HTTP server and clean up."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            logger.info("Trace receiver stopped")
        self._spans.clear()
        self._direct_tool_calls.clear()
        self._events.clear()
        self._direct_events.clear()

    def submit_tool_calls(
        self, conversation_id: str, turn_id: int, tool_calls: list[ToolCall]
    ) -> None:
        """Inject tool calls directly, bypassing HTTP.

        Use this when the agent runs in the same process as arksim.
        Thread-safe: can be called from any thread or from the event loop
        thread (e.g. a TracingProcessor callback during Runner.run).
        """
        if not tool_calls or self._loop is None:
            return

        key = (conversation_id, turn_id)

        # Write to the buffer and signal immediately. Both the threading
        # lock and threading.Event work from any thread without needing
        # the event loop.
        with self._submit_lock:
            self._direct_tool_calls[key].extend(tool_calls)
            evt = self._direct_events.setdefault(key, threading.Event())
            evt.set()

    async def wait_for_traces(
        self, conversation_id: str, turn_id: int
    ) -> list[ToolCall]:
        """Wait for tool call spans for the given conversation turn.

        Uses event-based signaling with a settling window. When traces
        arrive, waits an additional short period to catch trailing spans
        from multi-batch pushes, then drains immediately. Falls back to
        the full ``wait_timeout`` if no traces arrive.
        """
        key = (conversation_id, turn_id)

        async with self._lock:
            http_event = self._events.setdefault(key, asyncio.Event())

        # Also prepare a threading event for direct injection
        with self._submit_lock:
            direct_event = self._direct_events.setdefault(key, threading.Event())

        # Wait for either HTTP spans or direct injection to arrive.
        # Check direct_event in a polling loop since threading.Event
        # and asyncio.Event can't be awaited together.
        timed_out = True
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.wait_timeout
        while loop.time() < deadline:
            # Check direct injection (instant, no await)
            if direct_event.is_set():
                timed_out = False
                break
            # Check HTTP path (short wait to yield to event loop)
            try:
                remaining = deadline - loop.time()
                await asyncio.wait_for(
                    http_event.wait(), timeout=min(0.1, max(0, remaining))
                )
                timed_out = False
                break
            except asyncio.TimeoutError:
                continue

        if not timed_out:
            # Traces arrived; settle briefly to catch trailing batches
            await asyncio.sleep(_SETTLE_SECONDS)

        # Drain both buffers and clean up stale entries from earlier turns.
        # Compute stale keys from both buffers so cleanup works regardless
        # of which capture path the agent uses.
        async with self._lock:
            spans = self._spans.pop(key, [])
            self._events.pop(key, None)
            stale_keys = {
                k for k in self._spans if k[0] == conversation_id and k[1] < turn_id
            }
            for k in stale_keys:
                del self._spans[k]
                self._events.pop(k, None)

        with self._submit_lock:
            direct = self._direct_tool_calls.pop(key, [])
            self._direct_events.pop(key, None)
            stale_keys |= {
                k
                for k in self._direct_tool_calls
                if k[0] == conversation_id and k[1] < turn_id
            }
            for k in stale_keys:
                self._direct_tool_calls.pop(k, None)
                self._direct_events.pop(k, None)

        # Merge: HTTP-received spans (converted) + directly submitted ToolCalls
        tool_calls = spans_to_tool_calls(spans) + direct
        if tool_calls:
            logger.debug(
                "Collected %d tool calls for (%s, %d)",
                len(tool_calls),
                conversation_id,
                turn_id,
            )
        else:
            # Debug, not warning: agents may produce pure text responses
            # with no tool calls on some turns, which is normal behavior.
            logger.debug(
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
        """Handle a raw TCP connection as a minimal HTTP/1.1 request.

        Uses Content-Length for body framing. Chunked Transfer-Encoding is
        not supported since OTel SDK exporters always send Content-Length.
        """
        try:
            # Read request line
            request_line = await asyncio.wait_for(reader.readline(), timeout=10)
            if not request_line:
                return

            parts = request_line.decode("utf-8", errors="replace").strip().split()
            if len(parts) < 2:
                await self._send_response(writer, HTTPStatus.BAD_REQUEST)
                return

            method, path = parts[0], parts[1]

            # Read headers
            content_length = 0
            content_type = ""
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=10)
                if line in (b"\r\n", b"\n", b""):
                    break
                header = line.decode("utf-8", errors="replace").strip()
                header_lower = header.lower()
                if header_lower.startswith("content-length:"):
                    try:
                        content_length = int(header_lower.split(":", 1)[1].strip())
                    except ValueError:
                        await self._send_response(writer, HTTPStatus.BAD_REQUEST)
                        return
                elif header_lower.startswith("content-type:"):
                    content_type = header_lower.split(":", 1)[1].strip()

            # Only accept POST /v1/traces
            if method != "POST" or path != "/v1/traces":
                await self._send_response(writer, HTTPStatus.NOT_FOUND)
                return

            # Reject invalid or oversized payloads
            if content_length < 0 or content_length > _MAX_PAYLOAD_BYTES:
                logger.warning(
                    "Trace payload too large (%d bytes), rejecting",
                    content_length,
                )
                await self._send_response(writer, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                return

            # Read body
            body = b""
            if content_length > 0:
                body = await asyncio.wait_for(
                    reader.readexactly(content_length), timeout=30
                )

            # Reject protobuf when opentelemetry-proto is not installed
            is_protobuf = "application/x-protobuf" in content_type
            if is_protobuf and not _HAS_PROTOBUF:
                logger.warning(
                    "Received protobuf payload but opentelemetry-proto is not "
                    "installed. Install with: pip install arksim[otel]"
                )
                await self._send_response(
                    writer,
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    b'{"error": "Protobuf not supported. Install with: pip install arksim[otel]"}',
                )
                return

            parsed_ok = await self._handle_traces(body, is_protobuf=is_protobuf)
            if parsed_ok:
                await self._send_response(writer, HTTPStatus.OK, b"{}")
            else:
                await self._send_response(
                    writer,
                    HTTPStatus.BAD_REQUEST,
                    b'{"error": "Failed to parse trace payload"}',
                )

        except (asyncio.TimeoutError, ConnectionError, asyncio.IncompleteReadError):
            logger.debug("Connection error in trace receiver")
        except Exception:
            logger.exception("Unexpected error in trace receiver")
            with contextlib.suppress(Exception):
                await self._send_response(writer, HTTPStatus.INTERNAL_SERVER_ERROR)
        finally:
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    @staticmethod
    async def _send_response(
        writer: asyncio.StreamWriter,
        status: HTTPStatus,
        body: bytes = b"",
    ) -> None:
        """Write a minimal HTTP response."""
        response = (
            f"HTTP/1.1 {status.value} {status.phrase}\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Content-Type: application/json\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode() + body
        writer.write(response)
        await writer.drain()

    async def _handle_traces(self, body: bytes, *, is_protobuf: bool = False) -> bool:
        """Parse OTLP body (protobuf or JSON) and buffer spans by routing key.

        Returns True on success, False on parse failure.
        """
        if is_protobuf:
            try:
                payload = _parse_protobuf_payload(body)
            except Exception:
                logger.warning("Failed to parse protobuf trace payload")
                return False
        else:
            try:
                payload = json.loads(body)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Invalid JSON in trace payload")
                return False

        grouped = _extract_spans_with_routing(payload)
        if not grouped:
            logger.debug("No routable spans in trace payload")
            return True

        async with self._lock:
            for key, spans in grouped.items():
                self._spans[key].extend(spans)
                event = self._events.setdefault(key, asyncio.Event())
                event.set()

        return True
