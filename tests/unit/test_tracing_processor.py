# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.tracing.openai.ArksimTracingProcessor."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("agents", reason="openai-agents SDK not installed")

from arksim.tracing.openai import ArksimTracingProcessor


def _make_function_span(
    *,
    trace_id: str = "trace-1",
    span_id: str = "span-1",
    name: str = "search",
    input_json: str | None = '{"q": "test"}',
    output: str | None = '{"results": []}',
    error: dict | None = None,
) -> SimpleNamespace:
    """Build a minimal mock span with FunctionSpanData."""
    from agents.tracing.span_data import FunctionSpanData

    span_data = FunctionSpanData(name=name, input=input_json, output=output)
    return SimpleNamespace(
        trace_id=trace_id,
        span_id=span_id,
        span_data=span_data,
        error=error,
    )


def _make_non_function_span(*, trace_id: str = "trace-1") -> SimpleNamespace:
    """Build a mock span with non-FunctionSpanData."""
    return SimpleNamespace(
        trace_id=trace_id,
        span_id="span-other",
        span_data=SimpleNamespace(),  # not a FunctionSpanData instance
        error=None,
    )


class TestOnSpanEndExplicitContext:
    """Test on_span_end with explicit register_context (standalone .trace() path)."""

    def test_function_span_submits_tool_call(self) -> None:
        """A FunctionSpanData span with registered context submits a ToolCall."""
        processor = ArksimTracingProcessor()
        receiver = MagicMock()

        processor._trace_contexts["trace-1"] = ("conv-1", 0, receiver)
        span = _make_function_span()
        processor.on_span_end(span)

        receiver.submit_tool_calls.assert_called_once()
        args = receiver.submit_tool_calls.call_args
        assert args[0][0] == "conv-1"
        assert args[0][1] == 0
        tc_list = args[0][2]
        assert len(tc_list) == 1
        assert tc_list[0].name == "search"
        assert tc_list[0].arguments == {"q": "test"}
        assert tc_list[0].result == '{"results": []}'
        assert tc_list[0].id == "span-1"
        assert tc_list[0].error is None

    def test_non_function_span_is_skipped(self) -> None:
        """Non-FunctionSpanData spans (LLM, agent, etc.) are ignored."""
        processor = ArksimTracingProcessor()
        receiver = MagicMock()

        processor._trace_contexts["trace-1"] = ("conv-1", 0, receiver)
        span = _make_non_function_span()
        processor.on_span_end(span)

        receiver.submit_tool_calls.assert_not_called()

    def test_unregistered_trace_id_is_skipped(self) -> None:
        """Spans without registered context or contextvars are ignored."""
        processor = ArksimTracingProcessor()
        span = _make_function_span(trace_id="trace-unknown")
        processor.on_span_end(span)
        # No crash, no submit

    def test_no_receiver_does_not_crash(self) -> None:
        """When receiver is None, on_span_end completes without error."""
        processor = ArksimTracingProcessor()
        processor._trace_contexts["trace-1"] = ("conv-1", 0, None)
        span = _make_function_span()
        processor.on_span_end(span)

    def test_error_span_captures_message(self) -> None:
        """Span with an error dict captures the error message."""
        processor = ArksimTracingProcessor()
        receiver = MagicMock()

        processor._trace_contexts["trace-1"] = ("conv-1", 0, receiver)
        span = _make_function_span(error={"message": "connection timeout"})
        processor.on_span_end(span)

        tc_list = receiver.submit_tool_calls.call_args[0][2]
        assert tc_list[0].error == "connection timeout"

    def test_error_span_missing_message_key(self) -> None:
        """Span with error dict lacking 'message' falls back to 'unknown error'."""
        processor = ArksimTracingProcessor()
        receiver = MagicMock()

        processor._trace_contexts["trace-1"] = ("conv-1", 0, receiver)
        span = _make_function_span(error={"code": 500})
        processor.on_span_end(span)

        tc_list = receiver.submit_tool_calls.call_args[0][2]
        assert tc_list[0].error == "unknown error"

    def test_malformed_input_json_yields_empty_arguments(self) -> None:
        """Invalid JSON in span_data.input produces empty arguments dict."""
        processor = ArksimTracingProcessor()
        receiver = MagicMock()

        processor._trace_contexts["trace-1"] = ("conv-1", 0, receiver)
        span = _make_function_span(input_json="not valid json")
        processor.on_span_end(span)

        tc_list = receiver.submit_tool_calls.call_args[0][2]
        assert tc_list[0].arguments == {}

    def test_non_dict_input_json_yields_empty_arguments(self) -> None:
        """JSON that parses to a non-dict (e.g. list) produces empty arguments."""
        processor = ArksimTracingProcessor()
        receiver = MagicMock()

        processor._trace_contexts["trace-1"] = ("conv-1", 0, receiver)
        span = _make_function_span(input_json='["a", "b"]')
        processor.on_span_end(span)

        tc_list = receiver.submit_tool_calls.call_args[0][2]
        assert tc_list[0].arguments == {}

    def test_dict_output_is_json_serialized(self) -> None:
        """Non-string output (dict) is JSON-serialized in the ToolCall result."""
        processor = ArksimTracingProcessor()
        receiver = MagicMock()

        processor._trace_contexts["trace-1"] = ("conv-1", 0, receiver)
        from agents.tracing.span_data import FunctionSpanData

        span_data = FunctionSpanData(
            name="tool",
            input=None,
            output={"key": "value"},  # type: ignore[arg-type]
        )
        span = SimpleNamespace(
            trace_id="trace-1",
            span_id="span-dict",
            span_data=span_data,
            error=None,
        )
        processor.on_span_end(span)

        tc_list = receiver.submit_tool_calls.call_args[0][2]
        assert tc_list[0].result == '{"key": "value"}'

    def test_null_input_and_output(self) -> None:
        """Span with no input and no output produces empty args and None result."""
        processor = ArksimTracingProcessor()
        receiver = MagicMock()

        processor._trace_contexts["trace-1"] = ("conv-1", 0, receiver)
        span = _make_function_span(input_json=None, output=None)
        processor.on_span_end(span)

        tc_list = receiver.submit_tool_calls.call_args[0][2]
        assert tc_list[0].arguments == {}
        assert tc_list[0].result is None


class TestOnSpanEndContextVars:
    """Test on_span_end with contextvars (simulator-managed path)."""

    def test_contextvar_routing(self) -> None:
        """on_span_end reads routing from contextvars when no explicit context."""
        from arksim.tracing.context import (
            trace_conversation_id,
            trace_receiver_ref,
            trace_turn_id,
        )

        processor = ArksimTracingProcessor()
        receiver = MagicMock()

        trace_conversation_id.set("cv-conv")
        trace_turn_id.set(3)
        trace_receiver_ref.set(receiver)

        try:
            span = _make_function_span(trace_id="unregistered-trace")
            processor.on_span_end(span)

            receiver.submit_tool_calls.assert_called_once()
            args = receiver.submit_tool_calls.call_args
            assert args[0][0] == "cv-conv"
            assert args[0][1] == 3
        finally:
            trace_conversation_id.set(None)
            trace_turn_id.set(None)
            trace_receiver_ref.set(None)

    def test_explicit_context_takes_precedence(self) -> None:
        """Explicit register_context wins over contextvars."""
        from arksim.tracing.context import (
            trace_conversation_id,
            trace_receiver_ref,
            trace_turn_id,
        )

        processor = ArksimTracingProcessor()
        explicit_receiver = MagicMock()
        cv_receiver = MagicMock()

        processor._trace_contexts["trace-1"] = ("explicit-conv", 0, explicit_receiver)
        trace_conversation_id.set("cv-conv")
        trace_turn_id.set(99)
        trace_receiver_ref.set(cv_receiver)

        try:
            span = _make_function_span(trace_id="trace-1")
            processor.on_span_end(span)

            explicit_receiver.submit_tool_calls.assert_called_once()
            cv_receiver.submit_tool_calls.assert_not_called()
            assert (
                explicit_receiver.submit_tool_calls.call_args[0][0] == "explicit-conv"
            )
        finally:
            trace_conversation_id.set(None)
            trace_turn_id.set(None)
            trace_receiver_ref.set(None)

    def test_no_context_anywhere_skips_silently(self) -> None:
        """No explicit context and no contextvars: span is skipped."""
        processor = ArksimTracingProcessor()
        span = _make_function_span(trace_id="no-context")
        processor.on_span_end(span)
        # No crash


class TestContextLifecycle:
    """Test on_trace_end lifecycle."""

    def test_on_trace_end_cleans_up_context(self) -> None:
        """After on_trace_end, spans from that trace are no longer processed."""
        processor = ArksimTracingProcessor()
        receiver = MagicMock()

        processor._trace_contexts["trace-1"] = ("conv-1", 0, receiver)

        trace_mock = SimpleNamespace(trace_id="trace-1")
        processor.on_trace_end(trace_mock)

        span = _make_function_span(trace_id="trace-1")
        processor.on_span_end(span)

        receiver.submit_tool_calls.assert_not_called()

    def test_concurrent_traces_are_isolated(self) -> None:
        """Two traces with different contexts route to correct receivers."""
        processor = ArksimTracingProcessor()
        receiver_a = MagicMock()
        receiver_b = MagicMock()

        processor._trace_contexts["trace-a"] = ("conv-a", 0, receiver_a)
        processor._trace_contexts["trace-b"] = ("conv-b", 1, receiver_b)

        processor.on_span_end(_make_function_span(trace_id="trace-a", name="tool_a"))
        processor.on_span_end(_make_function_span(trace_id="trace-b", name="tool_b"))

        tc_a = receiver_a.submit_tool_calls.call_args[0][2]
        tc_b = receiver_b.submit_tool_calls.call_args[0][2]
        assert tc_a[0].name == "tool_a"
        assert tc_b[0].name == "tool_b"
        assert receiver_a.submit_tool_calls.call_count == 1
        assert receiver_b.submit_tool_calls.call_count == 1


class TestTraceContextManager:
    """Test .trace() context manager (standalone use path)."""

    @pytest.mark.asyncio
    async def test_trace_registers_and_cleans_up_context(self) -> None:
        """Context is registered on entry and removed on exit."""
        processor = ArksimTracingProcessor()

        async with processor.trace("conv-1", turn_id=0):
            assert len(processor._trace_contexts) == 1

        assert len(processor._trace_contexts) == 0

    @pytest.mark.asyncio
    async def test_trace_with_receiver(self) -> None:
        """Receiver passed to .trace() is used for tool call submission."""
        receiver = MagicMock()
        receiver.signal_turn_complete = MagicMock()
        processor = ArksimTracingProcessor()

        async with processor.trace("conv-1", turn_id=0, receiver=receiver):
            trace_id = list(processor._trace_contexts.keys())[0]
            span = _make_function_span(trace_id=trace_id)
            processor.on_span_end(span)

        receiver.submit_tool_calls.assert_called_once()
        receiver.signal_turn_complete.assert_called_once_with("conv-1", 0)

    @pytest.mark.asyncio
    async def test_trace_without_receiver_does_not_crash(self) -> None:
        """No crash when no receiver is provided."""
        processor = ArksimTracingProcessor()

        async with processor.trace("conv-1", turn_id=0):
            trace_id = list(processor._trace_contexts.keys())[0]
            span = _make_function_span(trace_id=trace_id)
            processor.on_span_end(span)

    @pytest.mark.asyncio
    async def test_add_trace_processor_called_once(self) -> None:
        """add_trace_processor is invoked only on the first .trace() call."""
        from unittest.mock import patch

        import arksim.tracing as _tracing_pkg

        # Reset the package-level singleton so this test starts clean
        old = _tracing_pkg._registered_processor
        _tracing_pkg._registered_processor = None

        try:
            processor = ArksimTracingProcessor()

            with patch("agents.tracing.add_trace_processor") as mock_add:
                async with processor.trace("conv-1", turn_id=0):
                    pass
                async with processor.trace("conv-2", turn_id=1):
                    pass

            mock_add.assert_called_once()
        finally:
            _tracing_pkg._registered_processor = old
