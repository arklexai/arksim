# SPDX-License-Identifier: Apache-2.0
"""Tests for arksim.tracing.openai_agents.ArksimTracingProcessor."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("agents", reason="openai-agents SDK not installed")

from arksim.tracing.openai_agents import ArksimTracingProcessor


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


class TestOnSpanEnd:
    """Test ArksimTracingProcessor.on_span_end behavior."""

    def test_function_span_submits_tool_call(self) -> None:
        """A FunctionSpanData span with registered context submits a ToolCall."""
        processor = ArksimTracingProcessor()
        receiver = MagicMock()

        processor.register_context("trace-1", "conv-1", 0, receiver)
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

        processor.register_context("trace-1", "conv-1", 0, receiver)
        span = _make_non_function_span()
        processor.on_span_end(span)

        receiver.submit_tool_calls.assert_not_called()

    def test_unregistered_trace_id_is_skipped(self) -> None:
        """Spans from traces without registered context are ignored."""
        processor = ArksimTracingProcessor()
        receiver = MagicMock()

        # Register for trace-1, but span comes from trace-unknown
        processor.register_context("trace-1", "conv-1", 0, receiver)
        span = _make_function_span(trace_id="trace-unknown")
        processor.on_span_end(span)

        receiver.submit_tool_calls.assert_not_called()

    def test_no_receiver_does_not_crash(self) -> None:
        """When receiver is None, on_span_end completes without error."""
        processor = ArksimTracingProcessor()

        processor.register_context("trace-1", "conv-1", 0, receiver=None)
        span = _make_function_span()
        # Should not raise
        processor.on_span_end(span)

    def test_error_span_captures_message(self) -> None:
        """Span with an error dict captures the error message."""
        processor = ArksimTracingProcessor()
        receiver = MagicMock()

        processor.register_context("trace-1", "conv-1", 0, receiver)
        span = _make_function_span(error={"message": "connection timeout"})
        processor.on_span_end(span)

        tc_list = receiver.submit_tool_calls.call_args[0][2]
        assert tc_list[0].error == "connection timeout"

    def test_error_span_missing_message_key(self) -> None:
        """Span with error dict lacking 'message' falls back to 'unknown error'."""
        processor = ArksimTracingProcessor()
        receiver = MagicMock()

        processor.register_context("trace-1", "conv-1", 0, receiver)
        span = _make_function_span(error={"code": 500})
        processor.on_span_end(span)

        tc_list = receiver.submit_tool_calls.call_args[0][2]
        assert tc_list[0].error == "unknown error"

    def test_malformed_input_json_yields_empty_arguments(self) -> None:
        """Invalid JSON in span_data.input produces empty arguments dict."""
        processor = ArksimTracingProcessor()
        receiver = MagicMock()

        processor.register_context("trace-1", "conv-1", 0, receiver)
        span = _make_function_span(input_json="not valid json")
        processor.on_span_end(span)

        tc_list = receiver.submit_tool_calls.call_args[0][2]
        assert tc_list[0].arguments == {}

    def test_non_dict_input_json_yields_empty_arguments(self) -> None:
        """JSON that parses to a non-dict (e.g. list) produces empty arguments."""
        processor = ArksimTracingProcessor()
        receiver = MagicMock()

        processor.register_context("trace-1", "conv-1", 0, receiver)
        span = _make_function_span(input_json='["a", "b"]')
        processor.on_span_end(span)

        tc_list = receiver.submit_tool_calls.call_args[0][2]
        assert tc_list[0].arguments == {}

    def test_dict_output_is_json_serialized(self) -> None:
        """Non-string output (dict) is JSON-serialized in the ToolCall result."""
        processor = ArksimTracingProcessor()
        receiver = MagicMock()

        processor.register_context("trace-1", "conv-1", 0, receiver)
        # Use a SimpleNamespace with dict output to simulate non-string
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

        processor.register_context("trace-1", "conv-1", 0, receiver)
        span = _make_function_span(input_json=None, output=None)
        processor.on_span_end(span)

        tc_list = receiver.submit_tool_calls.call_args[0][2]
        assert tc_list[0].arguments == {}
        assert tc_list[0].result is None


class TestContextLifecycle:
    """Test register_context / on_trace_end lifecycle."""

    def test_on_trace_end_cleans_up_context(self) -> None:
        """After on_trace_end, spans from that trace are no longer processed."""
        processor = ArksimTracingProcessor()
        receiver = MagicMock()

        processor.register_context("trace-1", "conv-1", 0, receiver)

        # Simulate trace end
        trace_mock = SimpleNamespace(trace_id="trace-1")
        processor.on_trace_end(trace_mock)

        # Subsequent spans should be skipped
        span = _make_function_span(trace_id="trace-1")
        processor.on_span_end(span)

        receiver.submit_tool_calls.assert_not_called()

    def test_concurrent_traces_are_isolated(self) -> None:
        """Two traces with different contexts route to correct receivers."""
        processor = ArksimTracingProcessor()
        receiver_a = MagicMock()
        receiver_b = MagicMock()

        processor.register_context("trace-a", "conv-a", 0, receiver_a)
        processor.register_context("trace-b", "conv-b", 1, receiver_b)

        processor.on_span_end(_make_function_span(trace_id="trace-a", name="tool_a"))
        processor.on_span_end(_make_function_span(trace_id="trace-b", name="tool_b"))

        tc_a = receiver_a.submit_tool_calls.call_args[0][2]
        tc_b = receiver_b.submit_tool_calls.call_args[0][2]
        assert tc_a[0].name == "tool_a"
        assert tc_b[0].name == "tool_b"
        # Each receiver called exactly once
        assert receiver_a.submit_tool_calls.call_count == 1
        assert receiver_b.submit_tool_calls.call_count == 1


class TestTraceContextManager:
    """Test ArksimTracingProcessor.trace() context manager behavior."""

    @pytest.mark.asyncio
    async def test_trace_registers_and_cleans_up_context(self) -> None:
        """Context is registered on entry and removed on exit."""
        processor = ArksimTracingProcessor()

        async with processor.trace("conv-1", turn_id=0):
            assert len(processor._trace_contexts) == 1

        assert len(processor._trace_contexts) == 0

    @pytest.mark.asyncio
    async def test_init_receiver_used_when_no_per_call_override(self) -> None:
        """Receiver passed at __init__ is used when no per-call override given."""
        receiver = MagicMock()
        processor = ArksimTracingProcessor(receiver=receiver)

        async with processor.trace("conv-1", turn_id=0):
            trace_id = list(processor._trace_contexts.keys())[0]
            span = _make_function_span(trace_id=trace_id)
            processor.on_span_end(span)

        receiver.submit_tool_calls.assert_called_once()

    @pytest.mark.asyncio
    async def test_per_call_receiver_overrides_init_receiver(self) -> None:
        """Receiver passed to .trace() overrides the one passed at __init__."""
        init_receiver = MagicMock()
        call_receiver = MagicMock()
        processor = ArksimTracingProcessor(receiver=init_receiver)

        async with processor.trace("conv-1", turn_id=0, receiver=call_receiver):
            trace_id = list(processor._trace_contexts.keys())[0]
            span = _make_function_span(trace_id=trace_id)
            processor.on_span_end(span)

        call_receiver.submit_tool_calls.assert_called_once()
        init_receiver.submit_tool_calls.assert_not_called()

    @pytest.mark.asyncio
    async def test_trace_without_receiver_does_not_crash(self) -> None:
        """No crash when neither init nor per-call receiver is provided."""
        processor = ArksimTracingProcessor()

        async with processor.trace("conv-1", turn_id=0):
            trace_id = list(processor._trace_contexts.keys())[0]
            span = _make_function_span(trace_id=trace_id)
            # Should not raise even with no receiver
            processor.on_span_end(span)

    @pytest.mark.asyncio
    async def test_set_trace_processors_called_once(self) -> None:
        """set_trace_processors is invoked only on the first .trace() call."""
        from unittest.mock import patch

        processor = ArksimTracingProcessor()

        with patch("agents.tracing.set_trace_processors") as mock_set:
            async with processor.trace("conv-1", turn_id=0):
                pass
            async with processor.trace("conv-2", turn_id=1):
                pass

        mock_set.assert_called_once()
