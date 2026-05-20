# SPDX-License-Identifier: Apache-2.0
"""Tests for ArksimLlamaIndexObserver split-event correlation."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from llama_index.core.agent.workflow import ToolCall as LIToolCall
from llama_index.core.agent.workflow import ToolCallResult as LIToolCallResult
from llama_index.core.tools.types import ToolOutput

from arksim.simulation_engine.tool_types import ToolCall, ToolCallSource
from arksim.tracing.context import _clear_trace_context, _set_trace_context
from arksim.tracing.integrations.llamaindex import ArksimLlamaIndexObserver


@pytest.fixture(autouse=True)
def _clean_context() -> Iterator[None]:
    _clear_trace_context()
    yield
    _clear_trace_context()


def _start(
    *,
    tool_name: str = "get_weather",
    tool_kwargs: dict[str, object] | None = None,
    tool_id: str = "t1",
) -> LIToolCall:
    return LIToolCall(
        tool_name=tool_name,
        tool_kwargs=tool_kwargs if tool_kwargs is not None else {"city": "NYC"},
        tool_id=tool_id,
    )


def _result(
    *,
    tool_name: str = "get_weather",
    tool_kwargs: dict[str, object] | None = None,
    tool_id: str = "t1",
    output: str = "sunny 75F",
    is_error: bool = False,
    exception: Exception | None = None,
) -> LIToolCallResult:
    tool_output = ToolOutput(
        tool_name=tool_name,
        content=output,
        raw_input=tool_kwargs if tool_kwargs is not None else {"city": "NYC"},
        raw_output=output,
        is_error=is_error,
        exception=exception,
    )
    return LIToolCallResult(
        tool_name=tool_name,
        tool_kwargs=tool_kwargs if tool_kwargs is not None else {"city": "NYC"},
        tool_id=tool_id,
        tool_output=tool_output,
        return_direct=False,
    )


def _only_call(receiver: MagicMock) -> ToolCall:
    assert receiver.submit_tool_calls.call_count == 1
    args, _ = receiver.submit_tool_calls.call_args
    conv, turn, tool_calls = args
    assert conv == "conv-1"
    assert turn == 0
    assert len(tool_calls) == 1
    return tool_calls[0]


def test_happy_path_split_events_produce_single_tool_call() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    observer = ArksimLlamaIndexObserver()

    observer.observe(_start())
    observer.observe(_result())

    tc = _only_call(receiver)
    assert tc.id == "t1"
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "NYC"}
    assert tc.result == "sunny 75F"
    assert tc.error is None
    assert tc.source == ToolCallSource.LLAMAINDEX


def test_error_path_with_exception_formats_type_and_message() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    observer = ArksimLlamaIndexObserver()

    observer.observe(_start())
    observer.observe(
        _result(
            output="",
            is_error=True,
            exception=ValueError("nope"),
        )
    )

    tc = _only_call(receiver)
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "NYC"}
    assert tc.error == "ValueError: nope"
    assert tc.result is None
    assert tc.source == ToolCallSource.LLAMAINDEX


def test_is_error_without_exception_falls_back_to_content() -> None:
    """is_error=True with no exception means error message is in content."""
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    observer = ArksimLlamaIndexObserver()

    observer.observe(_start())
    observer.observe(
        _result(
            output="Tool get_weather not found.",
            is_error=True,
            exception=None,
        )
    )

    tc = _only_call(receiver)
    assert tc.error == "Tool get_weather not found."
    assert tc.result is None
    assert tc.source == ToolCallSource.LLAMAINDEX


def test_source_field_set_on_success_and_error() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    observer = ArksimLlamaIndexObserver()

    observer.observe(_start(tool_id="t1"))
    observer.observe(_result(tool_id="t1"))
    observer.observe(_start(tool_id="t2"))
    observer.observe(
        _result(
            tool_id="t2",
            is_error=True,
            exception=RuntimeError("x"),
        )
    )

    assert receiver.submit_tool_calls.call_count == 2
    for call in receiver.submit_tool_calls.call_args_list:
        tool_call = call.args[2][0]
        assert tool_call.source == ToolCallSource.LLAMAINDEX


def test_no_trace_context_silently_drops() -> None:
    receiver = MagicMock()
    # Deliberately do NOT call _set_trace_context.
    observer = ArksimLlamaIndexObserver()

    observer.observe(_start())
    observer.observe(_result())

    receiver.submit_tool_calls.assert_not_called()


def test_unmatched_result_logs_and_does_not_submit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    observer = ArksimLlamaIndexObserver()

    with caplog.at_level("DEBUG", logger="arksim.tracing.integrations.llamaindex"):
        observer.observe(_result(tool_id="unknown"))

    receiver.submit_tool_calls.assert_not_called()
    assert "unmatched" in caplog.text.lower()


def test_multiple_concurrent_tool_calls_correlate_by_tool_id() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    observer = ArksimLlamaIndexObserver()

    observer.observe(_start(tool_name="a", tool_kwargs={"x": 1}, tool_id="t1"))
    observer.observe(_start(tool_name="b", tool_kwargs={"y": 2}, tool_id="t2"))
    observer.observe(
        _result(tool_name="b", tool_kwargs={"y": 2}, tool_id="t2", output="r2")
    )
    observer.observe(
        _result(tool_name="a", tool_kwargs={"x": 1}, tool_id="t1", output="r1")
    )

    assert receiver.submit_tool_calls.call_count == 2
    first = receiver.submit_tool_calls.call_args_list[0].args[2][0]
    second = receiver.submit_tool_calls.call_args_list[1].args[2][0]
    # results arrive in the order they were observed (b then a)
    assert first.id == "t2"
    assert first.name == "b"
    assert first.arguments == {"y": 2}
    assert first.result == "r2"
    assert second.id == "t1"
    assert second.name == "a"
    assert second.arguments == {"x": 1}
    assert second.result == "r1"


def test_empty_tool_kwargs_yields_empty_arguments() -> None:
    """LlamaIndex always types tool_kwargs as dict, so {} is the no-arg case."""
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    observer = ArksimLlamaIndexObserver()

    observer.observe(_start(tool_kwargs={}))
    observer.observe(_result(tool_kwargs={}))

    tc = _only_call(receiver)
    assert tc.arguments == {}


def test_non_tool_event_ignored() -> None:
    """Non-ToolCall/ToolCallResult events from the workflow stream are no-ops."""
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    observer = ArksimLlamaIndexObserver()

    # An AgentStream/AgentOutput would be an Event subclass that isn't a tool event.
    observer.observe(object())

    receiver.submit_tool_calls.assert_not_called()


def test_non_string_content_coerced_to_str() -> None:
    """Defensive coercion: if a custom ToolOutput returns a non-string
    content, the adapter must stringify rather than raise ValidationError
    when constructing ToolCall.result.
    """
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    observer = ArksimLlamaIndexObserver()

    structured = {"status": "ok", "rows": 3}

    class _StructuredToolOutput(ToolOutput):
        @property
        def content(self) -> object:  # type: ignore[override]
            return structured

    custom = _StructuredToolOutput(
        tool_name="get_weather",
        content="placeholder",
        raw_input={"city": "NYC"},
        raw_output=structured,
        is_error=False,
        exception=None,
    )
    result = LIToolCallResult(
        tool_name="get_weather",
        tool_kwargs={"city": "NYC"},
        tool_id="t1",
        tool_output=custom,
        return_direct=False,
    )

    observer.observe(_start())
    observer.observe(result)

    tc = _only_call(receiver)
    assert tc.result == str(structured)
    assert tc.error is None
