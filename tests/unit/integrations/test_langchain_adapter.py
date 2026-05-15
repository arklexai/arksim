# SPDX-License-Identifier: Apache-2.0
"""Tests for ArksimLangChainHandler split-event correlation."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler

from arksim.simulation_engine.tool_types import ToolCall, ToolCallSource
from arksim.tracing.context import _clear_trace_context, _set_trace_context
from arksim.tracing.integrations.langchain import ArksimLangChainHandler


@pytest.fixture(autouse=True)
def _clean_context() -> Iterator[None]:
    _clear_trace_context()
    yield
    _clear_trace_context()


def _serialized(name: str = "get_weather") -> dict[str, object]:
    return {"name": name}


def _only_call(receiver: MagicMock) -> ToolCall:
    assert receiver.submit_tool_calls.call_count == 1
    args, _ = receiver.submit_tool_calls.call_args
    conv, turn, tool_calls = args
    assert conv == "conv-1"
    assert turn == 0
    assert len(tool_calls) == 1
    return tool_calls[0]


def test_happy_path_sync_produces_single_tool_call() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    handler = ArksimLangChainHandler()
    run_id = uuid4()

    handler.on_tool_start(_serialized("get_weather"), '{"city": "NYC"}', run_id=run_id)
    handler.on_tool_end("sunny 75F", run_id=run_id)

    tc = _only_call(receiver)
    assert tc.id == str(run_id)
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "NYC"}
    assert tc.result == "sunny 75F"
    assert tc.error is None
    assert tc.source == ToolCallSource.LANGCHAIN


def test_error_path_populates_error_field() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    handler = ArksimLangChainHandler()
    run_id = uuid4()

    handler.on_tool_start(_serialized("get_weather"), '{"city": "NYC"}', run_id=run_id)
    handler.on_tool_error(ValueError("nope"), run_id=run_id)

    tc = _only_call(receiver)
    assert tc.id == str(run_id)
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "NYC"}
    assert tc.error == "nope"
    assert tc.result is None
    assert tc.source == ToolCallSource.LANGCHAIN


def test_unmatched_end_logs_debug_and_does_not_submit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    handler = ArksimLangChainHandler()
    stray = UUID("00000000-0000-0000-0000-000000000abc")

    with caplog.at_level("DEBUG", logger="arksim.tracing.integrations.langchain"):
        handler.on_tool_end("late", run_id=stray)

    receiver.submit_tool_calls.assert_not_called()
    assert any(
        "unmatched on_tool_end" in r.getMessage() and str(stray) in r.getMessage()
        for r in caplog.records
    )


def test_no_trace_context_silently_drops() -> None:
    receiver = MagicMock()
    # Deliberately do NOT call _set_trace_context.
    handler = ArksimLangChainHandler()
    run_id = uuid4()

    handler.on_tool_start(_serialized(), '{"city": "NYC"}', run_id=run_id)
    handler.on_tool_end("sunny", run_id=run_id)

    receiver.submit_tool_calls.assert_not_called()


def test_dual_inheritance_sync_and_async() -> None:
    handler = ArksimLangChainHandler()
    assert isinstance(handler, BaseCallbackHandler)
    assert isinstance(handler, AsyncCallbackHandler)


def test_multiple_overlapping_calls_correlate_correctly() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    handler = ArksimLangChainHandler()
    r1 = uuid4()
    r2 = uuid4()

    handler.on_tool_start(_serialized("a"), '{"x": 1}', run_id=r1)
    handler.on_tool_start(_serialized("b"), '{"y": 2}', run_id=r2)
    handler.on_tool_end("res_a", run_id=r1)
    handler.on_tool_end("res_b", run_id=r2)

    assert receiver.submit_tool_calls.call_count == 2
    calls = receiver.submit_tool_calls.call_args_list
    first = calls[0].args[2][0]
    second = calls[1].args[2][0]
    assert first.id == str(r1)
    assert first.name == "a"
    assert first.arguments == {"x": 1}
    assert first.result == "res_a"
    assert second.id == str(r2)
    assert second.name == "b"
    assert second.arguments == {"y": 2}
    assert second.result == "res_b"


def test_non_json_input_str_wrapped_in_value() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    handler = ArksimLangChainHandler()
    run_id = uuid4()

    handler.on_tool_start(_serialized("t"), "not-json", run_id=run_id)
    handler.on_tool_end("ok", run_id=run_id)

    tc = _only_call(receiver)
    assert tc.arguments == {"_value": "not-json"}


def test_empty_input_str_yields_empty_arguments() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    handler = ArksimLangChainHandler()
    run_id = uuid4()

    handler.on_tool_start(_serialized("t"), "", run_id=run_id)
    handler.on_tool_end("ok", run_id=run_id)

    tc = _only_call(receiver)
    assert tc.arguments == {}
