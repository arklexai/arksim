# SPDX-License-Identifier: Apache-2.0
"""Tests for ArksimLiveKitHandler."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from livekit.agents.llm.chat_context import FunctionCall, FunctionCallOutput
from livekit.agents.voice.events import FunctionToolsExecutedEvent

from arksim.simulation_engine.tool_types import ToolCall, ToolCallSource
from arksim.tracing.context import _clear_trace_context, _set_trace_context
from arksim.tracing.integrations.livekit import ArksimLiveKitHandler


@pytest.fixture(autouse=True)
def _clean_context() -> Iterator[None]:
    _clear_trace_context()
    yield
    _clear_trace_context()


def _call(
    *,
    name: str = "get_weather",
    arguments: str = '{"city": "NYC"}',
    call_id: str = "call_1",
) -> FunctionCall:
    return FunctionCall(
        call_id=call_id,
        arguments=arguments,
        name=name,
    )


def _output(
    *,
    call_id: str = "call_1",
    name: str = "get_weather",
    output: str = "sunny 75F",
    is_error: bool = False,
) -> FunctionCallOutput:
    return FunctionCallOutput(
        call_id=call_id,
        name=name,
        output=output,
        is_error=is_error,
    )


def _event(
    *,
    function_calls: list[FunctionCall] | None = None,
    function_call_outputs: list[FunctionCallOutput | None] | None = None,
) -> FunctionToolsExecutedEvent:
    if function_calls is None:
        function_calls = [_call()]
    if function_call_outputs is None:
        function_call_outputs = [_output()]
    return FunctionToolsExecutedEvent(
        function_calls=function_calls,
        function_call_outputs=function_call_outputs,
    )


def _calls_received(receiver: MagicMock) -> list[ToolCall]:
    """Flatten all tool_calls passed to receiver.submit_tool_calls."""
    collected: list[ToolCall] = []
    for call in receiver.submit_tool_calls.call_args_list:
        conv, turn, tool_calls = call.args
        assert conv == "conv-1"
        assert turn == 0
        collected.extend(tool_calls)
    return collected


def test_happy_path_single_call_submits_tool_call() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    handler = ArksimLiveKitHandler()

    handler.on_function_tools_executed(_event())

    calls = _calls_received(receiver)
    assert len(calls) == 1
    tc = calls[0]
    assert tc.id == "call_1"
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "NYC"}
    assert tc.result == "sunny 75F"
    assert tc.error is None
    assert tc.source == ToolCallSource.LIVEKIT


def test_is_error_output_populates_error_field() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    handler = ArksimLiveKitHandler()

    handler.on_function_tools_executed(
        _event(
            function_calls=[_call()],
            function_call_outputs=[_output(output="ValueError: nope", is_error=True)],
        )
    )

    calls = _calls_received(receiver)
    assert len(calls) == 1
    tc = calls[0]
    assert tc.name == "get_weather"
    assert tc.error == "ValueError: nope"
    assert tc.result is None
    assert tc.source == ToolCallSource.LIVEKIT


def test_source_field_set_on_success_and_error() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    handler = ArksimLiveKitHandler()

    handler.on_function_tools_executed(_event())
    handler.on_function_tools_executed(
        _event(
            function_calls=[_call()],
            function_call_outputs=[_output(output="boom", is_error=True)],
        )
    )

    calls = _calls_received(receiver)
    assert len(calls) == 2
    for tc in calls:
        assert tc.source == ToolCallSource.LIVEKIT


def test_no_trace_context_silently_drops() -> None:
    receiver = MagicMock()
    # Deliberately do NOT call _set_trace_context.
    handler = ArksimLiveKitHandler()

    handler.on_function_tools_executed(_event())

    receiver.submit_tool_calls.assert_not_called()


def test_batch_event_with_multiple_parallel_calls() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    handler = ArksimLiveKitHandler()

    handler.on_function_tools_executed(
        _event(
            function_calls=[
                _call(name="a", arguments='{"x": 1}', call_id="c1"),
                _call(name="b", arguments='{"y": 2}', call_id="c2"),
            ],
            function_call_outputs=[
                _output(call_id="c1", name="a", output="r1"),
                _output(call_id="c2", name="b", output="r2"),
            ],
        )
    )

    calls = _calls_received(receiver)
    assert len(calls) == 2
    assert calls[0].id == "c1"
    assert calls[0].name == "a"
    assert calls[0].arguments == {"x": 1}
    assert calls[0].result == "r1"
    assert calls[1].id == "c2"
    assert calls[1].name == "b"
    assert calls[1].arguments == {"y": 2}
    assert calls[1].result == "r2"


def test_none_output_leaves_result_and_error_unset() -> None:
    """LiveKit sets output to None when a tool raises StopResponse."""
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    handler = ArksimLiveKitHandler()

    handler.on_function_tools_executed(
        _event(
            function_calls=[_call()],
            function_call_outputs=[None],
        )
    )

    calls = _calls_received(receiver)
    assert len(calls) == 1
    tc = calls[0]
    assert tc.name == "get_weather"
    assert tc.result is None
    assert tc.error is None
    assert tc.source == ToolCallSource.LIVEKIT


def test_non_json_arguments_wrapped_in_value() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    handler = ArksimLiveKitHandler()

    handler.on_function_tools_executed(
        _event(
            function_calls=[_call(arguments="raw-string-not-json")],
            function_call_outputs=[_output()],
        )
    )

    calls = _calls_received(receiver)
    assert len(calls) == 1
    assert calls[0].arguments == {"_value": "raw-string-not-json"}


def test_non_string_output_coerced_to_str() -> None:
    """Defensive coercion: if LiveKit ever emits a non-string error/result
    object, the adapter must stringify rather than raise ``ValidationError``
    inside the event loop. LiveKit's current model types ``output`` as
    ``str``; ``model_construct`` bypasses that to exercise the coercion.
    """
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    handler = ArksimLiveKitHandler()

    structured = {"code": 500, "msg": "boom"}
    bad_output = FunctionCallOutput.model_construct(
        call_id="call_1",
        name="get_weather",
        output=structured,  # type: ignore[arg-type]
        is_error=True,
    )
    event = FunctionToolsExecutedEvent(
        function_calls=[_call()],
        function_call_outputs=[bad_output],
    )

    handler.on_function_tools_executed(event)

    calls = _calls_received(receiver)
    assert len(calls) == 1
    tc = calls[0]
    assert tc.error == str(structured)
    assert tc.result is None


def test_attach_to_session_subscribes_event() -> None:
    """attach_to must call session.on with the verified event name."""
    handler = ArksimLiveKitHandler()
    session: Any = MagicMock()

    handler.attach_to(session)

    session.on.assert_called_once_with(
        "function_tools_executed", handler.on_function_tools_executed
    )
