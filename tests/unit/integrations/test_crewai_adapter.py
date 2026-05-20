# SPDX-License-Identifier: Apache-2.0
"""Tests for ArksimCrewEventListener."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from crewai.events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    crewai_event_bus,
)

from arksim.simulation_engine.tool_types import ToolCall, ToolCallSource
from arksim.tracing.context import _set_trace_context
from arksim.tracing.integrations.crewai import ArksimCrewEventListener

OnlyCall = Callable[[MagicMock], ToolCall]


@pytest.fixture(autouse=True)
def _isolated_bus() -> Iterator[None]:
    """Each test gets a fresh crewai event-bus state.

    ``BaseEventListener.__init__`` registers handlers eagerly into the global
    ``crewai_event_bus`` singleton; without isolation, instances leak across
    tests and a single emit fires every listener ever constructed.
    """
    with crewai_event_bus.scoped_handlers():
        yield


def _emit_and_wait(event: ToolUsageFinishedEvent | ToolUsageErrorEvent) -> None:
    """Emit and block until handlers finish.

    ``crewai_event_bus.emit`` dispatches sync handlers via a
    ``ThreadPoolExecutor`` and returns a ``Future``; tests must wait on it
    or the next assertion races the worker thread.
    """
    future = crewai_event_bus.emit(source=object(), event=event)
    if future is not None:
        future.result(timeout=5.0)


def _finished_event(
    *, tool_name: str = "get_weather", tool_args: object = None, output: object = "ok"
) -> ToolUsageFinishedEvent:
    now = datetime.now(timezone.utc)
    return ToolUsageFinishedEvent(
        timestamp=now,
        tool_name=tool_name,
        tool_args=tool_args if tool_args is not None else {},
        started_at=now,
        finished_at=now,
        output=output,
    )


def _error_event(
    *, tool_name: str = "get_weather", tool_args: object = None, error: object = "boom"
) -> ToolUsageErrorEvent:
    now = datetime.now(timezone.utc)
    return ToolUsageErrorEvent(
        timestamp=now,
        tool_name=tool_name,
        tool_args=tool_args if tool_args is not None else {},
        error=error,
    )


def test_happy_path_finished_event_submits_tool_call(only_call: OnlyCall) -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    ArksimCrewEventListener()

    _emit_and_wait(
        _finished_event(
            tool_name="get_weather",
            tool_args={"city": "NYC"},
            output="sunny 75F",
        )
    )

    tc = only_call(receiver)
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "NYC"}
    assert tc.result == "sunny 75F"
    assert tc.error is None
    assert tc.source == ToolCallSource.CREWAI


def test_error_event_populates_error_field(only_call: OnlyCall) -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    ArksimCrewEventListener()

    _emit_and_wait(
        _error_event(
            tool_name="get_weather",
            tool_args={"city": "NYC"},
            error=ValueError("nope"),
        )
    )

    tc = only_call(receiver)
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "NYC"}
    assert tc.error == "ValueError: nope"
    assert tc.result is None
    assert tc.source == ToolCallSource.CREWAI


def test_no_trace_context_silently_drops() -> None:
    receiver = MagicMock()
    # Deliberately do NOT call _set_trace_context.
    ArksimCrewEventListener()

    _emit_and_wait(_finished_event(tool_args={"city": "NYC"}, output="sunny"))

    receiver.submit_tool_calls.assert_not_called()


def test_missing_tool_args_yields_empty_arguments(only_call: OnlyCall) -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    ArksimCrewEventListener()

    # tool_args is dict | str on the model; empty dict represents the
    # "missing" case for required-field events.
    _emit_and_wait(_finished_event(tool_args={}, output="ok"))

    tc = only_call(receiver)
    assert tc.arguments == {}


def test_non_dict_tool_args_wrapped_in_value(only_call: OnlyCall) -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    ArksimCrewEventListener()

    _emit_and_wait(_finished_event(tool_args="raw-string", output="ok"))

    tc = only_call(receiver)
    assert tc.arguments == {"_value": "raw-string"}


def test_multiple_events_each_submitted() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    ArksimCrewEventListener()

    _emit_and_wait(_finished_event(tool_name="a", tool_args={"x": 1}, output="r1"))
    _emit_and_wait(_finished_event(tool_name="b", tool_args={"y": 2}, output="r2"))

    assert receiver.submit_tool_calls.call_count == 2
    first = receiver.submit_tool_calls.call_args_list[0].args[2][0]
    second = receiver.submit_tool_calls.call_args_list[1].args[2][0]
    assert first.name == "a"
    assert first.arguments == {"x": 1}
    assert first.result == "r1"
    assert second.name == "b"
    assert second.arguments == {"y": 2}
    assert second.result == "r2"


def test_source_field_set_on_both_success_and_error() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    ArksimCrewEventListener()

    _emit_and_wait(_finished_event())
    _emit_and_wait(_error_event())

    assert receiver.submit_tool_calls.call_count == 2
    for call in receiver.submit_tool_calls.call_args_list:
        tool_call = call.args[2][0]
        assert tool_call.source == ToolCallSource.CREWAI
