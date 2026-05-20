# SPDX-License-Identifier: Apache-2.0
"""Tests for ArksimSmolagentsCallback."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

from smolagents.memory import ActionStep, PlanningStep
from smolagents.memory import ToolCall as SmolToolCall
from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import Timing

from arksim.simulation_engine.tool_types import ToolCall, ToolCallSource
from arksim.tracing.context import _set_trace_context
from arksim.tracing.integrations.smolagents import ArksimSmolagentsCallback

OnlyCall = Callable[[MagicMock], ToolCall]


def _action_step(
    *,
    tool_calls: list[SmolToolCall] | None = None,
    observations: str | None = None,
) -> ActionStep:
    return ActionStep(
        step_number=1,
        timing=Timing(start_time=0.0, end_time=1.0),
        tool_calls=tool_calls,
        observations=observations,
    )


def _smol_tool_call(
    *, name: str = "get_weather", arguments: object = None, call_id: str = "call_1"
) -> SmolToolCall:
    return SmolToolCall(name=name, arguments=arguments, id=call_id)


def _planning_step() -> PlanningStep:
    return PlanningStep(
        model_input_messages=[],
        model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content=""),
        plan="plan text",
        timing=Timing(start_time=0.0, end_time=1.0),
    )


def test_happy_path_action_step_submits_tool_call(only_call: OnlyCall) -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    observer = ArksimSmolagentsCallback()

    observer(
        _action_step(
            tool_calls=[
                _smol_tool_call(
                    name="get_weather",
                    arguments={"city": "NYC"},
                    call_id="call_1",
                )
            ],
            observations="sunny 75F",
        )
    )

    tc = only_call(receiver)
    assert tc.id == "call_1"
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "NYC"}
    assert tc.result == "sunny 75F"
    assert tc.error is None
    assert tc.source == ToolCallSource.SMOLAGENTS


def test_source_field_set_on_every_emission() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    observer = ArksimSmolagentsCallback()

    observer(
        _action_step(
            tool_calls=[_smol_tool_call(arguments={"x": 1})],
            observations="r1",
        )
    )
    observer(
        _action_step(
            tool_calls=[_smol_tool_call(arguments={"y": 2}, call_id="call_2")],
            observations="r2",
        )
    )

    assert receiver.submit_tool_calls.call_count == 2
    for call in receiver.submit_tool_calls.call_args_list:
        tool_call = call.args[2][0]
        assert tool_call.source == ToolCallSource.SMOLAGENTS


def test_no_trace_context_silently_drops() -> None:
    receiver = MagicMock()
    # Deliberately do NOT call _set_trace_context.
    observer = ArksimSmolagentsCallback()

    observer(
        _action_step(
            tool_calls=[_smol_tool_call(arguments={"x": 1})],
            observations="r",
        )
    )

    receiver.submit_tool_calls.assert_not_called()


def test_non_action_step_ignored() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    observer = ArksimSmolagentsCallback()

    observer(_planning_step())

    receiver.submit_tool_calls.assert_not_called()


def test_multiple_tool_calls_in_one_step_each_submitted() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    observer = ArksimSmolagentsCallback()

    observer(
        _action_step(
            tool_calls=[
                _smol_tool_call(name="a", arguments={"x": 1}, call_id="c1"),
                _smol_tool_call(name="b", arguments={"y": 2}, call_id="c2"),
            ],
            observations="shared result",
        )
    )

    assert receiver.submit_tool_calls.call_count == 2
    first = receiver.submit_tool_calls.call_args_list[0].args[2][0]
    second = receiver.submit_tool_calls.call_args_list[1].args[2][0]
    assert first.id == "c1"
    assert first.name == "a"
    assert first.arguments == {"x": 1}
    assert first.result == "shared result"
    assert second.id == "c2"
    assert second.name == "b"
    assert second.arguments == {"y": 2}
    assert second.result == "shared result"


def test_none_arguments_yields_empty_dict(only_call: OnlyCall) -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    observer = ArksimSmolagentsCallback()

    observer(
        _action_step(
            tool_calls=[_smol_tool_call(arguments=None)],
            observations="ok",
        )
    )

    tc = only_call(receiver)
    assert tc.arguments == {}


def test_non_dict_arguments_wrapped_in_value(only_call: OnlyCall) -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    observer = ArksimSmolagentsCallback()

    observer(
        _action_step(
            tool_calls=[_smol_tool_call(arguments="raw")],
            observations="ok",
        )
    )

    tc = only_call(receiver)
    assert tc.arguments == {"_value": "raw"}


def test_empty_observation_preserved_as_empty_string(only_call: OnlyCall) -> None:
    """observations='' should round-trip as result='', distinguishable from None."""
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    observer = ArksimSmolagentsCallback()

    observer(
        _action_step(
            tool_calls=[_smol_tool_call(arguments={"x": 1})],
            observations="",
        )
    )

    tc = only_call(receiver)
    assert tc.result == ""
