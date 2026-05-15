# SPDX-License-Identifier: Apache-2.0
"""Tests for ArksimStrandsHookProvider."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from strands.hooks import AfterToolCallEvent, HookRegistry
from strands.types.tools import AgentTool, ToolResult, ToolUse

from arksim.simulation_engine.tool_types import ToolCall, ToolCallSource
from arksim.tracing.context import _clear_trace_context, _set_trace_context
from arksim.tracing.integrations.strands import ArksimStrandsHookProvider


@pytest.fixture(autouse=True)
def _clean_context() -> Iterator[None]:
    _clear_trace_context()
    yield
    _clear_trace_context()


def _tool_use(
    *, name: str = "get_weather", tool_input: object = None, use_id: str = "u1"
) -> ToolUse:
    """Build a ToolUse TypedDict.

    ``input`` is typed ``Any``, so we can stuff a non-dict in for the
    "raw value" test case without bending typing.
    """
    return cast(
        "ToolUse",
        {"name": name, "toolUseId": use_id, "input": tool_input},
    )


def _tool_result(*, output: str = "ok", use_id: str = "u1") -> ToolResult:
    return cast(
        "ToolResult",
        {
            "toolUseId": use_id,
            "status": "success",
            "content": [{"text": output}],
        },
    )


def _make_event(
    *,
    selected_tool: AgentTool | None = None,
    tool_use: ToolUse | None = None,
    result: ToolResult | None = None,
    exception: Exception | None = None,
) -> AfterToolCallEvent:
    return AfterToolCallEvent(
        agent=cast("Any", MagicMock()),
        selected_tool=selected_tool,
        tool_use=tool_use if tool_use is not None else _tool_use(),
        invocation_state={},
        result=result if result is not None else _tool_result(),
        exception=exception,
    )


def _register(provider: ArksimStrandsHookProvider) -> HookRegistry:
    """Run the provider's hook registration and return the populated registry."""
    registry = HookRegistry()
    provider.register_hooks(registry)
    return registry


def _fire(registry: HookRegistry, event: AfterToolCallEvent) -> None:
    registry.invoke_callbacks(event)


def _only_call(receiver: MagicMock) -> ToolCall:
    assert receiver.submit_tool_calls.call_count == 1
    args, _ = receiver.submit_tool_calls.call_args
    conv, turn, tool_calls = args
    assert conv == "conv-1"
    assert turn == 0
    assert len(tool_calls) == 1
    return tool_calls[0]


def test_happy_path_after_tool_call_submits_tool_call() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    registry = _register(ArksimStrandsHookProvider())

    _fire(
        registry,
        _make_event(
            tool_use=_tool_use(
                name="get_weather", tool_input={"city": "NYC"}, use_id="u1"
            ),
            result=_tool_result(output="sunny 75F", use_id="u1"),
        ),
    )

    tc = _only_call(receiver)
    assert tc.id == "u1"
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "NYC"}
    assert tc.result is not None
    assert "sunny 75F" in tc.result
    assert tc.error is None
    assert tc.source == ToolCallSource.STRANDS


def test_exception_populates_error_field() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    registry = _register(ArksimStrandsHookProvider())

    _fire(
        registry,
        _make_event(
            tool_use=_tool_use(name="get_weather", tool_input={"city": "NYC"}),
            exception=ValueError("nope"),
        ),
    )

    tc = _only_call(receiver)
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "NYC"}
    assert tc.error == "ValueError: nope"
    assert tc.result is None
    assert tc.source == ToolCallSource.STRANDS


def test_source_field_set_on_success_and_exception() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    registry = _register(ArksimStrandsHookProvider())

    _fire(registry, _make_event())
    _fire(registry, _make_event(exception=RuntimeError("x")))

    assert receiver.submit_tool_calls.call_count == 2
    for call in receiver.submit_tool_calls.call_args_list:
        tool_call = call.args[2][0]
        assert tool_call.source == ToolCallSource.STRANDS


def test_no_trace_context_silently_drops() -> None:
    receiver = MagicMock()
    # Deliberately do NOT call _set_trace_context.
    registry = _register(ArksimStrandsHookProvider())

    _fire(registry, _make_event())

    receiver.submit_tool_calls.assert_not_called()


def test_missing_tool_use_input_yields_empty_arguments() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    registry = _register(ArksimStrandsHookProvider())

    _fire(registry, _make_event(tool_use=_tool_use(tool_input=None)))

    tc = _only_call(receiver)
    assert tc.arguments == {}


def test_non_dict_tool_use_input_wrapped_in_value() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    registry = _register(ArksimStrandsHookProvider())

    _fire(registry, _make_event(tool_use=_tool_use(tool_input="raw")))

    tc = _only_call(receiver)
    assert tc.arguments == {"_value": "raw"}


def test_multiple_calls_each_submitted() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    registry = _register(ArksimStrandsHookProvider())

    _fire(
        registry,
        _make_event(
            tool_use=_tool_use(name="a", tool_input={"x": 1}, use_id="u1"),
            result=_tool_result(output="r1", use_id="u1"),
        ),
    )
    _fire(
        registry,
        _make_event(
            tool_use=_tool_use(name="b", tool_input={"y": 2}, use_id="u2"),
            result=_tool_result(output="r2", use_id="u2"),
        ),
    )

    assert receiver.submit_tool_calls.call_count == 2
    first = receiver.submit_tool_calls.call_args_list[0].args[2][0]
    second = receiver.submit_tool_calls.call_args_list[1].args[2][0]
    assert first.id == "u1"
    assert first.name == "a"
    assert first.arguments == {"x": 1}
    assert "r1" in (first.result or "")
    assert second.id == "u2"
    assert second.name == "b"
    assert second.arguments == {"y": 2}
    assert "r2" in (second.result or "")
