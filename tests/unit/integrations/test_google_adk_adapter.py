# SPDX-License-Identifier: Apache-2.0
"""Tests for ArksimADKPlugin."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from arksim.simulation_engine.tool_types import ToolCall, ToolCallSource
from arksim.tracing.context import _set_trace_context
from arksim.tracing.integrations.google_adk import ArksimADKPlugin

OnlyCall = Callable[[MagicMock], ToolCall]


def _tool(name: str = "lookup_order") -> Any:  # noqa: ANN401
    tool = MagicMock()
    tool.name = name
    return tool


def _tool_context(invocation_id: str | None = "inv-1") -> Any:  # noqa: ANN401
    ctx = MagicMock(spec=["invocation_id"])
    if invocation_id is not None:
        ctx.invocation_id = invocation_id
    else:
        # Remove the attribute so getattr falls back to default.
        del ctx.invocation_id
    return ctx


@pytest.mark.asyncio
async def test_happy_path_after_tool_callback_submits_tool_call(
    only_call: OnlyCall,
) -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    plugin = ArksimADKPlugin()

    out = await plugin.after_tool_callback(
        tool=_tool("lookup_order"),
        tool_args={"order_id": "12345"},
        tool_context=_tool_context("inv-1"),
        result={"status": "shipped"},
    )

    assert out is None
    tc = only_call(receiver)
    assert tc.id == "inv-1"
    assert tc.name == "lookup_order"
    assert tc.arguments == {"order_id": "12345"}
    assert tc.result == str({"status": "shipped"})
    assert tc.error is None
    assert tc.source == ToolCallSource.GOOGLE_ADK


@pytest.mark.asyncio
async def test_source_field_set_on_every_emission() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    plugin = ArksimADKPlugin()

    await plugin.after_tool_callback(
        tool=_tool("a"),
        tool_args={},
        tool_context=_tool_context("inv-1"),
        result={"ok": True},
    )
    await plugin.after_tool_callback(
        tool=_tool("b"),
        tool_args={},
        tool_context=_tool_context("inv-2"),
        result={"ok": True},
    )

    assert receiver.submit_tool_calls.call_count == 2
    for call in receiver.submit_tool_calls.call_args_list:
        tool_call = call.args[2][0]
        assert tool_call.source == ToolCallSource.GOOGLE_ADK


@pytest.mark.asyncio
async def test_no_trace_context_silently_drops() -> None:
    receiver = MagicMock()
    # Deliberately do NOT call _set_trace_context.
    plugin = ArksimADKPlugin()

    await plugin.after_tool_callback(
        tool=_tool("lookup_order"),
        tool_args={"order_id": "12345"},
        tool_context=_tool_context("inv-1"),
        result={"status": "shipped"},
    )

    receiver.submit_tool_calls.assert_not_called()


@pytest.mark.asyncio
async def test_none_tool_args_yields_empty_arguments(only_call: OnlyCall) -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    plugin = ArksimADKPlugin()

    await plugin.after_tool_callback(
        tool=_tool("lookup_order"),
        tool_args=cast("Any", None),
        tool_context=_tool_context("inv-1"),
        result={"ok": True},
    )

    tc = only_call(receiver)
    assert tc.arguments == {}


@pytest.mark.asyncio
async def test_non_dict_tool_args_wrapped_in_value(only_call: OnlyCall) -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    plugin = ArksimADKPlugin()

    await plugin.after_tool_callback(
        tool=_tool("lookup_order"),
        tool_args=cast("Any", "raw"),
        tool_context=_tool_context("inv-1"),
        result={"ok": True},
    )

    tc = only_call(receiver)
    assert tc.arguments == {"_value": "raw"}


@pytest.mark.asyncio
async def test_return_value_is_none_to_preserve_original_result() -> None:
    """ADK BasePlugin.after_tool_callback can return Optional[dict].

    Returning a non-None value short-circuits remaining plugins and replaces
    the tool result. Tracing must be observation-only: always return None so
    the agent sees the original tool output.
    """
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    plugin = ArksimADKPlugin()

    out = await plugin.after_tool_callback(
        tool=_tool("lookup_order"),
        tool_args={"order_id": "12345"},
        tool_context=_tool_context("inv-1"),
        result={"status": "shipped"},
    )

    assert out is None


@pytest.mark.asyncio
async def test_missing_invocation_id_falls_back_to_empty_string(
    only_call: OnlyCall,
) -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    plugin = ArksimADKPlugin()

    await plugin.after_tool_callback(
        tool=_tool("lookup_order"),
        tool_args={"order_id": "12345"},
        tool_context=_tool_context(None),
        result={"status": "shipped"},
    )

    tc = only_call(receiver)
    assert tc.id == ""
