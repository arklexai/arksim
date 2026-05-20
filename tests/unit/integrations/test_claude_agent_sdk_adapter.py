# SPDX-License-Identifier: Apache-2.0
"""Tests for ArksimClaudeHooks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from claude_agent_sdk import HookMatcher

from arksim.simulation_engine.tool_types import ToolCall, ToolCallSource
from arksim.tracing.context import _set_trace_context
from arksim.tracing.integrations.claude_agent_sdk import ArksimClaudeHooks

OnlyCall = Callable[[MagicMock], ToolCall]


def _input(
    *,
    tool_name: str = "lookup_order",
    tool_input: Any = None,  # noqa: ANN401  (PostToolUseHookInput.tool_input is Any-ish)
    tool_response: Any = "shipped",  # noqa: ANN401  (PostToolUseHookInput.tool_response is Any)
) -> dict[str, Any]:
    """Build a synthetic PostToolUseHookInput dict for direct hook invocation."""
    payload: dict[str, Any] = {
        "tool_name": tool_name,
        "tool_response": tool_response,
    }
    if tool_input is not None:
        payload["tool_input"] = tool_input
    return payload


@pytest.mark.asyncio
async def test_happy_path_post_tool_use_submits_tool_call(
    only_call: OnlyCall,
) -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    hooks = ArksimClaudeHooks()

    result = await hooks.post_tool_use(
        _input(
            tool_name="lookup_order",
            tool_input={"order_id": "12345"},
            tool_response="shipped",
        ),
        "t1",
        cast("Any", {}),
    )

    assert result == {}
    tc = only_call(receiver)
    assert tc.id == "t1"
    assert tc.name == "lookup_order"
    assert tc.arguments == {"order_id": "12345"}
    assert tc.result == "shipped"
    assert tc.error is None
    assert tc.source == ToolCallSource.CLAUDE_AGENT_SDK


@pytest.mark.asyncio
async def test_source_field_set_on_every_emission() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    hooks = ArksimClaudeHooks()

    await hooks.post_tool_use(_input(), "t1", cast("Any", {}))
    await hooks.post_tool_use(_input(tool_name="other"), "t2", cast("Any", {}))

    assert receiver.submit_tool_calls.call_count == 2
    for call in receiver.submit_tool_calls.call_args_list:
        tool_call = call.args[2][0]
        assert tool_call.source == ToolCallSource.CLAUDE_AGENT_SDK


@pytest.mark.asyncio
async def test_no_trace_context_silently_drops() -> None:
    receiver = MagicMock()
    # Deliberately do NOT call _set_trace_context.
    hooks = ArksimClaudeHooks()

    await hooks.post_tool_use(_input(), "t1", cast("Any", {}))

    receiver.submit_tool_calls.assert_not_called()


@pytest.mark.asyncio
async def test_missing_tool_input_yields_empty_arguments(
    only_call: OnlyCall,
) -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    hooks = ArksimClaudeHooks()

    # tool_input key absent from the input dict entirely.
    await hooks.post_tool_use(
        {"tool_name": "lookup_order", "tool_response": "ok"},
        "t1",
        cast("Any", {}),
    )

    tc = only_call(receiver)
    assert tc.arguments == {}


@pytest.mark.asyncio
async def test_non_dict_tool_input_wrapped_in_value(only_call: OnlyCall) -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    hooks = ArksimClaudeHooks()

    await hooks.post_tool_use(
        _input(tool_input="raw"),
        "t1",
        cast("Any", {}),
    )

    tc = only_call(receiver)
    assert tc.arguments == {"_value": "raw"}


@pytest.mark.asyncio
async def test_dict_tool_response_serialized_to_string(
    only_call: OnlyCall,
) -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    hooks = ArksimClaudeHooks()

    # PostToolUseHookInput.tool_response is typed Any. When a tool returns
    # a structured payload, we serialize with str() to keep ToolCall.result a
    # printable string consistent with the cross-adapter contract.
    await hooks.post_tool_use(
        _input(tool_response={"status": "ok"}),
        "t1",
        cast("Any", {}),
    )

    tc = only_call(receiver)
    assert tc.result == str({"status": "ok"})


@pytest.mark.asyncio
async def test_hooks_dict_returns_post_tool_use_matcher() -> None:
    """The shape users pass to ClaudeAgentOptions(hooks=...)."""
    hooks = ArksimClaudeHooks()

    out = hooks.hooks_dict()

    assert set(out) == {"PostToolUse"}
    matchers = out["PostToolUse"]
    assert len(matchers) == 1
    matcher = matchers[0]
    assert isinstance(matcher, HookMatcher)
    assert matcher.hooks == [hooks.post_tool_use]
