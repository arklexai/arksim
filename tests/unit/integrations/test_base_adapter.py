# SPDX-License-Identifier: Apache-2.0
"""Tests for BaseTracingAdapter._submit behavior."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

from arksim.simulation_engine.tool_types import ToolCall, ToolCallSource
from arksim.tracing.context import _clear_trace_context, _set_trace_context
from arksim.tracing.integrations._base import BaseTracingAdapter


@pytest.fixture(autouse=True)
def _clean_context() -> Iterator[None]:
    _clear_trace_context()
    yield
    _clear_trace_context()


def _tool_call(name: str = "f") -> ToolCall:
    return ToolCall(id="t1", name=name, arguments={}, source=ToolCallSource.LANGCHAIN)


def test_submit_no_op_when_no_routing_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    receiver = MagicMock()
    adapter = BaseTracingAdapter()
    with caplog.at_level("DEBUG", logger="arksim.tracing.integrations._base"):
        adapter._submit(_tool_call(name="missing"))
    receiver.submit_tool_calls.assert_not_called()
    assert any(
        "no routing context" in r.getMessage() and "missing" in r.getMessage()
        for r in caplog.records
    )


def test_submit_swallows_receiver_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Observability must never break the observed.

    If the receiver raises, ``_submit`` logs and absorbs the error rather
    than propagating it back into the SDK callback that invoked the adapter.
    """
    receiver = MagicMock()
    receiver.submit_tool_calls.side_effect = RuntimeError("boom")
    _set_trace_context("conv-1", 0, receiver=receiver)
    adapter = BaseTracingAdapter()

    with caplog.at_level("ERROR", logger="arksim.tracing.integrations._base"):
        adapter._submit(_tool_call(name="oops"))  # must not raise

    receiver.submit_tool_calls.assert_called_once()
    assert any(
        "receiver raised" in r.getMessage()
        and "oops" in r.getMessage()
        and "conv-1" in r.getMessage()
        for r in caplog.records
    )


def test_submit_debug_logs_when_ids_set_but_no_receiver(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _set_trace_context("conv-1", 0, receiver=None)
    adapter = BaseTracingAdapter()
    with caplog.at_level("DEBUG", logger="arksim.tracing.integrations._base"):
        adapter._submit(_tool_call(name="foo"))
    assert any(
        "no receiver" in r.getMessage() and "foo" in r.getMessage()
        for r in caplog.records
    )


def test_submit_calls_submit_tool_calls_plural_with_list() -> None:
    receiver = MagicMock()
    _set_trace_context("conv-1", 0, receiver=receiver)
    adapter = BaseTracingAdapter()
    tc = _tool_call(name="bar")
    adapter._submit(tc)
    receiver.submit_tool_calls.assert_called_once_with("conv-1", 0, [tc])
