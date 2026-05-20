# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for SDK tracing-adapter unit tests.

Every adapter test exercises the same trace-context plumbing and asserts
the same shape of receiver call. Centralizing the fixtures here keeps
that plumbing in one place: the autouse ``_clean_context`` fixture
resets ``trace_conversation_id`` / ``trace_turn_id`` / ``trace_receiver_ref``
between tests, and ``only_call`` returns the single ``ToolCall`` that
the adapter submitted (or fails the test if zero or many were submitted).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from unittest.mock import MagicMock

import pytest

from arksim.simulation_engine.tool_types import ToolCall
from arksim.tracing.context import _clear_trace_context


@pytest.fixture(autouse=True)
def _clean_context() -> Iterator[None]:
    """Reset the trace-context contextvars before and after each test."""
    _clear_trace_context()
    yield
    _clear_trace_context()


@pytest.fixture
def only_call() -> Callable[[MagicMock], ToolCall]:
    """Return a helper that asserts the receiver got exactly one submission.

    Use as ``tc = only_call(receiver)`` in tests where the adapter is
    expected to emit a single ``ToolCall``. Validates routing context
    matched ``conv-1`` / turn ``0`` along the way.
    """

    def _only_call(receiver: MagicMock) -> ToolCall:
        assert receiver.submit_tool_calls.call_count == 1
        args, _ = receiver.submit_tool_calls.call_args
        conv, turn, tool_calls = args
        assert conv == "conv-1"
        assert turn == 0
        assert len(tool_calls) == 1
        return tool_calls[0]

    return _only_call
