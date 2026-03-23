# SPDX-License-Identifier: Apache-2.0
"""Tests for the customer-service example agent's _extract_tool_calls logic.

Requires: pip install openai-agents
Skipped automatically when the SDK is not installed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest

agents = pytest.importorskip("agents", reason="openai-agents SDK not installed")

from agents import Agent  # noqa: E402
from agents.items import ToolCallItem, ToolCallOutputItem  # noqa: E402
from openai.types.responses import ResponseFunctionToolCall  # noqa: E402

# Load the example module dynamically (directory name has a hyphen)
_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "customer-service"
    / "custom_agent.py"
)
_spec = importlib.util.spec_from_file_location("tool_calls_example", _EXAMPLE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ToolCallExampleAgent = _mod.ToolCallExampleAgent

# Convenience: stub Agent for constructing SDK item dataclasses
_STUB_AGENT = Agent(name="stub", instructions="stub")


def _make_function_call(call_id: str, name: str, arguments: str) -> ToolCallItem:
    raw = ResponseFunctionToolCall(
        call_id=call_id,
        name=name,
        arguments=arguments,
        type="function_call",
    )
    return ToolCallItem(agent=_STUB_AGENT, raw_item=raw)


def _make_function_output(call_id: str, output: str) -> ToolCallOutputItem:
    raw = {
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
    }
    return ToolCallOutputItem(agent=_STUB_AGENT, raw_item=raw, output=output)


class TestExtractToolCalls:
    def test_single_tool_call_with_output(self) -> None:
        result = MagicMock()
        result.new_items = [
            _make_function_call("c1", "get_order", '{"order_id": "ORD-1001"}'),
            _make_function_output("c1", '{"status": "shipped", "total": 249.99}'),
        ]

        tool_calls = ToolCallExampleAgent._extract_tool_calls(result)
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "c1"
        assert tool_calls[0].name == "get_order"
        assert tool_calls[0].arguments == {"order_id": "ORD-1001"}
        assert tool_calls[0].result == '{"status": "shipped", "total": 249.99}'

    def test_multiple_tool_calls(self) -> None:
        result = MagicMock()
        result.new_items = [
            _make_function_call(
                "c1", "lookup_customer", '{"email": "alice@example.com"}'
            ),
            _make_function_output("c1", '{"id": "C-001", "name": "Alice"}'),
            _make_function_call("c2", "get_order", '{"order_id": "ORD-1001"}'),
            _make_function_output("c2", '{"status": "shipped"}'),
        ]

        tool_calls = ToolCallExampleAgent._extract_tool_calls(result)
        assert len(tool_calls) == 2
        assert tool_calls[0].name == "lookup_customer"
        assert tool_calls[1].name == "get_order"
        assert tool_calls[1].result == '{"status": "shipped"}'

    def test_tool_call_without_output(self) -> None:
        result = MagicMock()
        result.new_items = [
            _make_function_call("c1", "get_order", '{"order_id": "ORD-1001"}'),
        ]

        tool_calls = ToolCallExampleAgent._extract_tool_calls(result)
        assert len(tool_calls) == 1
        assert tool_calls[0].result is None

    def test_empty_arguments(self) -> None:
        result = MagicMock()
        result.new_items = [
            _make_function_call("c1", "some_tool", ""),
            _make_function_output("c1", "ok"),
        ]

        tool_calls = ToolCallExampleAgent._extract_tool_calls(result)
        assert len(tool_calls) == 1
        assert tool_calls[0].arguments == {}

    def test_no_items_returns_empty(self) -> None:
        result = MagicMock()
        result.new_items = []

        tool_calls = ToolCallExampleAgent._extract_tool_calls(result)
        assert tool_calls == []

    def test_non_function_items_skipped(self) -> None:
        """Non-function tool call items (e.g. computer actions) are skipped."""
        result = MagicMock()
        non_function_item = ToolCallItem(
            agent=_STUB_AGENT,
            raw_item={"type": "computer_call", "id": "x"},
        )
        result.new_items = [non_function_item]

        tool_calls = ToolCallExampleAgent._extract_tool_calls(result)
        assert tool_calls == []

    def test_output_with_list_value(self) -> None:
        """List outputs are JSON-serialized."""
        result = MagicMock()
        raw_output = {
            "type": "function_call_output",
            "call_id": "c1",
            "output": [{"name": "item1"}, {"name": "item2"}],
        }
        result.new_items = [
            _make_function_call("c1", "search_products", '{"query": "laptop"}'),
            ToolCallOutputItem(
                agent=_STUB_AGENT,
                raw_item=raw_output,
                output=raw_output["output"],
            ),
        ]

        tool_calls = ToolCallExampleAgent._extract_tool_calls(result)
        assert len(tool_calls) == 1
        assert tool_calls[0].result == '[{"name": "item1"}, {"name": "item2"}]'
