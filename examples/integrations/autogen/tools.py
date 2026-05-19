# SPDX-License-Identifier: Apache-2.0
"""Standard mock tools for the AutoGen integration example.

Each tool is wrapped in an OpenTelemetry span that follows the OTel GenAI
semantic conventions (``gen_ai.tool.name``, ``gen_ai.tool.call.arguments``,
``gen_ai.tool.call.result``). Arksim's OTLP receiver converts these spans
into ``ToolCall`` records on the appropriate turn.

AutoGen does not emit ``gen_ai.tool.*`` spans natively for direct
``AssistantAgent.on_messages`` calls, so the wrapper is what produces the
captured tool calls.

Outputs are deterministic so the example runs offline (apart from the LLM
call) and produces predictable evaluation results.
"""

from __future__ import annotations

import json

from opentelemetry import trace

_tracer = trace.get_tracer("arksim.examples.autogen")


def lookup_order(order_id: str) -> str:
    """Look up an order by ID and return its status."""
    arguments = {"order_id": order_id}
    with _tracer.start_as_current_span("execute_tool lookup_order") as span:
        span.set_attribute("gen_ai.tool.name", "lookup_order")
        span.set_attribute("gen_ai.tool.call.arguments", json.dumps(arguments))
        result = f"Order {order_id}: shipped, arrives Tuesday."
        span.set_attribute("gen_ai.tool.call.result", result)
        return result


def book_table(party_size: int, time: str) -> str:
    """Book a restaurant table for the given party size and time."""
    arguments = {"party_size": party_size, "time": time}
    with _tracer.start_as_current_span("execute_tool book_table") as span:
        span.set_attribute("gen_ai.tool.name", "book_table")
        span.set_attribute("gen_ai.tool.call.arguments", json.dumps(arguments))
        result = f"Booked table for {party_size} at {time}."
        span.set_attribute("gen_ai.tool.call.result", result)
        return result
