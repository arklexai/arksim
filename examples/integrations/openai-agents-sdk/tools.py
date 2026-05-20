# SPDX-License-Identifier: Apache-2.0
"""Standard mock tools for the OpenAI Agents SDK integration example.

Both tools are wrapped with ``@function_tool`` so the SDK emits
``FunctionSpanData`` entries via its tracing pipeline; ``ArksimTracingProcessor``
turns each completed span into a ``ToolCall`` on the active turn. Outputs are
deterministic so the example runs offline (apart from the LLM call) and
produces predictable evaluation results.
"""

from __future__ import annotations

from agents import function_tool


@function_tool
def lookup_order(order_id: str) -> str:
    """Look up an order by ID and return its status."""
    return f"Order {order_id}: shipped, arrives Tuesday."


@function_tool
def book_table(party_size: int, time: str) -> str:
    """Book a restaurant table for the given party size and time."""
    return f"Booked table for {party_size} at {time}."
