# SPDX-License-Identifier: Apache-2.0
"""Standard mock tools for the LlamaIndex integration example.

Both tools showcase tool-call capture by ``ArksimLlamaIndexObserver``.
They return deterministic strings so the example runs offline (apart
from the LLM call) and produces predictable evaluation results.
"""

from __future__ import annotations

from llama_index.core.tools import FunctionTool


def _lookup_order(order_id: str) -> str:
    """Look up an order by ID and return its status."""
    return f"Order {order_id}: shipped, arrives Tuesday."


def _book_table(party_size: int, time: str) -> str:
    """Book a restaurant table for the given party size and time."""
    return f"Booked table for {party_size} at {time}."


lookup_order = FunctionTool.from_defaults(_lookup_order, name="lookup_order")
book_table = FunctionTool.from_defaults(_book_table, name="book_table")
