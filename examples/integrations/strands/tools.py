# SPDX-License-Identifier: Apache-2.0
"""Standard mock tools for the Strands Agents integration example.

Both tools showcase tool-call capture by ``ArksimStrandsHookProvider``.
They return deterministic strings so the example runs offline (apart
from the LLM call) and produces predictable evaluation results.
"""

from __future__ import annotations

from strands.tools import tool


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order by ID and return its status."""
    return f"Order {order_id}: shipped, arrives Tuesday."


@tool
def book_table(party_size: int, time: str) -> str:
    """Book a restaurant table for the given party size and time."""
    return f"Booked table for {party_size} at {time}."
