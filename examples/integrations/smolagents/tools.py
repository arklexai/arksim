# SPDX-License-Identifier: Apache-2.0
"""Standard mock tools for the Smolagents integration example.

Both tools showcase tool-call capture by ``ArksimSmolagentsCallback``.
They return deterministic strings so the example runs offline (apart
from the LLM call) and produces predictable evaluation results.
"""

from __future__ import annotations

from smolagents import tool


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order by ID and return its status.

    Args:
        order_id: The order identifier to look up.
    """
    return f"Order {order_id}: shipped, arrives Tuesday."


@tool
def book_table(party_size: int, time: str) -> str:
    """Book a restaurant table for the given party size and time.

    Args:
        party_size: Number of people to seat.
        time: The time to book for, for example "7pm".
    """
    return f"Booked table for {party_size} at {time}."
