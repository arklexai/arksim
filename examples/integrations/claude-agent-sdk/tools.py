# SPDX-License-Identifier: Apache-2.0
"""Standard mock tools for the Claude Agent SDK integration example.

Both tools showcase tool-call capture by ``ArksimClaudeHooks``. They are
registered with an in-process SDK MCP server via ``create_sdk_mcp_server``
and return deterministic strings so the example runs offline (apart from
the LLM call) and produces predictable evaluation results.
"""

from __future__ import annotations

from typing import Any

from claude_agent_sdk import tool


@tool(
    "lookup_order",
    "Look up an order by ID and return its status.",
    {"order_id": str},
)
async def lookup_order(args: dict[str, Any]) -> dict[str, Any]:
    text = f"Order {args['order_id']}: shipped, arrives Tuesday."
    return {"content": [{"type": "text", "text": text}]}


@tool(
    "book_table",
    "Book a restaurant table for the given party size and time.",
    {"party_size": int, "time": str},
)
async def book_table(args: dict[str, Any]) -> dict[str, Any]:
    text = f"Booked table for {args['party_size']} at {args['time']}."
    return {"content": [{"type": "text", "text": text}]}
