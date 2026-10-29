# SPDX-License-Identifier: Apache-2.0
"""Turns store_tools return values into MCP CallToolResult shapes.

Kept free of ADK and arksim imports so the protocol shapes can be tested on
their own. Real MCP servers do not all answer in one shape, and a connector
has to normalize every one of them, so this module deliberately spreads the
tools across the shapes the spec allows:

- structured_content plus a text fallback (list_product_categories)
- one text block per passage (the search tools)
- a single prose block (the policy and guide tools)
- a non-text block (render_room_preview returns an image)
- a protocol-level error, isError=True (a failed lookup, or a raised tool)
- an in-band error payload on an otherwise successful call (an empty search)

The last two are distinct on purpose: an empty result set is a successful
call that found nothing, while a lookup of a document that does not exist is
a failed call.
"""

from __future__ import annotations

import json
from typing import Any

import mcp.types as mcp_types

# 1x1 transparent PNG. The bytes do not matter; what matters is that a
# non-text block reaches the connector, which must skip it for grounding.
_PIXEL_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

# Tools whose result is a list of passages, emitted one text block each.
SEARCH_TOOLS = frozenset({"search_store_knowledge", "search_products"})

# Tools whose result is a single document; a miss is a failed call.
DOCUMENT_TOOLS = frozenset({"get_knowledge_article", "get_document"})

# Tools that return prose rather than JSON.
PROSE_TOOLS = frozenset({"get_return_policy", "get_product_guide"})


def _text(value: str) -> mcp_types.TextContent:
    return mcp_types.TextContent(type="text", text=value)


def error_result(message: str) -> mcp_types.CallToolResult:
    """A protocol-level failure: isError=True, nothing to ground on."""
    return mcp_types.CallToolResult(content=[_text(message)], isError=True)


def _passage(row: dict[str, Any]) -> str:
    """Render one search hit as a citable passage."""
    header = (
        f"[{row.get('id', '?')}] {row.get('title') or row.get('name') or ''}".strip()
    )
    facets = " | ".join(str(row[key]) for key in ("topic", "category") if row.get(key))
    if facets:
        header = f"{header} ({facets})"
    lines = [header]
    if row.get("source_uri"):
        lines.append(str(row["source_uri"]))
    body = row.get("excerpt") or row.get("description") or ""
    if row.get("price") is not None:
        stock = "in stock" if row.get("in_stock") else "out of stock"
        lines.append(f"Price: {row['price']} ({stock})")
    if body:
        lines.append(str(body))
    if row.get("truncated"):
        lines.append(
            f"[excerpt of {row['content_chars']} characters - "
            f"call get_document with the source URI above for the full text]"
        )
    return "\n".join(lines)


def build_result(tool_name: str, raw: str) -> mcp_types.CallToolResult:
    """Shape one tool's return value into a CallToolResult."""
    if tool_name in PROSE_TOOLS:
        return mcp_types.CallToolResult(content=[_text(raw)])

    if tool_name == "render_room_preview":
        payload = json.loads(raw)
        return mcp_types.CallToolResult(
            content=[
                mcp_types.ImageContent(
                    type="image", data=_PIXEL_PNG, mimeType="image/png"
                ),
                _text(
                    f"Preview of {payload['room']} in {payload['color']}. "
                    f"{payload['note']}"
                ),
            ],
            structuredContent=payload,
        )

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return mcp_types.CallToolResult(content=[_text(raw)])

    if tool_name == "list_product_categories":
        summary = ", ".join(
            f"{row['category']} ({row['product_count']})" for row in payload
        )
        return mcp_types.CallToolResult(
            content=[_text(f"{len(payload)} categories: {summary}")],
            structuredContent={"categories": payload},
        )

    if tool_name in DOCUMENT_TOOLS:
        if isinstance(payload, dict) and "error" in payload:
            return error_result(payload["error"])
        title = payload.get("title", payload.get("id", ""))
        uri = payload.get("source_uri", "")
        return mcp_types.CallToolResult(
            content=[_text(f"{title}\n{uri}\n\n{payload.get('content', '')}")],
            structuredContent=payload,
        )

    if tool_name in SEARCH_TOOLS:
        # An empty result set is a successful call that found nothing, so it
        # stays in band rather than becoming a protocol error.
        if isinstance(payload, dict) and "error" in payload:
            return mcp_types.CallToolResult(
                content=[_text(payload["error"])],
                structuredContent={"results": [], "total": 0, "has_more": False},
            )
        blocks = [_passage(row) for row in payload["results"]]
        if payload.get("has_more"):
            blocks.append(
                f"{payload['total']} results total; showing "
                f"{payload['offset']}-{payload['offset'] + len(payload['results'])}. "
                f"Call again with offset={payload['next_offset']} for more."
            )
        return mcp_types.CallToolResult(
            content=[_text(b) for b in blocks],
            structuredContent=payload,
        )

    if isinstance(payload, dict) and "error" in payload:
        return mcp_types.CallToolResult(
            content=[_text(payload["error"])],
            structuredContent=payload,
        )

    return mcp_types.CallToolResult(
        content=[_text(raw)],
        structuredContent=payload if isinstance(payload, dict) else {"items": payload},
    )
