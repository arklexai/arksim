# SPDX-License-Identifier: Apache-2.0
"""Smoke-test client for mcp_server.py.

Connects as a generic MCP client (no ADK or arksim dependency) and asserts the
protocol behaviours the platform's MCP connector has to handle: tool discovery
with safety annotations, multi-block and structured results, excerpt-then-read
retrieval, pagination, non-text content, and both error shapes.

Exits non-zero on the first failed check, so it works as a CI protocol check
rather than only as a demo.

Run (in a separate terminal from mcp_server.py):
    python -m google_adk.mcp_server      # terminal 1
    python google_adk/smoke_client.py    # terminal 2
"""

from __future__ import annotations

import asyncio
import os
import sys

from mcp import Client

MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:9997/mcp")

_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}{f' - {detail}' if detail else ''}")
    if not condition:
        _failures.append(label)


def texts(result: object) -> list[str]:
    return [b.text for b in result.content if getattr(b, "type", "") == "text"]


async def main() -> int:
    async with Client(MCP_URL) as client:
        listing = await client.list_tools()
        tools = {tool.name: tool for tool in listing.tools}
        print(f"Discovered {len(tools)} tools\n")

        # ── Discovery and safety annotations ──
        read_only = [
            name
            for name, tool in tools.items()
            if tool.annotations and tool.annotations.read_only_hint
        ]
        check("discovery returns the full tool set", len(tools) >= 16, f"{len(tools)}")
        check(
            "cancel_order is annotated destructive",
            bool(tools["cancel_order"].annotations.destructive_hint),
        )
        check(
            "send_verification_code carries no annotations (fail-safe default)",
            tools["send_verification_code"].annotations is None,
        )
        check(
            "render_room_preview is not read-only",
            not tools["render_room_preview"].annotations.read_only_hint,
        )
        check(
            "read-only lookup tools are discoverable",
            len(read_only) >= 10,
            f"{len(read_only)} read-only",
        )

        # ── Multi-block and structured results ──
        result = await client.call_tool("search_products", {"query": "laptop"})
        check(
            "search_products returns one text block per passage",
            len(texts(result)) > 1,
            f"{len(texts(result))} blocks",
        )
        check(
            "search_products carries structured content",
            bool(
                result.structured_content and result.structured_content.get("results")
            ),
        )

        result = await client.call_tool("list_product_categories", {})
        check(
            "list_product_categories prefers structuredContent",
            bool(result.structured_content.get("categories")),
            f"{len(result.structured_content['categories'])} categories",
        )

        # ── Excerpt, then read the full document ──
        result = await client.call_tool(
            "search_store_knowledge", {"query": "price match", "limit": 3}
        )
        top = result.structured_content["results"][0]
        check("knowledge search truncates into excerpts", top["truncated"])
        check(
            "excerpt is materially shorter than the source",
            len(top["excerpt"]) < top["content_chars"],
            f"{len(top['excerpt'])} of {top['content_chars']} chars",
        )

        result = await client.call_tool("get_document", {"uri": top["source_uri"]})
        full = result.structured_content["content"]
        check(
            "get_document returns the full text by source URI",
            len(full) == top["content_chars"],
            f"{len(full)} chars",
        )

        # ── Pagination ──
        page1 = await client.call_tool(
            "search_store_knowledge", {"query": "return policy", "limit": 3}
        )
        envelope = page1.structured_content
        check("search paginates", envelope["has_more"], f"{envelope['total']} total")
        page2 = await client.call_tool(
            "search_store_knowledge",
            {"query": "return policy", "limit": 3, "offset": envelope["next_offset"]},
        )
        ids1 = [r["id"] for r in envelope["results"]]
        ids2 = [r["id"] for r in page2.structured_content["results"]]
        check("second page returns different rows", not set(ids1) & set(ids2))

        # ── Non-text content ──
        result = await client.call_tool(
            "render_room_preview", {"room": "kitchen", "color": "sage"}
        )
        kinds = [getattr(b, "type", "") for b in result.content]
        check(
            "render_room_preview returns a non-text block", "image" in kinds, str(kinds)
        )

        # ── Both error shapes ──
        result = await client.call_tool("simulate_tool_failure", {"mode": "exception"})
        check("a raising tool surfaces as isError", bool(result.is_error))

        result = await client.call_tool(
            "get_knowledge_article", {"article_id": "KB-9999"}
        )
        check("a missing document is a failed call", bool(result.is_error))

        result = await client.call_tool(
            "search_store_knowledge", {"query": "zzzznotarealterm"}
        )
        check(
            "an empty search is a successful call, not a protocol error",
            not result.is_error and result.structured_content["total"] == 0,
        )

        result = await client.call_tool("simulate_tool_failure", {"mode": "in_band"})
        check("an in-band error payload stays a successful call", not result.is_error)

        # ── Size cap ──
        result = await client.call_tool("export_catalog_snapshot", {})
        size = sum(len(t) for t in texts(result))
        check(
            "export_catalog_snapshot exceeds a typical size cap",
            size > 50_000,
            f"{size} chars",
        )

    print()
    if _failures:
        print(f"{len(_failures)} check(s) failed: {', '.join(_failures)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
