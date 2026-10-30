# SPDX-License-Identifier: Apache-2.0
"""Protocol-shape checks for the MCP server, over an in-memory transport.

Covers the same ground as google_adk/smoke_client.py without binding a port,
so it can run in CI. Skipped when the MCP or ADK dependencies are absent,
since the root test environment does not install the ADK variant.

    python -m unittest test_mcp_server
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "google_adk"))

# Probe only the third-party dependencies here. Guarding the mcp_server
# import as well would turn a genuine breakage in it - the module these tests
# exist to protect - into a silent skip and a green CI run.
try:
    import google.adk  # noqa: F401
    import mcp  # noqa: F401

    DEPS_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the local environment
    DEPS_AVAILABLE = False

if DEPS_AVAILABLE:
    import mcp_server
    from mcp import Client
    from store_tools import init_db


def texts(result: object) -> list[str]:
    return [b.text for b in result.content if getattr(b, "type", "") == "text"]


@unittest.skipUnless(DEPS_AVAILABLE, "mcp and google-adk are not installed")
class MCPServerTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        init_db()

    async def test_discovery_exposes_safety_annotations(self) -> None:
        async with Client(mcp_server.server) as client:
            tools = {t.name: t for t in (await client.list_tools()).tools}

        self.assertGreaterEqual(len(tools), 16)
        self.assertTrue(tools["cancel_order"].annotations.destructive_hint)
        # An absent annotation must be treated as unsafe by the consumer, per
        # the MCP spec's default of destructiveHint=true when omitted.
        self.assertIsNone(tools["send_verification_code"].annotations)
        self.assertFalse(tools["render_room_preview"].annotations.read_only_hint)
        self.assertTrue(tools["search_store_knowledge"].annotations.read_only_hint)

        # ADK emits an empty input schema for zero-argument tools; MCP requires
        # a type, so the server normalizes it.
        self.assertEqual(
            tools["list_product_categories"].input_schema["type"], "object"
        )

    async def test_search_returns_one_text_block_per_passage(self) -> None:
        async with Client(mcp_server.server) as client:
            result = await client.call_tool("search_products", {"query": "laptop"})

        self.assertGreater(len(texts(result)), 1)
        self.assertTrue(result.structured_content["results"])
        self.assertFalse(result.is_error)

    async def test_structured_content_is_preferred_where_it_exists(self) -> None:
        async with Client(mcp_server.server) as client:
            result = await client.call_tool("list_product_categories", {})

        self.assertEqual(len(result.structured_content["categories"]), 25)
        self.assertEqual(len(texts(result)), 1)

    async def test_excerpt_then_read_by_source_uri(self) -> None:
        async with Client(mcp_server.server) as client:
            found = await client.call_tool(
                "search_store_knowledge", {"query": "price match", "limit": 3}
            )
            top = found.structured_content["results"][0]
            full = await client.call_tool("get_document", {"uri": top["source_uri"]})

        self.assertTrue(top["truncated"])
        self.assertLess(len(top["excerpt"]), top["content_chars"])
        self.assertEqual(len(full.structured_content["content"]), top["content_chars"])

    async def test_pagination_advances(self) -> None:
        async with Client(mcp_server.server) as client:
            first = await client.call_tool(
                "search_store_knowledge", {"query": "return policy", "limit": 3}
            )
            envelope = first.structured_content
            second = await client.call_tool(
                "search_store_knowledge",
                {
                    "query": "return policy",
                    "limit": 3,
                    "offset": envelope["next_offset"],
                },
            )

        self.assertTrue(envelope["has_more"])
        self.assertFalse(
            {r["id"] for r in envelope["results"]}
            & {r["id"] for r in second.structured_content["results"]}
        )

    async def test_non_text_content_is_returned_intact(self) -> None:
        async with Client(mcp_server.server) as client:
            result = await client.call_tool(
                "render_room_preview", {"room": "kitchen", "color": "sage"}
            )

        self.assertIn("image", [getattr(b, "type", "") for b in result.content])

    async def test_error_shapes_are_distinct(self) -> None:
        async with Client(mcp_server.server) as client:
            raised = await client.call_tool(
                "simulate_tool_failure", {"mode": "exception"}
            )
            missing = await client.call_tool(
                "get_knowledge_article", {"article_id": "KB-9999"}
            )
            empty = await client.call_tool(
                "search_store_knowledge", {"query": "zzzznotarealterm"}
            )
            in_band = await client.call_tool(
                "simulate_tool_failure", {"mode": "in_band"}
            )
            unknown = await client.call_tool("no_such_tool", {})

        # A raised tool and a failed lookup are protocol errors.
        self.assertTrue(raised.is_error)
        self.assertTrue(missing.is_error)
        self.assertTrue(unknown.is_error)
        # Finding nothing, and an in-band error payload, are successful calls.
        self.assertFalse(empty.is_error)
        self.assertEqual(empty.structured_content["total"], 0)
        self.assertFalse(in_band.is_error)

    async def test_oversized_result_is_delivered_for_client_side_capping(self) -> None:
        async with Client(mcp_server.server) as client:
            result = await client.call_tool("export_catalog_snapshot", {})

        self.assertGreater(sum(len(t) for t in texts(result)), 50_000)


if __name__ == "__main__":
    unittest.main()
