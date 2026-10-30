# SPDX-License-Identifier: Apache-2.0
"""Google ADK tool wrappers over the shared tool logic in store_tools.py.

ADK_TOOLS is the customer-service agent's own tool set. MCP_TOOLS adds the
fixture tools that exist only to give the platform's MCP connector something
to exercise its normalization, size-cap, and error paths against - the two
sets differ for the same reason a partner's MCP surface rarely matches their
agent's tool list exactly.

Also builds TOOL_ANNOTATIONS, a plain lookup table - not an ADK concept
(ADK's FunctionTool has no native readOnlyHint/destructiveHint attribute),
this is bookkeeping mcp_server.py will read later to annotate the tools
it exposes over MCP.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the parent directory (customer-service/, where store_tools.py
# lives) importable regardless of how this file gets loaded - as part of
# the google_adk package (`-m google_adk.x`), or standalone by arksim's
# module_path loader, which does not treat this file as part of a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import store_tools  # noqa: E402
from google.adk.tools import FunctionTool  # noqa: E402

ADK_TOOLS: list[FunctionTool] = [
    FunctionTool(func=store_tools.lookup_customer),
    FunctionTool(func=store_tools.get_order),
    FunctionTool(func=store_tools.get_delivery_status),
    FunctionTool(func=store_tools.search_products),
    FunctionTool(func=store_tools.list_product_categories),
    FunctionTool(func=store_tools.cancel_order),
    FunctionTool(func=store_tools.send_verification_code),
    FunctionTool(func=store_tools.verify_customer),
    FunctionTool(func=store_tools.get_return_policy),
    FunctionTool(func=store_tools.get_product_guide),
    FunctionTool(func=store_tools.search_store_knowledge),
    FunctionTool(func=store_tools.get_knowledge_article),
    FunctionTool(func=store_tools.get_document),
]

# Exposed over MCP but withheld from the agent: an oversized export, a
# non-text (image) result, and a tool that fails on demand.
MCP_FIXTURE_TOOLS: list[FunctionTool] = [
    FunctionTool(func=store_tools.export_catalog_snapshot),
    FunctionTool(func=store_tools.render_room_preview),
    FunctionTool(func=store_tools.simulate_tool_failure),
]

MCP_TOOLS: list[FunctionTool] = [*ADK_TOOLS, *MCP_FIXTURE_TOOLS]

# Per-tool MCP annotations. A tool with destructive_hint True can never be
# allowlisted downstream; a tool that omits the hint entirely is treated as
# destructive by default (fail-safe) - send_verification_code below is a
# deliberate test case for that fallback path.
TOOL_ANNOTATIONS: dict[str, dict[str, bool]] = {
    "lookup_customer": {"readOnlyHint": True},
    "get_order": {"readOnlyHint": True},
    "get_delivery_status": {"readOnlyHint": True},
    "search_products": {"readOnlyHint": True},
    "list_product_categories": {"readOnlyHint": True},
    "cancel_order": {"readOnlyHint": False, "destructiveHint": True},
    # send_verification_code: deliberately no entry here.
    "verify_customer": {"readOnlyHint": True},
    "get_return_policy": {"readOnlyHint": True},
    "get_product_guide": {"readOnlyHint": True},
    "search_store_knowledge": {"readOnlyHint": True},
    "get_knowledge_article": {"readOnlyHint": True},
    "get_document": {"readOnlyHint": True},
    "export_catalog_snapshot": {"readOnlyHint": True},
    # Generative image output: not
    # read-only, so it is never auto-allowlisted as a lookup tool.
    "render_room_preview": {"readOnlyHint": False, "destructiveHint": False},
    "simulate_tool_failure": {"readOnlyHint": True},
}
