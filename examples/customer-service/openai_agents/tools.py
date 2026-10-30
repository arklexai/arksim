# SPDX-License-Identifier: Apache-2.0
"""OpenAI Agents SDK wrappers over the shared tool logic in store_tools.py.

Shared across the Python connector (custom_agent.py), traced agent
(traced_agent.py), and A2A server (a2a_server/) variants. Each
variant imports TOOLS_LIST, AGENT_INSTRUCTIONS, and init_db from here.

Install: pip install openai-agents
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Make the parent directory (customer-service/, where store_tools.py
# lives) importable regardless of how this file gets loaded - as part of
# the openai_agents package (`-m openai_agents.x`), or standalone by
# arksim's module_path loader, which does not treat this file as part of
# a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import store_tools  # noqa: E402
from agents import function_tool  # noqa: E402
from store_tools import AGENT_INSTRUCTIONS, DB_PATH, init_db  # noqa: E402, F401

if TYPE_CHECKING:
    from agents import FunctionTool

lookup_customer = function_tool(store_tools.lookup_customer)
get_order = function_tool(store_tools.get_order)
get_delivery_status = function_tool(store_tools.get_delivery_status)
search_products = function_tool(store_tools.search_products)
list_product_categories = function_tool(store_tools.list_product_categories)
cancel_order = function_tool(store_tools.cancel_order)
send_verification_code = function_tool(store_tools.send_verification_code)
verify_customer = function_tool(store_tools.verify_customer)
get_return_policy = function_tool(store_tools.get_return_policy)
get_product_guide = function_tool(store_tools.get_product_guide)
search_store_knowledge = function_tool(store_tools.search_store_knowledge)
get_knowledge_article = function_tool(store_tools.get_knowledge_article)
get_document = function_tool(store_tools.get_document)

TOOLS_LIST: list[FunctionTool] = [
    lookup_customer,
    get_order,
    get_delivery_status,
    search_products,
    list_product_categories,
    cancel_order,
    send_verification_code,
    verify_customer,
    get_return_policy,
    get_product_guide,
    search_store_knowledge,
    get_knowledge_article,
    get_document,
]
