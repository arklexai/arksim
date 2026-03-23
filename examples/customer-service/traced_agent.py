# SPDX-License-Identifier: Apache-2.0
"""Traced variant of the tool-call example agent.

Same SQLite-backed agent as custom_agent.py, but instead of returning
tool calls in AgentResponse, it pushes them as OTel spans to arksim's
trace receiver. This demonstrates the trace receiver capture path.
custom_agent.py demonstrates the AgentResponse capture path. Both
can coexist since the simulator deduplicates by ID and (name, args).

    Agent executes tools -> OTel spans pushed -> arksim captures -> evaluator scores

Install: pip install openai-agents opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
Auth:    export OPENAI_API_KEY="<your-key>"
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from pathlib import Path

from agents import Agent, Runner, RunResult, function_tool
from agents.items import ToolCallItem, ToolCallOutputItem
from openai.types.responses import ResponseFunctionToolCall
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent

# ── Database setup (shared with custom_agent.py) ──

DB_PATH = Path(__file__).parent / "store.db"


def _init_db() -> None:
    """Create and populate the SQLite database if it doesn't exist."""
    if DB_PATH.exists():
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE customers (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            tier TEXT NOT NULL DEFAULT 'standard'
        );
        CREATE TABLE orders (
            id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            status TEXT NOT NULL,
            total REAL NOT NULL,
            items TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        );
        CREATE TABLE products (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            price REAL NOT NULL,
            category TEXT NOT NULL,
            in_stock INTEGER NOT NULL DEFAULT 1
        );

        INSERT INTO customers VALUES
            ('C-001', 'Alice Johnson', 'alice@example.com', 'premium'),
            ('C-002', 'Bob Smith', 'bob@example.com', 'standard');

        INSERT INTO orders VALUES
            ('ORD-1001', 'C-001', 'shipped', 249.99,
             'Wireless Headphones x1'),
            ('ORD-1002', 'C-001', 'processing', 1299.00,
             'ThinkPad X1 Carbon x1'),
            ('ORD-1003', 'C-002', 'delivered', 59.99,
             'USB-C Hub x1'),
            ('ORD-1004', 'C-002', 'cancelled', 899.00,
             'MacBook Air M4 x1');

        INSERT INTO products VALUES
            ('P-001', 'ThinkPad X1 Carbon', 1299.00, 'laptop', 1),
            ('P-002', 'MacBook Air M4', 1199.00, 'laptop', 1),
            ('P-003', 'Dell XPS 13', 949.00, 'laptop', 0),
            ('P-004', 'Sony WH-1000XM5', 249.99, 'headphones', 1),
            ('P-005', 'USB-C Hub', 59.99, 'accessories', 1),
            ('P-006', 'Mechanical Keyboard', 149.99, 'accessories', 0);
    """)
    conn.commit()
    conn.close()


def _query(sql: str, params: tuple = ()) -> list[dict]:
    """Run a query and return results as dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Tools ──


@function_tool
def lookup_customer(email: str) -> str:
    """Look up a customer by email address. Returns customer profile or error."""
    rows = _query("SELECT * FROM customers WHERE email = ?", (email,))
    if not rows:
        return json.dumps({"error": f"No customer found with email {email}"})
    return json.dumps(rows[0])


@function_tool
def get_order(order_id: str) -> str:
    """Get order details by order ID. Returns order info or error."""
    rows = _query(
        "SELECT o.*, c.name as customer_name, c.email as customer_email "
        "FROM orders o JOIN customers c ON o.customer_id = c.id "
        "WHERE o.id = ?",
        (order_id,),
    )
    if not rows:
        return json.dumps({"error": f"Order {order_id} not found"})
    return json.dumps(rows[0])


@function_tool
def search_products(query: str, max_price: float = 0) -> str:
    """Search product catalog by keyword. Optionally filter by max price."""
    sql = "SELECT * FROM products WHERE name LIKE ? OR category LIKE ?"
    params: list = [f"%{query}%", f"%{query}%"]
    if max_price > 0:
        sql += " AND price <= ?"
        params.append(max_price)
    rows = _query(sql, tuple(params))
    if not rows:
        return json.dumps({"error": f"No products matching '{query}'"})
    return json.dumps(rows)


@function_tool
def cancel_order(order_id: str) -> str:
    """Cancel an order. Only orders with status 'processing' can be cancelled."""
    rows = _query("SELECT * FROM orders WHERE id = ?", (order_id,))
    if not rows:
        return json.dumps({"error": f"Order {order_id} not found"})
    order = rows[0]
    if order["status"] != "processing":
        return json.dumps(
            {
                "error": f"Cannot cancel order {order_id}: "
                f"status is '{order['status']}', only 'processing' orders can be cancelled"
            }
        )
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE orders SET status = 'cancelled' WHERE id = ?", (order_id,))
    conn.commit()
    conn.close()
    return json.dumps({"success": True, "message": f"Order {order_id} cancelled"})


# ── Traced Agent ──

# Default trace receiver endpoint (matches arksim's default port)
_DEFAULT_TRACE_URL = "http://127.0.0.1:4318/v1/traces"


class ToolCallExampleAgent(BaseAgent):
    """Agent that pushes tool calls as OTel spans instead of returning them."""

    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        _init_db()
        self._chat_id = str(uuid.uuid4())
        self._agent = Agent(
            name="assistant",
            instructions=(
                "You are a customer service assistant for an online store. "
                "You have access to tools to look up customers, check orders, "
                "search products, and cancel orders. Use them to help the user. "
                "Always confirm destructive actions like cancellations before proceeding."
            ),
            tools=[lookup_customer, get_order, search_products, cancel_order],
        )
        self._last_result: RunResult | None = None
        self._provider: TracerProvider | None = None
        self._tracer: trace.Tracer | None = None

    async def get_chat_id(self) -> str:
        return self._chat_id

    def _ensure_tracer(self, conversation_id: str) -> None:
        """Lazily init the OTel provider using the simulator's conversation_id."""
        if self._provider is not None:
            return
        self._provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": "traced-tool-call-agent",
                    "arksim.conversation_id": conversation_id,
                }
            )
        )
        self._provider.add_span_processor(
            SimpleSpanProcessor(OTLPSpanExporter(endpoint=_DEFAULT_TRACE_URL))
        )
        self._tracer = self._provider.get_tracer("traced-tool-call-agent")

    async def execute(self, user_query: str, **kwargs: object) -> str:
        metadata = kwargs.get("metadata", {})
        turn_id = metadata.get("turn_id", 0)
        # Use the simulator's chat_id so spans route correctly
        chat_id = metadata.get("chat_id", self._chat_id)
        self._ensure_tracer(chat_id)

        if self._last_result is not None:
            input_list = self._last_result.to_input_list() + [
                {"role": "user", "content": user_query}
            ]
        else:
            input_list = [{"role": "user", "content": user_query}]

        self._last_result = await Runner.run(self._agent, input=input_list)

        # Push tool calls as OTel spans (instead of returning in AgentResponse).
        # Run in a thread executor because the OTel exporter uses synchronous
        # HTTP (requests library) which would block the asyncio event loop and
        # prevent the trace receiver from processing the incoming connection.
        await asyncio.to_thread(self._push_tool_call_spans, self._last_result, turn_id)

        # Return plain string, NOT AgentResponse with tool_calls
        return self._last_result.final_output

    def _push_tool_call_spans(self, result: RunResult, turn_id: int) -> None:
        """Extract tool calls from RunResult and push as OTel spans."""
        outputs: dict[str, str] = {}
        for item in result.new_items:
            if isinstance(item, ToolCallOutputItem):
                raw = item.raw_item
                call_id = (
                    raw.get("call_id", "")
                    if isinstance(raw, dict)
                    else getattr(raw, "call_id", "")
                )
                output = (
                    raw.get("output", "")
                    if isinstance(raw, dict)
                    else getattr(raw, "output", "")
                )
                if isinstance(output, list):
                    output = json.dumps(output)
                outputs[call_id] = str(output)

        for item in result.new_items:
            if not isinstance(item, ToolCallItem):
                continue
            raw = item.raw_item
            if not isinstance(raw, ResponseFunctionToolCall):
                continue

            call_id = raw.call_id
            with self._tracer.start_as_current_span(f"execute_tool {raw.name}") as span:
                span.set_attribute("arksim.turn_id", turn_id)
                span.set_attribute("gen_ai.tool.name", raw.name)
                span.set_attribute("gen_ai.tool.call.id", call_id)
                if raw.arguments:
                    span.set_attribute("gen_ai.tool.call.arguments", raw.arguments)
                tool_result = outputs.get(call_id)
                if tool_result is not None:
                    span.set_attribute("gen_ai.tool.call.result", tool_result)

        # Force flush to ensure spans are sent before arksim collects them
        self._provider.force_flush()

    async def close(self) -> None:
        self._provider.shutdown()
