# SPDX-License-Identifier: Apache-2.0
"""Example agent with SQLite-backed tool calls for ArkSim tool call evaluation.

Demonstrates how a custom agent can return AgentResponse with ToolCall
data so the evaluator can run the tool_call_behavior_failure metric.

Tools are backed by a local SQLite database with realistic data and
error paths, allowing the evaluation layer to be validated against
non-trivial tool call scenarios.

Install: pip install openai-agents
Auth:    export OPENAI_API_KEY="<your-key>"
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

from agents import Agent, Runner, RunResult, function_tool

from arksim import AgentConfig, AgentResponse
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.tracing.openai import extract_tool_calls

# ── Database setup ──

DB_PATH = Path(__file__).parent / "store.db"


def _init_db() -> None:
    """Create the SQLite database or reset mutable state.

    On first run, creates all tables and inserts seed data.
    On subsequent runs, resets order statuses so each simulation
    starts from a clean state (prevents cross-conversation side
    effects when num_conversations_per_scenario > 1).
    """
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        conn.executescript("""
            UPDATE orders SET status = 'shipped' WHERE id = 'ORD-1001';
            UPDATE orders SET status = 'processing' WHERE id = 'ORD-1002';
            UPDATE orders SET status = 'delivered' WHERE id = 'ORD-1003';
            UPDATE orders SET status = 'cancelled' WHERE id = 'ORD-1004';
        """)
        conn.close()
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

        CREATE TABLE verification_codes (
            customer_id TEXT PRIMARY KEY,
            code TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        );
        INSERT INTO verification_codes VALUES
            ('C-001', '123456'),
            ('C-002', '789012');
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


@function_tool
def send_verification_code(email: str) -> str:
    """Send a verification code to the customer's email. Must be called before verify_customer."""
    rows = _query("SELECT c.id FROM customers c WHERE c.email = ?", (email,))
    if not rows:
        return json.dumps({"error": f"No customer found with email {email}"})
    customer_id = rows[0]["id"]
    code_rows = _query(
        "SELECT code FROM verification_codes WHERE customer_id = ?", (customer_id,)
    )
    if not code_rows:
        return json.dumps({"error": "No verification code on file for this customer"})
    # In a real system this would send an email; here we just confirm it was sent
    return json.dumps(
        {"success": True, "message": f"Verification code sent to {email}"}
    )


@function_tool
def verify_customer(email: str, code: str) -> str:
    """Verify a customer's identity using the code sent to their email."""
    rows = _query("SELECT c.id FROM customers c WHERE c.email = ?", (email,))
    if not rows:
        return json.dumps({"error": f"No customer found with email {email}"})
    customer_id = rows[0]["id"]
    code_rows = _query(
        "SELECT code FROM verification_codes WHERE customer_id = ?", (customer_id,)
    )
    if not code_rows:
        return json.dumps({"error": "No verification code on file"})
    if code_rows[0]["code"] != code:
        return json.dumps({"error": "Invalid verification code"})
    return json.dumps(
        {"success": True, "message": f"Customer {email} verified successfully"}
    )


# ── Agent ──


class ToolCallExampleAgent(BaseAgent):
    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        _init_db()
        self._chat_id = str(uuid.uuid4())
        self._agent = Agent(
            name="assistant",
            instructions=(
                "You are a customer service assistant for an online store. "
                "You have access to tools to look up customers, check orders, "
                "search products, cancel orders, and verify customer identity. "
                "Use them to help the user. "
                "Always confirm destructive actions like cancellations before proceeding. "
                "For sensitive operations, verify the customer's identity first by "
                "sending a verification code and then verifying it."
            ),
            tools=[
                lookup_customer,
                get_order,
                search_products,
                cancel_order,
                send_verification_code,
                verify_customer,
            ],
        )
        self._last_result: RunResult | None = None

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> AgentResponse:
        if self._last_result is not None:
            input_list = self._last_result.to_input_list() + [
                {"role": "user", "content": user_query}
            ]
        else:
            input_list = [{"role": "user", "content": user_query}]

        self._last_result = await Runner.run(self._agent, input=input_list)

        return AgentResponse(
            content=self._last_result.final_output,
            tool_calls=extract_tool_calls(self._last_result),
        )
