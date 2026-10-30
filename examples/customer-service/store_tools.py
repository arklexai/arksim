# SPDX-License-Identifier: Apache-2.0
"""Shared tool logic for the customer-service example.

Framework-agnostic: plain functions with no SDK decorators, so both
the OpenAI Agents SDK (tools.py) and Google ADK (google_adk/adk_tools.py)
variants can wrap them independently.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from seed_data import (
    EXTRA_CUSTOMER_COUNT,
    EXTRA_ORDER_COUNT,
    EXTRA_PRODUCT_COUNT,
    FULFILLMENT_LEAD_DAYS,
    KNOWLEDGE_ARTICLE_COUNT,
    generated_customers,
    generated_orders,
    generated_products,
    generated_verification_codes,
    knowledge_articles,
)

# ── Database setup ──

DB_PATH = Path(__file__).parent / "store.db"
DATASET_VERSION = 3

# Search results carry a short excerpt so that grounding on the full text
# requires a follow-up read call, the way a real knowledge server behaves.
EXCERPT_CHARS = 240

CATALOG_URI_PREFIX = "store://catalog/"


def _fts5_available() -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE _probe USING fts5(x)")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()


HAS_FTS5 = _fts5_available()


def init_db() -> None:
    """Create the SQLite database or reset mutable state.

    On first run, creates all tables and inserts seed data.
    On subsequent runs, resets order statuses so each simulation
    starts from a clean state (prevents cross-conversation side
    effects when num_conversations_per_scenario > 1).
    """
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version == DATASET_VERSION:
            conn.execute(
                "UPDATE orders SET status = ("
                "SELECT initial_status FROM order_seed_status "
                "WHERE order_seed_status.order_id = orders.id)"
            )
            conn.commit()
            conn.close()
            return
        conn.close()
        # This database is disposable example state. Rebuilding on a dataset
        # version change prevents an old store.db from silently hiding new seed
        # tables and rows after a branch update.
        DB_PATH.unlink()
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
            created_at TEXT NOT NULL DEFAULT '',
            fulfillment TEXT NOT NULL DEFAULT 'parcel',
            promised_date TEXT NOT NULL DEFAULT '',
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        );
        CREATE TABLE products (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            price REAL NOT NULL,
            category TEXT NOT NULL,
            in_stock INTEGER NOT NULL DEFAULT 1,
            description TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX idx_products_category_price
            ON products(category, price);
        CREATE INDEX idx_products_stock
            ON products(in_stock);

        -- IDs/values below C-001/C-002, ORD-1001..1004, and P-001..P-006
        -- must stay exactly as originally seeded: scenarios.json's
        -- trajectory assertions and knowledge hints reference them by
        -- these exact IDs and field values. Everything past them is new
        -- seed data added for corpus scale, not exercised by assertions.

        INSERT INTO customers VALUES
            ('C-001', 'Alice Johnson', 'alice@example.com', 'premium'),
            ('C-002', 'Bob Smith', 'bob@example.com', 'standard'),
            ('C-003', 'Carlos Mendez', 'carlos@example.com', 'standard'),
            ('C-004', 'Diana Osei', 'diana@example.com', 'premium'),
            ('C-005', 'Ethan Wright', 'ethan@example.com', 'standard'),
            ('C-006', 'Fatima Al-Sayed', 'fatima@example.com', 'premium'),
            ('C-007', 'George Kim', 'george@example.com', 'standard'),
            ('C-008', 'Hannah Muller', 'hannah@example.com', 'standard');

        INSERT INTO orders VALUES
            ('ORD-1001', 'C-001', 'shipped', 249.99,
             'Wireless Headphones x1', '2026-02-24', 'parcel', '2026-02-28'),
            ('ORD-1002', 'C-001', 'processing', 1299.00,
             'ThinkPad X1 Carbon x1', '2026-02-27', 'parcel', '2026-03-03'),
            ('ORD-1003', 'C-002', 'delivered', 59.99,
             'USB-C Hub x1', '2026-02-10', 'parcel', '2026-02-14'),
            ('ORD-1004', 'C-002', 'cancelled', 899.00,
             'MacBook Air M4 x1', '2026-02-15', 'parcel', '2026-02-19'),
            ('ORD-1005', 'C-003', 'delivered', 229.00,
             'Lenovo Tab P11 x1', '2026-01-28', 'parcel', '2026-02-01'),
            ('ORD-1006', 'C-003', 'shipped', 49.99,
             'Amazon Echo Dot x1', '2026-02-25', 'parcel', '2026-03-01'),
            ('ORD-1007', 'C-004', 'processing', 1599.00,
             'ASUS ROG Zephyrus G14 x1', '2026-02-26', 'parcel', '2026-03-02'),
            ('ORD-1008', 'C-004', 'delivered', 429.00,
             'Bose QuietComfort Ultra x1', '2026-01-19', 'parcel', '2026-01-23'),
            ('ORD-1009', 'C-005', 'pending', 99.99,
             'Logitech MX Master 3S x1', '2026-03-01', 'parcel', '2026-03-05'),
            ('ORD-1010', 'C-005', 'cancelled', 329.00,
             'LG UltraGear 27 QHD x1', '2026-02-05', 'parcel', '2026-02-09'),
            ('ORD-1011', 'C-006', 'shipped', 599.00,
             'iPad Air x1', '2026-02-23', 'parcel', '2026-02-27'),
            ('ORD-1012', 'C-006', 'delivered', 249.00,
             'Apple AirPods Pro 2 x1', '2026-01-12', 'pickup', '2026-01-13'),
            ('ORD-1013', 'C-007', 'processing', 449.00,
             'Dell UltraSharp 27 4K x1', '2026-02-28', 'parcel', '2026-03-04'),
            ('ORD-1014', 'C-007', 'delivered', 199.00,
             'Philips Hue Starter Kit x1', '2026-02-02', 'parcel', '2026-02-06'),
            ('ORD-1015', 'C-008', 'shipped', 129.99,
             'Google Nest Thermostat x1', '2026-02-22', 'parcel', '2026-02-26');

        INSERT INTO products VALUES
            ('P-001', 'ThinkPad X1 Carbon', 1299.00, 'laptop', 1,
             'A premium business laptop known for its durable build, comfortable keyboard, and long battery life.'),
            ('P-002', 'MacBook Air M4', 1199.00, 'laptop', 1,
             'Apple''s thin-and-light laptop with the M4 chip, offering strong performance and all-day battery life.'),
            ('P-003', 'Dell XPS 13', 949.00, 'laptop', 0,
             'A compact 13-inch laptop with a nearly borderless display, popular for everyday computing.'),
            ('P-004', 'Sony WH-1000XM5', 249.99, 'headphones', 1,
             'Over-ear headphones widely regarded for industry-leading noise cancellation and sound quality.'),
            ('P-005', 'USB-C Hub', 59.99, 'accessories', 1,
             'A multiport adapter that adds USB-A, HDMI, and card-reader ports to laptops with limited connectivity.'),
            ('P-006', 'Mechanical Keyboard', 149.99, 'accessories', 0,
             'A tactile mechanical keyboard favored for typing feel and durability over standard membrane keyboards.'),
            ('P-007', 'Lenovo Yoga Slim 7', 899.00, 'laptop', 1,
             'A lightweight 14-inch laptop with a bright OLED display, ideal for everyday productivity.'),
            ('P-008', 'ASUS ROG Zephyrus G14', 1599.00, 'laptop', 1,
             'A compact gaming laptop with a powerful GPU, built for demanding games and creative work.'),
            ('P-009', 'Bose QuietComfort Ultra', 429.00, 'headphones', 1,
             'Over-ear headphones with class-leading active noise cancellation and spatial audio.'),
            ('P-010', 'Apple AirPods Pro 2', 249.00, 'headphones', 1,
             'Compact in-ear earbuds with adaptive noise cancellation and a long-lasting charging case.'),
            ('P-011', 'JBL Tune 660NC', 79.99, 'headphones', 0,
             'Budget-friendly on-ear headphones with active noise cancellation and 44 hours of battery life.'),
            ('P-012', 'Anker 65W USB-C Charger', 39.99, 'accessories', 1,
             'A compact fast-charging adapter that can power a laptop and a phone at the same time.'),
            ('P-013', 'Logitech MX Master 3S', 99.99, 'accessories', 1,
             'An ergonomic wireless mouse with quiet clicks and an ultra-precise scroll wheel.'),
            ('P-014', 'Dell UltraSharp 27 4K', 449.00, 'monitors', 1,
             'A 27-inch 4K monitor with accurate color reproduction, suited for design and photo editing.'),
            ('P-015', 'LG UltraGear 27 QHD', 329.00, 'monitors', 1,
             'A 27-inch QHD gaming monitor with a 165Hz refresh rate and fast response time.'),
            ('P-016', 'Samsung Odyssey G9', 1299.00, 'monitors', 0,
             'An ultra-wide curved monitor designed for immersive gaming and multitasking.'),
            ('P-017', 'ASUS ProArt 24', 279.00, 'monitors', 1,
             'A 24-inch monitor calibrated for accurate colors, aimed at creative professionals.'),
            ('P-018', 'iPad Air', 599.00, 'tablets', 1,
             'A slim, powerful tablet suited for note-taking, drawing, and everyday browsing.'),
            ('P-019', 'Samsung Galaxy Tab S9', 799.00, 'tablets', 1,
             'An Android tablet with a vivid AMOLED display and S Pen support included.'),
            ('P-020', 'Lenovo Tab P11', 229.00, 'tablets', 1,
             'An affordable tablet with a large display, good for streaming and light multitasking.'),
            ('P-021', 'Amazon Fire HD 10', 149.00, 'tablets', 0,
             'A budget tablet geared toward media consumption and casual use.'),
            ('P-022', 'Amazon Echo Dot', 49.99, 'smart home', 1,
             'A compact smart speaker with a built-in voice assistant for music and home control.'),
            ('P-023', 'Philips Hue Starter Kit', 199.00, 'smart home', 1,
             'A smart lighting kit with color-changing bulbs and a hub for app and voice control.'),
            ('P-024', 'Google Nest Thermostat', 129.99, 'smart home', 1,
             'A smart thermostat that learns your schedule and helps save energy automatically.'),
            ('P-025', 'Ring Video Doorbell', 99.99, 'smart home', 1,
             'A video doorbell with motion detection and two-way audio, viewable from a phone app.');

        CREATE TABLE verification_codes (
            customer_id TEXT PRIMARY KEY,
            code TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        );
        INSERT INTO verification_codes VALUES
            ('C-001', '123456'),
            ('C-002', '789012'),
            ('C-003', '234567'),
            ('C-004', '345678'),
            ('C-005', '456789'),
            ('C-006', '567890'),
            ('C-007', '678901'),
            ('C-008', '890123');

        CREATE TABLE knowledge_articles (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            topic TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL,
            source_uri TEXT NOT NULL UNIQUE
        );
        CREATE INDEX idx_knowledge_articles_topic_category
            ON knowledge_articles(topic, category);

        CREATE TABLE order_seed_status (
            order_id TEXT PRIMARY KEY,
            initial_status TEXT NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(id)
        );
    """)

    products = generated_products()
    c.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", generated_customers())
    c.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?)", products)
    c.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?)", generated_orders(products)
    )
    c.executemany(
        "INSERT INTO verification_codes VALUES (?, ?)",
        generated_verification_codes(),
    )
    c.executemany(
        "INSERT INTO knowledge_articles VALUES (?, ?, ?, ?, ?, ?)",
        knowledge_articles(),
    )
    c.execute(
        "INSERT INTO order_seed_status(order_id, initial_status) "
        "SELECT id, status FROM orders"
    )
    if HAS_FTS5:
        c.executescript("""
            CREATE VIRTUAL TABLE knowledge_fts USING fts5(
                id UNINDEXED, title, topic, category, content
            );
            INSERT INTO knowledge_fts
                SELECT id, title, topic, category, content FROM knowledge_articles;

            CREATE VIRTUAL TABLE products_fts USING fts5(
                id UNINDEXED, name, category, description
            );
            INSERT INTO products_fts
                SELECT id, name, category, description FROM products;
        """)
    c.execute(f"PRAGMA user_version = {DATASET_VERSION}")
    conn.commit()
    conn.close()


def _query(sql: str, params: tuple = ()) -> list[dict]:
    """Run a query and return results as dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Search helpers ──


def _terms(query: str) -> list[str]:
    """Split a free-text query into searchable terms."""
    return [t for t in re.findall(r"[A-Za-z0-9]+", query.lower()) if len(t) > 1]


def _fts_expr(terms: list[str], operator: str) -> str:
    """Build an FTS5 MATCH expression with prefix matching on each term."""
    return f" {operator} ".join(f'"{t}"*' for t in terms)


def _excerpt(text: str) -> str:
    """Shorten article text to a search-result-sized excerpt."""
    flat = " ".join(text.replace("#", "").split())
    if len(flat) <= EXCERPT_CHARS:
        return flat
    cut = flat[:EXCERPT_CHARS]
    if " " in cut:
        cut = cut[: cut.rindex(" ")]
    return cut + "..."


def _page(rows: list[dict], offset: int, limit: int) -> dict:
    """Wrap rows in a pagination envelope."""
    window = rows[offset : offset + limit]
    return {
        "results": window,
        "total": len(rows),
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(window) < len(rows),
        "next_offset": offset + len(window)
        if offset + len(window) < len(rows)
        else None,
    }


def _ranked_ids(table: str, terms: list[str]) -> list[str]:
    """Return ids from an FTS table, best match first, narrowing before broadening.

    An exact AND match wins outright. Otherwise results are ranked by how many
    query terms they match before bm25 is consulted. Ranking on bm25 alone lets
    a document matching one rare term beat one matching four, which is the
    wrong answer for the multi-term natural-language queries a retrieval agent
    composes.
    """
    exact = _query(
        f"SELECT id FROM {table} WHERE {table} MATCH ? ORDER BY bm25({table})",
        (_fts_expr(terms, "AND"),),
    )
    if exact:
        return [r["id"] for r in exact]

    broad = _query(
        f"SELECT id FROM {table} WHERE {table} MATCH ? ORDER BY bm25({table})",
        (_fts_expr(terms, "OR"),),
    )
    if not broad:
        return []
    bm25_rank = {row["id"]: rank for rank, row in enumerate(broad)}

    matches: dict[str, int] = {}
    for term in terms:
        for row in _query(
            f"SELECT id FROM {table} WHERE {table} MATCH ?", (_fts_expr([term], "OR"),)
        ):
            matches[row["id"]] = matches.get(row["id"], 0) + 1

    return sorted(bm25_rank, key=lambda i: (-matches.get(i, 0), bm25_rank[i]))


# ── Tools ──

AGENT_INSTRUCTIONS = (
    "You are a customer service assistant for an online store. "
    "You have access to tools to look up customers, check orders, "
    "search products and store knowledge, cancel orders, and verify customer identity. "
    "Use them to help the user. "
    "Always confirm destructive actions like cancellations before proceeding. "
    "For sensitive operations, verify the customer's identity first by "
    "sending a verification code and then verifying it."
)


def lookup_customer(email: str) -> str:
    """Look up a customer by email address. Returns customer profile or error."""
    rows = _query("SELECT * FROM customers WHERE email = ?", (email,))
    if not rows:
        return json.dumps({"error": f"No customer found with email {email}"})
    return json.dumps(rows[0])


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


def get_delivery_status(order_id: str) -> str:
    """Get fulfillment progress, method, and promised date for an order."""
    rows = _query(
        "SELECT id, status, fulfillment, created_at, promised_date, items "
        "FROM orders WHERE id = ?",
        (order_id,),
    )
    if not rows:
        return json.dumps({"error": f"Order {order_id} not found"})
    order = rows[0]
    stages = ["pending", "processing", "shipped", "delivered"]
    status = order["status"]
    if status == "cancelled":
        completed = ["pending", "processing"]
        remaining: list[str] = []
    else:
        index = stages.index(status) if status in stages else 0
        completed = stages[: index + 1]
        remaining = stages[index + 1 :]
    method = order["fulfillment"]
    detail = {
        "parcel": "Carrier parcel delivery with tracking once shipped.",
        "pickup": "Held at the store for seven days from the ready notification.",
        "appointment": "Scheduled four-hour appointment window with a delivery crew.",
        "freight": "Curbside freight delivery to an accessible hard surface.",
    }[method]
    return json.dumps(
        {
            "order_id": order["id"],
            "status": status,
            "fulfillment": method,
            "fulfillment_detail": detail,
            "ordered_on": order["created_at"],
            "promised_date": order["promised_date"],
            "lead_days": FULFILLMENT_LEAD_DAYS[method],
            "items": order["items"],
            "stages_completed": completed,
            "stages_remaining": remaining,
            "tracking_number": (
                f"1Z{order['id'].replace('-', '')}"
                if method == "parcel" and status in ("shipped", "delivered")
                else None
            ),
        }
    )


def search_products(
    query: str,
    max_price: float = 0,
    category: str = "",
    in_stock_only: bool = False,
    limit: int = 20,
    offset: int = 0,
) -> str:
    """Search products by name, category, or description with optional filters."""
    limit = max(1, min(limit, 50))
    offset = max(0, offset)
    terms = _terms(query)
    clauses: list[str] = []
    params: list = []
    if terms and HAS_FTS5:
        ids = _ranked_ids("products_fts", terms)
        if not ids:
            return json.dumps({"error": f"No products matching '{query}'"})
        order_by = (
            "CASE id "
            + " ".join(f"WHEN '{pid}' THEN {rank}" for rank, pid in enumerate(ids))
            + " END"
        )
        clauses.append(f"id IN ({','.join('?' * len(ids))})")
        params.extend(ids)
    else:
        order_by = "in_stock DESC, price, name"
        for term in terms or [query.strip()]:
            needle = f"%{term}%"
            clauses.append("(name LIKE ? OR category LIKE ? OR description LIKE ?)")
            params.extend([needle, needle, needle])
    if max_price > 0:
        clauses.append("price <= ?")
        params.append(max_price)
    if category.strip():
        clauses.append("category LIKE ?")
        params.append(f"%{category.strip()}%")
    if in_stock_only:
        clauses.append("in_stock = 1")
    rows = _query(
        "SELECT id, name, price, category, in_stock, description, "
        f"'{CATALOG_URI_PREFIX}' || id AS source_uri FROM products "
        "WHERE " + " AND ".join(clauses) + f" ORDER BY {order_by}",
        tuple(params),
    )
    if not rows:
        return json.dumps({"error": f"No products matching '{query}'"})
    return json.dumps(_page(rows, offset, limit))


def list_product_categories() -> str:
    """List product categories with counts, availability, and price ranges."""
    rows = _query(
        "SELECT category, COUNT(*) AS product_count, "
        "SUM(in_stock) AS in_stock_count, MIN(price) AS min_price, "
        "MAX(price) AS max_price FROM products GROUP BY category ORDER BY category"
    )
    return json.dumps(rows)


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


# ── Knowledge-shaped tools ──
#
# Unlike the transactional tools above (which return JSON for the agent to
# parse), these return prose: they exist to give the retrieval loop actual
# paragraphs to ground on, like the return-policy and how-to guide tools a
# retail agent has. search_store_knowledge returns excerpts only, so grounding on a
# full policy takes a second call to get_knowledge_article or get_document.

_RETURN_POLICIES: dict[str, str] = {
    "laptop": (
        "Laptops can be returned within 30 days of delivery in their original "
        "packaging, with all accessories included. A 15% restocking fee applies "
        "unless the laptop is defective. Proof of purchase is required."
    ),
    "headphones": (
        "Headphones can be returned within 30 days if unopened, or within 14 "
        "days if opened and unused, in the original packaging. No restocking "
        "fee applies. Proof of purchase is required."
    ),
    "accessories": (
        "Accessories can be returned within 30 days of delivery, unopened and "
        "in their original packaging. No restocking fee applies. Proof of "
        "purchase is required."
    ),
    "monitors": (
        "Monitors can be returned within 30 days of delivery if free of dead "
        "pixels or physical damage, in their original packaging. A 10% "
        "restocking fee applies to opened units. Proof of purchase is required."
    ),
    "tablets": (
        "Tablets can be returned within 14 days if unopened, or within 7 days "
        "if opened and in like-new condition, in the original packaging. No "
        "restocking fee applies. Proof of purchase is required."
    ),
    "smart home": (
        "Smart home devices can be returned within 30 days of delivery in "
        "their original packaging with all accessories included. Devices must "
        "be reset to factory settings before return. Proof of purchase is "
        "required."
    ),
}

_PRODUCT_GUIDES: dict[str, str] = {
    "thinkpad x1 carbon": (
        "Charge the ThinkPad X1 Carbon fully before first use. Clean the "
        "display with a soft, dry cloth; avoid harsh chemicals. Store between "
        "0-35°C for battery longevity."
    ),
    "macbook air m4": (
        "Charge the MacBook Air M4 fully before first use. Clean the screen "
        "with a dry, microfiber cloth only. Avoid exposing the device to "
        "extreme heat or direct sunlight for long periods."
    ),
    "dell xps 13": (
        "Charge the Dell XPS 13 fully before first use. Keep the vents clear "
        "of dust to avoid overheating. Use the included USB-C charger for the "
        "best charging speed."
    ),
    "sony wh-1000xm5": (
        "Charge the Sony WH-1000XM5 headphones fully before first use via "
        "USB-C. Store them in the included case when not in use to protect "
        "the hinges and ear cushions."
    ),
    "usb-c hub": (
        "No special setup is required for the USB-C Hub; simply connect it to "
        "a USB-C port. Avoid exposing the hub to liquids."
    ),
    "mechanical keyboard": (
        "Clean the Mechanical Keyboard with compressed air to remove dust "
        "between keys. Avoid liquids near the keyboard."
    ),
    "lenovo yoga slim 7": (
        "Charge the Lenovo Yoga Slim 7 fully before first use. Clean the OLED "
        "display with a soft, dry cloth only, and avoid pressing on the screen."
    ),
    "bose quietcomfort ultra": (
        "Charge the Bose QuietComfort Ultra headphones fully before first use. "
        "Store them in the included case and wipe the ear cushions with a "
        "slightly damp cloth to remove buildup."
    ),
    "logitech mx master 3s": (
        "The Logitech MX Master 3S ships pre-paired; replace the battery via "
        "USB-C charging as needed. Avoid using it on glass surfaces for best "
        "tracking."
    ),
    "dell ultrasharp 27 4k": (
        "Set up the Dell UltraSharp 27 4K on a stable, level surface and "
        "connect via USB-C or DisplayPort for full resolution. Clean the "
        "screen with a dry microfiber cloth only."
    ),
    "ipad air": (
        "Charge the iPad Air fully before first use. Clean the display with a "
        "soft, dry cloth, and avoid exposing it to extreme temperatures."
    ),
    "amazon echo dot": (
        "Plug in the Amazon Echo Dot and follow the companion app to connect "
        "it to Wi-Fi. Keep it away from water and dust it with a dry cloth."
    ),
}


def search_store_knowledge(
    query: str,
    topic: str = "",
    category: str = "",
    limit: int = 8,
    offset: int = 0,
) -> str:
    """Search store policies, buying guides, product care, and service articles.

    Returns short excerpts. Call get_knowledge_article or get_document with the
    returned id or source_uri to read the full text.
    """
    limit = max(1, min(limit, 20))
    offset = max(0, offset)
    terms = _terms(query)
    clauses: list[str] = []
    params: list = []
    if terms and HAS_FTS5:
        ids = _ranked_ids("knowledge_fts", terms)
        if not ids:
            return json.dumps({"error": f"No store knowledge matching '{query}'"})
        order_by = (
            "CASE id "
            + " ".join(f"WHEN '{aid}' THEN {rank}" for rank, aid in enumerate(ids))
            + " END"
        )
        clauses.append(f"id IN ({','.join('?' * len(ids))})")
        params.extend(ids)
    else:
        order_by = "id"
        for term in terms or [query.strip()]:
            needle = f"%{term}%"
            clauses.append(
                "(title LIKE ? OR content LIKE ? OR topic LIKE ? OR category LIKE ?)"
            )
            params.extend([needle, needle, needle, needle])
    if topic.strip():
        clauses.append("topic LIKE ?")
        params.append(f"%{topic.strip()}%")
    if category.strip():
        clauses.append("category LIKE ?")
        params.append(f"%{category.strip()}%")
    rows = _query(
        "SELECT id, title, topic, category, content, source_uri "
        "FROM knowledge_articles WHERE "
        + " AND ".join(clauses)
        + f" ORDER BY {order_by}",
        tuple(params),
    )
    if not rows:
        return json.dumps({"error": f"No store knowledge matching '{query}'"})
    for row in rows:
        content = row.pop("content")
        row["excerpt"] = _excerpt(content)
        row["content_chars"] = len(content)
        row["truncated"] = len(row["excerpt"]) < len(content)
    return json.dumps(_page(rows, offset, limit))


def get_knowledge_article(article_id: str) -> str:
    """Get a complete knowledge article by the ID returned from search."""
    rows = _query(
        "SELECT id, title, topic, category, content, source_uri "
        "FROM knowledge_articles WHERE id = ?",
        (article_id.strip().upper(),),
    )
    if not rows:
        return json.dumps({"error": f"Knowledge article {article_id} not found"})
    return json.dumps(rows[0])


def get_document(uri: str) -> str:
    """Get a full document by its store:// source URI.

    Accepts the source_uri values returned by search_store_knowledge and
    search_products.
    """
    key = uri.strip()
    if key.startswith(CATALOG_URI_PREFIX):
        rows = _query(
            "SELECT id, name AS title, category, price, in_stock, "
            "description AS content, "
            f"'{CATALOG_URI_PREFIX}' || id AS source_uri "
            "FROM products WHERE id = ?",
            (key[len(CATALOG_URI_PREFIX) :].upper(),),
        )
    else:
        rows = _query(
            "SELECT id, title, topic, category, content, source_uri "
            "FROM knowledge_articles WHERE source_uri = ?",
            (key,),
        )
    if not rows:
        return json.dumps({"error": f"No document found at {uri}"})
    return json.dumps(rows[0])


def get_return_policy(category: str) -> str:
    """Get the return policy for a product category (e.g. 'laptop', 'headphones', 'accessories')."""
    key = category.lower().strip()
    for cat, policy in _RETURN_POLICIES.items():
        if key in cat or cat in key:
            return policy
    rows = _query(
        "SELECT content FROM knowledge_articles "
        "WHERE topic = 'returns' AND (category = ? OR category LIKE ?) "
        "ORDER BY category LIMIT 1",
        (key, f"%{key}%"),
    )
    if rows:
        return rows[0]["content"]
    return f"No return policy on file for category '{category}'."


def get_product_guide(product_name: str) -> str:
    """Get the setup/care guide for a product by name."""
    key = product_name.lower().strip()
    for name, guide in _PRODUCT_GUIDES.items():
        if key in name or name in key:
            return guide
    products = _query(
        "SELECT category FROM products WHERE name LIKE ? ORDER BY name LIMIT 1",
        (f"%{key}%",),
    )
    category = products[0]["category"] if products else key
    rows = _query(
        "SELECT content FROM knowledge_articles "
        "WHERE topic = 'product care' AND category = ? LIMIT 1",
        (category,),
    )
    if rows:
        return rows[0]["content"]
    return f"No product guide on file for '{product_name}'."


# ── MCP fixture tools ──
#
# Not part of the customer-service agent's tool set. These exist so the
# platform's MCP connector has something to exercise its normalization,
# size-cap, and error-handling paths against.


def export_catalog_snapshot(category: str = "") -> str:
    """Export the full product catalog. Deliberately large, for size-cap testing."""
    if category.strip():
        rows = _query(
            "SELECT * FROM products WHERE category = ? ORDER BY id",
            (category.strip(),),
        )
    else:
        rows = _query("SELECT * FROM products ORDER BY id")
    return json.dumps({"products": rows, "count": len(rows)})


def render_room_preview(room: str, color: str) -> str:
    """Render a room visualization. Returns an image, not grounding text."""
    return json.dumps(
        {
            "render": "image",
            "room": room.strip() or "living room",
            "color": color.strip() or "white",
            "note": "Generative image output; no text to ground on.",
        }
    )


def simulate_tool_failure(mode: str = "exception") -> str:
    """Fail on purpose so the connector's error handling can be tested.

    mode='exception' raises (surfaces as an MCP isError result);
    mode='in_band' returns an ordinary error payload as a successful call.
    """
    if mode.strip() == "in_band":
        return json.dumps({"error": "Upstream knowledge service returned no data"})
    raise RuntimeError("Simulated upstream knowledge service failure")


def dataset_summary() -> dict:
    """Return stable counts and facets for local development checks."""
    counts = {
        table: _query(f"SELECT COUNT(*) AS count FROM {table}")[0]["count"]
        for table in ("customers", "orders", "products", "knowledge_articles")
    }
    counts["product_categories"] = len(_query("SELECT DISTINCT category FROM products"))
    counts["knowledge_topics"] = len(
        _query("SELECT DISTINCT topic FROM knowledge_articles")
    )
    counts["avg_article_chars"] = _query(
        "SELECT CAST(AVG(LENGTH(content)) AS INT) AS count FROM knowledge_articles"
    )[0]["count"]
    counts["fts5"] = HAS_FTS5
    counts["expected"] = {
        "customers": 8 + EXTRA_CUSTOMER_COUNT,
        "orders": 15 + EXTRA_ORDER_COUNT,
        "products": 25 + EXTRA_PRODUCT_COUNT,
        "knowledge_articles": KNOWLEDGE_ARTICLE_COUNT,
    }
    return counts


if __name__ == "__main__":
    init_db()
    print(json.dumps(dataset_summary(), indent=2))
