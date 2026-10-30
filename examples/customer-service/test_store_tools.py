# SPDX-License-Identifier: Apache-2.0
"""Focused checks for the customer-service example dataset and tools."""

from __future__ import annotations

import ast
import json
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path

import seed_data
import store_tools

EXPECTED_ARTICLES = 89


class StoreToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_db_path = store_tools.DB_PATH
        store_tools.DB_PATH = Path(self.temp_dir.name) / "store.db"
        store_tools.init_db()

    def tearDown(self) -> None:
        store_tools.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def test_expanded_dataset_keeps_canonical_rows(self) -> None:
        summary = store_tools.dataset_summary()
        self.assertEqual(summary["customers"], 108)
        self.assertEqual(summary["orders"], 500)
        self.assertEqual(summary["products"], 225)
        self.assertEqual(summary["knowledge_articles"], EXPECTED_ARTICLES)
        self.assertEqual(summary["product_categories"], 25)

        order = json.loads(store_tools.get_order("ORD-1001"))
        self.assertEqual(order["customer_email"], "alice@example.com")
        self.assertEqual(order["status"], "shipped")
        self.assertEqual(order["total"], 249.99)

    def test_catalog_search_filters_rich_generated_products(self) -> None:
        page = json.loads(
            store_tools.search_products(
                "cordless drill", max_price=150, in_stock_only=True, limit=50
            )
        )
        matches = page["results"]
        self.assertTrue(matches)
        self.assertTrue(all(row["price"] <= 150 for row in matches))
        self.assertTrue(all(row["in_stock"] == 1 for row in matches))
        self.assertTrue(
            any("brushless" in row["description"].lower() for row in matches)
        )
        self.assertTrue(
            all(row["source_uri"].startswith("store://catalog/") for row in matches)
        )

        categories = json.loads(store_tools.list_product_categories())
        self.assertTrue(any(row["category"] == "power tools" for row in categories))
        self.assertTrue(any(row["category"] == "laptop" for row in categories))

    def test_search_is_word_order_independent(self) -> None:
        """Substring matching would make 'match price' find nothing."""
        forward = json.loads(store_tools.search_store_knowledge("price match"))
        reversed_ = json.loads(store_tools.search_store_knowledge("match price"))
        self.assertEqual(forward["results"][0]["id"], "KB-0067")
        self.assertEqual(reversed_["results"][0]["id"], "KB-0067")

    def test_multi_term_queries_rank_by_term_coverage(self) -> None:
        """Ranking on bm25 alone lets one rare term outrank four relevant ones."""
        top = json.loads(
            store_tools.search_store_knowledge("appliance delivery preparation")
        )["results"]
        self.assertEqual(top[0]["id"], "KB-0063")

        # "washer" appears in no article, so this query only resolves through
        # the remaining terms. Ranking by coverage puts the appliance material
        # first; ranking by bm25 alone used to surface a paint article, which
        # shares only one incidental term. Lower ranks still contain
        # incidental matches - that is lexical search working as intended.
        wide = json.loads(
            store_tools.search_store_knowledge(
                "washer delivery preparation installation requirements", limit=3
            )
        )["results"]
        self.assertEqual(wide[0]["category"], "appliances")

    def test_knowledge_search_returns_truncated_citable_excerpts(self) -> None:
        page = json.loads(store_tools.search_store_knowledge("price match", limit=3))
        top = page["results"][0]
        self.assertEqual(top["title"], "Price-match policy")
        self.assertTrue(top["source_uri"].startswith("store://knowledge/"))

        # The excerpt must be a fraction of the document, otherwise a retrieval
        # loop never needs the follow-up read call this fixture exists to test.
        self.assertTrue(top["truncated"])
        self.assertLess(len(top["excerpt"]), top["content_chars"])

        article = json.loads(store_tools.get_knowledge_article(top["id"]))
        self.assertEqual(len(article["content"]), top["content_chars"])
        self.assertIn("marketplace", article["content"].lower())
        # Deciding detail lives past the excerpt boundary.
        self.assertNotIn("price adjustment", top["excerpt"].lower())
        self.assertIn("price adjustment", article["content"].lower())

    def test_get_document_resolves_source_uris(self) -> None:
        top = json.loads(store_tools.search_store_knowledge("price match"))["results"][
            0
        ]
        doc = json.loads(store_tools.get_document(top["source_uri"]))
        self.assertEqual(doc["id"], top["id"])
        self.assertEqual(len(doc["content"]), top["content_chars"])

        product = json.loads(store_tools.search_products("laptop"))["results"][0]
        catalog_doc = json.loads(store_tools.get_document(product["source_uri"]))
        self.assertEqual(catalog_doc["id"], product["id"])

        self.assertIn("error", json.loads(store_tools.get_document("store://nope/x")))

    def test_search_paginates(self) -> None:
        first = json.loads(store_tools.search_store_knowledge("return policy", limit=5))
        self.assertGreater(first["total"], 5)
        self.assertTrue(first["has_more"])
        self.assertEqual(first["next_offset"], 5)

        second = json.loads(
            store_tools.search_store_knowledge(
                "return policy", limit=5, offset=first["next_offset"]
            )
        )
        self.assertEqual(second["offset"], 5)
        self.assertFalse(
            {r["id"] for r in first["results"]} & {r["id"] for r in second["results"]}
        )

        last = json.loads(
            store_tools.search_store_knowledge(
                "return policy", limit=5, offset=first["total"] - 1
            )
        )
        self.assertFalse(last["has_more"])
        self.assertIsNone(last["next_offset"])

    def test_compat_policy_and_guide_tools_fall_back_to_articles(self) -> None:
        policy = store_tools.get_return_policy("power tools")
        self.assertIn("90 days", policy)
        guide = store_tools.get_product_guide("Atlas Cordless Drill 1")
        self.assertIn("battery", guide.lower())

    def test_delivery_status_reports_fulfillment_progress(self) -> None:
        status = json.loads(store_tools.get_delivery_status("ORD-1001"))
        self.assertEqual(status["status"], "shipped")
        self.assertEqual(status["fulfillment"], "parcel")
        self.assertEqual(status["ordered_on"], "2026-02-24")
        self.assertEqual(status["promised_date"], "2026-02-28")
        self.assertIn("delivered", status["stages_remaining"])
        self.assertIsNotNone(status["tracking_number"])

        cancelled = json.loads(store_tools.get_delivery_status("ORD-1004"))
        self.assertEqual(cancelled["stages_remaining"], [])
        self.assertIsNone(cancelled["tracking_number"])

        self.assertIn("error", json.loads(store_tools.get_delivery_status("ORD-0")))

    def test_fixture_tools_support_connector_edge_cases(self) -> None:
        snapshot = json.loads(store_tools.export_catalog_snapshot())
        self.assertEqual(snapshot["count"], 225)
        self.assertGreater(len(json.dumps(snapshot)), 50_000)

        with self.assertRaises(RuntimeError):
            store_tools.simulate_tool_failure("exception")
        self.assertIn("error", json.loads(store_tools.simulate_tool_failure("in_band")))

    def test_init_resets_mutable_generated_orders(self) -> None:
        conn = sqlite3.connect(store_tools.DB_PATH)
        order_id = conn.execute(
            "SELECT id FROM orders WHERE id > 'ORD-1015' AND status = 'processing' "
            "ORDER BY id LIMIT 1"
        ).fetchone()[0]
        conn.close()

        result = json.loads(store_tools.cancel_order(order_id))
        self.assertTrue(result["success"])
        self.assertEqual(
            json.loads(store_tools.get_order(order_id))["status"], "cancelled"
        )

        store_tools.init_db()
        self.assertEqual(
            json.loads(store_tools.get_order(order_id))["status"], "processing"
        )

    def test_old_example_database_is_rebuilt(self) -> None:
        conn = sqlite3.connect(store_tools.DB_PATH)
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
        conn.close()

        store_tools.init_db()

        conn = sqlite3.connect(store_tools.DB_PATH)
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        conn.close()
        self.assertEqual(version, store_tools.DATASET_VERSION)
        self.assertEqual(
            store_tools.dataset_summary()["knowledge_articles"], EXPECTED_ARTICLES
        )

    def test_mcp_grounding_cases_reference_real_fixture_rows(self) -> None:
        cases_path = Path(__file__).with_name("mcp_grounding_cases.json")
        document = json.loads(cases_path.read_text())
        cases = document["cases"]
        shapes = set(document["grounding_shapes"])
        self.assertGreaterEqual(len(cases), 15)

        conn = sqlite3.connect(store_tools.DB_PATH)
        conn.row_factory = sqlite3.Row
        selectors = {
            "KB-": "SELECT title || ' ' || content AS value "
            "FROM knowledge_articles WHERE id = ?",
            "ORD-": "SELECT id || ' ' || status || ' ' || fulfillment || ' ' "
            "|| items AS value FROM orders WHERE id = ?",
            "P-": "SELECT name || ' price=' || price || ' in_stock=' || in_stock "
            "|| ' ' || description AS value FROM products WHERE id = ?",
        }
        for case in cases:
            self.assertIn(case["grounding_shape"], shapes, case["id"])
            self.assertTrue(case["expected_tools"], case["id"])
            for tool_name in case["expected_tools"]:
                self.assertTrue(
                    callable(getattr(store_tools, tool_name, None)),
                    f"{case['id']}: missing tool {tool_name}",
                )
            if case["grounding_shape"] == "no_evidence":
                self.assertEqual(case["expected_evidence"], [], case["id"])
                continue
            self.assertTrue(case["expected_evidence"], case["id"])
            for expected in case["expected_evidence"]:
                source_ref = expected["source_ref"]
                prefix = next(p for p in selectors if source_ref.startswith(p))
                row = conn.execute(selectors[prefix], (source_ref,)).fetchone()
                self.assertIsNotNone(row, f"{case['id']}: missing {source_ref}")
                for fragment in expected["must_contain"]:
                    self.assertIn(fragment.lower(), row["value"].lower(), case["id"])
        conn.close()

    def test_corpus_export_subsets_stay_disjoint(self) -> None:
        """A narrower subset must not inherit a wider run's files."""
        import export_corpus

        out = Path(self.temp_dir.name) / "corpus"
        keep = out / "my-own-notes.md"

        count_all, removed = export_corpus.export(out, "all")
        self.assertEqual(count_all, EXPECTED_ARTICLES)
        self.assertEqual(removed, 0)
        keep.write_text("not written by the exporter")

        count_service, removed = export_corpus.export(out, "service")
        self.assertEqual(removed, EXPECTED_ARTICLES)
        # Reporting 29 while leaving 89 on disk silently merges the subsets.
        self.assertEqual(len(list(out.glob("KB-*.md"))), count_service)
        # Only the exporter's own KB-*.md files are cleared.
        self.assertTrue(keep.exists())

        count_categories, _ = export_corpus.export(out, "categories")
        self.assertEqual(count_service + count_categories, EXPECTED_ARTICLES)

    def test_no_evidence_case_really_has_no_evidence(self) -> None:
        """The negative case is only meaningful while the corpus stays silent."""
        cases = json.loads(
            Path(__file__).with_name("mcp_grounding_cases.json").read_text()
        )["cases"]
        for case in cases:
            if case["grounding_shape"] != "no_evidence":
                continue
            result = json.loads(store_tools.search_store_knowledge("cryptocurrency"))
            self.assertIn("error", result, case["id"])

    def test_every_order_agrees_with_its_own_dates(self) -> None:
        """Status and dates used to be two independent cycles.

        That produced 168 of the 485 generated orders contradicting
        themselves - "pending" six months past its promised date, "delivered"
        before that date arrived. An agent asked "where is my order" reads both
        fields and cannot answer either sensibly.
        """
        anchor = seed_data.DATASET_ANCHOR
        rows = seed_data.generated_orders(seed_data.generated_products())
        self.assertTrue(rows)

        for (
            order_id,
            _customer,
            status,
            _total,
            _items,
            created,
            _method,
            promised,
        ) in rows:
            ordered_on = date.fromisoformat(created)
            promised_on = date.fromisoformat(promised)
            self.assertGreaterEqual(promised_on, ordered_on, order_id)
            if status in ("pending", "processing"):
                self.assertGreater(promised_on, anchor, f"{order_id} is {status}")
            if status == "delivered":
                self.assertLessEqual(promised_on, anchor, f"{order_id} is {status}")

    def test_generated_orders_still_cover_every_status(self) -> None:
        """Deriving dates from status must not collapse the mix to one value."""
        rows = seed_data.generated_orders(seed_data.generated_products())
        statuses = {row[2] for row in rows}

        self.assertEqual(
            statuses,
            {"pending", "processing", "shipped", "delivered", "cancelled"},
        )


def _agent_tool_names() -> set[str]:
    """Names in TOOLS_LIST, read from the source.

    Parsed rather than imported: these tests run without the agent SDKs
    installed, and importing the wrappers would pull in `agents`.
    """
    source = Path(__file__).with_name("openai_agents").joinpath("tools.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        is_tools_list = (
            isinstance(node, ast.AnnAssign)
            and getattr(node.target, "id", None) == "TOOLS_LIST"
        )
        if is_tools_list and isinstance(node.value, ast.List):
            return {
                element.id
                for element in node.value.elts
                if isinstance(element, ast.Name)
            }
    raise AssertionError("TOOLS_LIST not found in openai_agents/tools.py")


class ScenarioAssertionTest(unittest.TestCase):
    """The example scenarios have to survive the agent gaining a tool.

    ``strict``, ``unordered`` and ``within`` all fail on a call outside the
    expected set, so adding a tool to the agent can break a scenario that has
    nothing to do with it. These pin which scenarios carry that coupling, so
    the next tool addition has to look at them deliberately.
    """

    def setUp(self) -> None:
        self.scenarios = json.loads(
            Path(__file__).with_name("scenarios.json").read_text()
        )

    def test_the_file_is_schema_v2(self) -> None:
        self.assertEqual(self.scenarios["schema_version"], "v2")

    def test_every_scenario_can_be_satisfied(self) -> None:
        """A scenario with no checklist has no sub-goal to satisfy, so it can
        only ever end in gave_up or max_turns."""
        for scenario in self.scenarios["scenarios"]:
            with self.subTest(scenario["scenario_id"]):
                checklist = scenario.get("goal_checklist") or []
                self.assertTrue(checklist)
                self.assertTrue(
                    any(item.get("mandatory", True) for item in checklist),
                    "every item optional leaves nothing to satisfy",
                )

    def test_checklist_items_reference_real_knowledge_chunks(self) -> None:
        for scenario in self.scenarios["scenarios"]:
            with self.subTest(scenario["scenario_id"]):
                chunk_ids = {c.get("id") for c in scenario.get("knowledge", [])}
                for item in scenario.get("goal_checklist") or []:
                    for ref in item.get("knowledge_chunk_ids", []):
                        self.assertIn(ref, chunk_ids)

    def test_closed_set_assertions_list_every_tool_the_agent_has(self) -> None:
        """A "within" assertion is a closed set: any other call fails it.

        So every tool the agent could reasonably reach has to be named, or the
        scenario fails for a reason that has nothing to do with what it tests.
        """
        available = _agent_tool_names()
        for scenario in self.scenarios["scenarios"]:
            for assertion in scenario.get("assertions", []):
                if assertion.get("match_mode") != "within":
                    continue
                with self.subTest(scenario["scenario_id"]):
                    named = {call["name"] for call in assertion["expected"]}
                    self.assertTrue(
                        named <= available,
                        f"expects tools the agent does not have: {named - available}",
                    )

    def test_only_the_trajectory_scenario_is_strict(self) -> None:
        """Strict pins order AND count, so it breaks on any extra call.

        One scenario earns that: the verify-then-cancel sequence is its whole
        subject. Its agent context names the four tools and rules out the rest.
        """
        strict = [
            scenario["scenario_id"]
            for scenario in self.scenarios["scenarios"]
            for assertion in scenario.get("assertions", [])
            if assertion.get("match_mode") in ("strict", "unordered")
        ]

        self.assertEqual(strict, ["auth_cancel_order"])


if __name__ == "__main__":
    unittest.main()
