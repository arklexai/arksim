# SPDX-License-Identifier: Apache-2.0
"""Unit tests for arksim.scenario.goal_discovery."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from arksim.scenario.goal_discovery.clusterer import select_exemplars
from arksim.scenario.goal_discovery.models import ConversationInput, GoalDiscoveryResult
from arksim.scenario.goal_discovery.preprocessing import (
    clean_text,
    extract_first_turns,
    sample_conversations,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def make_conv(*turns: tuple[str, str]) -> ConversationInput:
    """Build a ConversationInput from (role, content) pairs."""
    return ConversationInput(turns=[{"role": r, "content": c} for r, c in turns])


# ── ConversationInput ─────────────────────────────────────────────────────────


class TestConversationInput:
    def test_first_user_turn_basic(self) -> None:
        conv = make_conv(("assistant", "Hi!"), ("user", "I need help with my order"))
        assert conv.first_user_turn() == "I need help with my order"

    def test_first_user_turn_skips_assistant(self) -> None:
        conv = make_conv(("assistant", "Welcome"), ("user", "Cancel my order please"))
        assert conv.first_user_turn() == "Cancel my order please"

    def test_first_user_turn_human_role(self) -> None:
        conv = make_conv(("human", "Where is my package?"))
        assert conv.first_user_turn() == "Where is my package?"

    def test_first_user_turn_customer_role(self) -> None:
        conv = make_conv(("customer", "Refund request"))
        assert conv.first_user_turn() == "Refund request"

    def test_first_user_turn_min_words(self) -> None:
        conv = make_conv(
            ("user", "Hi"), ("user", "I want to reschedule my appointment")
        )
        assert (
            conv.first_user_turn(min_words=3) == "I want to reschedule my appointment"
        )

    def test_first_user_turn_no_user(self) -> None:
        conv = make_conv(("assistant", "Hello!"))
        assert conv.first_user_turn() is None

    def test_first_user_turn_empty_content(self) -> None:
        conv = make_conv(("user", "   "), ("user", "Track my shipment"))
        assert conv.first_user_turn() == "Track my shipment"

    def test_first_user_turn_no_turns(self) -> None:
        conv = ConversationInput(turns=[])
        assert conv.first_user_turn() is None


# ── GoalDiscoveryResult ───────────────────────────────────────────────────────


class TestGoalDiscoveryResult:
    def _make_result(self) -> GoalDiscoveryResult:
        from arksim.scenario.goal_discovery.models import GoalCluster

        return GoalDiscoveryResult(
            goals=[
                GoalCluster(
                    id="goal_000",
                    name="Track a Shipment",
                    description="User wants to check the status of their order.",
                    exemplars=["Where is my package?"],
                    size=40,
                    prevalence=0.4,
                ),
                GoalCluster(
                    id="goal_001",
                    name="Request a Refund",
                    description="User wants money back for a purchase.",
                    exemplars=["I want a refund"],
                    size=20,
                    prevalence=0.2,
                ),
            ],
            method="goal_discovery",
            n_input=100,
        )

    def test_to_goal_list_length(self) -> None:
        result = self._make_result()
        assert len(result.to_goal_list()) == 2

    def test_to_goal_list_fields(self) -> None:
        result = self._make_result()
        item = result.to_goal_list()[0]
        assert item["goal"] == "Track a Shipment"
        assert "description" in item
        assert "exemplars" in item
        assert "prevalence" in item
        assert item["size"] == 40
        assert item["confidence"] == 1.0
        assert item["source"] == "clustering"

    def test_goal_cluster_defaults(self) -> None:
        from arksim.scenario.goal_discovery.models import GoalCluster

        g = GoalCluster(
            id="g0", name="X", description="d", exemplars=[], size=1, prevalence=1.0
        )
        assert g.confidence == 1.0
        assert g.source == "clustering"

    def test_goal_cluster_provided_intent_source(self) -> None:
        from arksim.scenario.goal_discovery.models import GoalCluster

        g = GoalCluster(
            id="g0",
            name="X",
            description="d",
            exemplars=[],
            size=5,
            prevalence=0.5,
            confidence=1.0,
            source="provided_intent",
        )
        assert g.source == "provided_intent"


# ── preprocessing ─────────────────────────────────────────────────────────────


class TestPreprocessing:
    def test_extract_first_turns_basic(self) -> None:
        convs = [
            make_conv(("user", "Hello there, I need help")),
            make_conv(("assistant", "Hi")),  # no user turn
            make_conv(("user", "Track my order please")),
        ]
        result = extract_first_turns(convs, min_words=1)
        assert len(result) == 2
        assert result[0] == (0, "Hello there, I need help")
        assert result[1] == (2, "Track my order please")

    def test_extract_first_turns_min_words_filters(self) -> None:
        convs = [
            make_conv(("user", "Hi")),
            make_conv(("user", "Cancel my appointment please")),
        ]
        result = extract_first_turns(convs, min_words=3)
        assert len(result) == 1
        assert "Cancel" in result[0][1]

    def test_extract_first_turns_preserves_original_index(self) -> None:
        convs = [
            make_conv(("assistant", "only assistant")),
            make_conv(("user", "I need a refund for my order")),
        ]
        result = extract_first_turns(convs, min_words=1)
        assert result[0][0] == 1

    def test_contains_profanity_detects_word(self) -> None:
        from arksim.scenario.goal_discovery.preprocessing import contains_profanity

        assert contains_profanity("wanna add fucking products to the basket") is True

    def test_contains_profanity_clean_text(self) -> None:
        from arksim.scenario.goal_discovery.preprocessing import contains_profanity

        assert contains_profanity("I need to add an item to the cart") is False

    def test_contains_profanity_no_false_positive(self) -> None:
        from arksim.scenario.goal_discovery.preprocessing import contains_profanity

        assert contains_profanity("classic assignment for the class") is False

    def test_select_exemplars_excludes_profane(self) -> None:
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((4, 8)).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        labels = np.zeros(4, dtype=int)
        texts = [
            "I need to add an item",
            "wanna add fucking products",
            "how do I add to cart",
            "add item to basket",
        ]
        from arksim.scenario.goal_discovery.clusterer import select_exemplars
        from arksim.scenario.goal_discovery.preprocessing import contains_profanity

        result = select_exemplars(
            emb, labels, texts, cluster_id=0, n=4, text_filter=contains_profanity
        )
        assert "wanna add fucking products" not in result

    def test_select_exemplars_falls_back_when_all_profane(self) -> None:
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((2, 8)).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        labels = np.zeros(2, dtype=int)
        texts = ["what the fuck", "holy shit"]
        from arksim.scenario.goal_discovery.clusterer import select_exemplars
        from arksim.scenario.goal_discovery.preprocessing import contains_profanity

        result = select_exemplars(
            emb, labels, texts, cluster_id=0, n=2, text_filter=contains_profanity
        )
        assert len(result) == 2  # fell back to full pool

    def test_clean_text_collapses_whitespace(self) -> None:
        assert clean_text("  hello   world  ") == "hello world"

    def test_clean_text_strips_newlines(self) -> None:
        assert clean_text("hello\nworld\n\n") == "hello world"

    def test_clean_text_strips_redacted_token(self) -> None:
        assert (
            clean_text("My name is [REDACTED] and I need help")
            == "My name is and I need help"
        )

    def test_clean_text_strips_asterisk_masking(self) -> None:
        assert clean_text("Call me at ***-***-1234") == "Call me at - -1234"

    def test_clean_text_strips_multiple_pii_tokens(self) -> None:
        result = clean_text("[NAME] lives at [ADDRESS], email [EMAIL]")
        assert result == "lives at , email"

    def test_sample_conversations_exact(self) -> None:
        convs = [make_conv(("user", f"message {i}")) for i in range(10)]
        sample = sample_conversations(convs, n=5, seed=0)
        assert len(sample) == 5

    def test_sample_conversations_larger_than_corpus(self) -> None:
        convs = [make_conv(("user", "hi there"))]
        sample = sample_conversations(convs, n=100, seed=0)
        assert sample is convs

    def test_sample_conversations_reproducible(self) -> None:
        convs = [make_conv(("user", f"message {i}")) for i in range(20)]
        a = sample_conversations(convs, n=5, seed=7)
        b = sample_conversations(convs, n=5, seed=7)
        assert a == b

    def test_extract_first_turns_prefers_reformulated(self) -> None:
        conv = ConversationInput(
            turns=[{"role": "user", "content": "What do I need to know?"}],
            meta={
                "reformulated_question": "What should I know before buying this bathroom fan?"
            },
        )
        result = extract_first_turns(
            [conv], min_words=3, reformulated_key="reformulated_question"
        )
        assert result[0][1] == "What should I know before buying this bathroom fan?"

    def test_extract_first_turns_falls_back_when_reformulated_too_short(self) -> None:
        conv = ConversationInput(
            turns=[{"role": "user", "content": "I need help with my order return"}],
            meta={"reformulated_question": "hi"},
        )
        result = extract_first_turns(
            [conv], min_words=3, reformulated_key="reformulated_question"
        )
        assert result[0][1] == "I need help with my order return"

    def test_extract_first_turns_no_reformulated_key(self) -> None:
        conv = ConversationInput(
            turns=[{"role": "user", "content": "Track my shipment please"}],
            meta={"reformulated_question": "Track the status of my order shipment"},
        )
        # Without reformulated_key, should use raw turn
        result = extract_first_turns([conv], min_words=1)
        assert result[0][1] == "Track my shipment please"


# ── ConversationInput.from_flat_record ─────────────────────────────────────────


class TestFromFlatRecord:
    def _base_record(self) -> dict:
        return {
            "mcvis_id": "60494091257804828082499201313671060973",
            "svoc_id": None,
            "user_question": "What do I need to know before I buy?",
            "page_type": "PIP",
            "item_id": "341282955",
            "referrer_page": "https://www.homedepot.com/p/some-product/341282933",
            "last_updated": "2026-06-29 17:49:08",
            "session_id": "fallback-sessionId-01d41a48-75eb-47d6-9b3f-50a97f0791d6",
            "store_id": "6003",
            "zip_code": "82901",
            "device_type": "desktop",
            "language_code": "en-US",
            "output_details": '{"intent": "Product Information", "products": []}',
            "model_name": "flash25",
            "reformulated_user_question": "What should I know before buying this bathroom fan motor assembly?",
            "summarized_answers": "This Vrbgify Bathroom Fan Motor Assembly is compatible with Nutone 763RLN...",
            "intent": "Product Information",
            "env": "prod",
            "received_timestamp": "2026-06-29 17:49:09",
            "customer_type": "B2C",
            "image_id": None,
            "action_id": None,
            "input_type": "text",
        }

    def test_turns_populated(self) -> None:
        conv = ConversationInput.from_flat_record(self._base_record())
        assert len(conv.turns) == 2
        assert conv.turns[0]["role"] == "user"
        assert conv.turns[0]["content"] == "What do I need to know before I buy?"
        assert conv.turns[1]["role"] == "assistant"

    def test_reformulated_question_in_meta(self) -> None:
        conv = ConversationInput.from_flat_record(self._base_record())
        assert conv.meta["reformulated_question"] == (
            "What should I know before buying this bathroom fan motor assembly?"
        )

    def test_intent_in_meta(self) -> None:
        conv = ConversationInput.from_flat_record(self._base_record())
        assert conv.meta["intent"] == "Product Information"

    def test_metadata_fields_present(self) -> None:
        conv = ConversationInput.from_flat_record(self._base_record())
        assert conv.meta["page_type"] == "PIP"
        assert conv.meta["item_id"] == "341282955"
        assert conv.meta["store_id"] == "6003"
        assert conv.meta["customer_type"] == "B2C"
        assert conv.meta["device_type"] == "desktop"

    def test_none_fields_excluded_from_meta(self) -> None:
        conv = ConversationInput.from_flat_record(self._base_record())
        assert "svoc_id" not in conv.meta
        assert "image_id" not in conv.meta
        assert "action_id" not in conv.meta

    def test_missing_summarized_answers(self) -> None:
        record = self._base_record()
        record["summarized_answers"] = None
        conv = ConversationInput.from_flat_record(record)
        assert len(conv.turns) == 1
        assert conv.turns[0]["role"] == "user"

    def test_missing_user_question(self) -> None:
        record = self._base_record()
        record["user_question"] = None
        conv = ConversationInput.from_flat_record(record)
        assert len(conv.turns) == 1
        assert conv.turns[0]["role"] == "assistant"

    def test_first_user_turn_returns_raw_question(self) -> None:
        conv = ConversationInput.from_flat_record(self._base_record())
        assert conv.first_user_turn() == "What do I need to know before I buy?"

    def test_extract_first_turns_uses_reformulated(self) -> None:
        conv = ConversationInput.from_flat_record(self._base_record())
        result = extract_first_turns(
            [conv], min_words=3, reformulated_key="reformulated_question"
        )
        assert result[0][1] == (
            "What should I know before buying this bathroom fan motor assembly?"
        )


# ── from_conversations_record ─────────────────────────────────────────────────


class TestFromConversationsRecord:
    def _base_record(self) -> dict:
        return {
            "id": "conv-000001",
            "created_at": "2026-02-10T09:25:00Z",
            "modified_at": "2026-02-10T09:32:00Z",
            "user_rating": {"rating": 4, "feedback": "Good."},
            "messages": [
                {
                    "role": "user",
                    "content": "I need to add an item to the cart",
                    "intent": "add_product",
                },
                {"role": "assistant", "content": "Sure, here is how you add an item."},
            ],
            "category": "CART",
            "tags": "BL",
        }

    def test_turns_mapped_from_messages(self) -> None:
        conv = ConversationInput.from_conversations_record(self._base_record())
        assert len(conv.turns) == 2
        assert conv.turns[0]["role"] == "user"
        assert conv.turns[0]["content"] == "I need to add an item to the cart"
        assert conv.turns[1]["role"] == "assistant"

    def test_intent_lifted_to_meta(self) -> None:
        conv = ConversationInput.from_conversations_record(self._base_record())
        assert conv.meta["intent"] == "add_product"

    def test_id_in_meta(self) -> None:
        conv = ConversationInput.from_conversations_record(self._base_record())
        assert conv.meta["id"] == "conv-000001"

    def test_category_and_tags_in_meta(self) -> None:
        conv = ConversationInput.from_conversations_record(self._base_record())
        assert conv.meta["category"] == "CART"
        assert conv.meta["tags"] == "BL"

    def test_user_rating_in_meta(self) -> None:
        conv = ConversationInput.from_conversations_record(self._base_record())
        assert conv.meta["user_rating"] == {"rating": 4, "feedback": "Good."}

    def test_null_fields_excluded_from_meta(self) -> None:
        record = self._base_record()
        record["created_at"] = None
        record["modified_at"] = None
        conv = ConversationInput.from_conversations_record(record)
        assert "created_at" not in conv.meta
        assert "modified_at" not in conv.meta

    def test_first_user_turn(self) -> None:
        conv = ConversationInput.from_conversations_record(self._base_record())
        assert conv.first_user_turn() == "I need to add an item to the cart"

    def test_empty_messages(self) -> None:
        record = self._base_record()
        record["messages"] = []
        conv = ConversationInput.from_conversations_record(record)
        assert conv.turns == []
        assert conv.first_user_turn() is None


# ── clusterer ─────────────────────────────────────────────────────────────────


class TestSelectExemplars:
    def _make_embeddings(self, n: int, d: int = 8) -> np.ndarray:
        rng = np.random.default_rng(0)
        e = rng.standard_normal((n, d)).astype(np.float32)
        norms = np.linalg.norm(e, axis=1, keepdims=True)
        return e / norms

    def test_returns_up_to_n(self) -> None:
        emb = self._make_embeddings(20)
        labels = np.array([0] * 10 + [1] * 10)
        texts = [f"text {i}" for i in range(20)]
        result = select_exemplars(emb, labels, texts, cluster_id=0, n=4)
        assert len(result) == 4

    def test_returns_fewer_than_n_when_cluster_small(self) -> None:
        emb = self._make_embeddings(5)
        labels = np.zeros(5, dtype=int)
        texts = [f"t{i}" for i in range(5)]
        result = select_exemplars(emb, labels, texts, cluster_id=0, n=10)
        assert len(result) == 5

    def test_empty_cluster_returns_empty(self) -> None:
        emb = self._make_embeddings(5)
        labels = np.zeros(5, dtype=int)
        texts = [f"t{i}" for i in range(5)]
        result = select_exemplars(emb, labels, texts, cluster_id=99, n=3)
        assert result == []

    def test_all_exemplars_from_correct_cluster(self) -> None:
        emb = self._make_embeddings(10)
        labels = np.array([0] * 5 + [1] * 5)
        texts = [f"cluster0_{i}" for i in range(5)] + [
            f"cluster1_{i}" for i in range(5)
        ]
        result = select_exemplars(emb, labels, texts, cluster_id=1, n=3)
        assert all("cluster1_" in t for t in result)


# ── GoalDiscoveryPipeline integration (mocked LLM + embedder) ────────────────


class TestGoalDiscoveryPipeline:
    def _fake_embeddings(self, n: int, d: int = 8) -> np.ndarray:
        rng = np.random.default_rng(42)
        e = rng.standard_normal((n, d)).astype(np.float32)
        norms = np.linalg.norm(e, axis=1, keepdims=True)
        return e / norms

    def _make_convs(self, n: int) -> list[ConversationInput]:
        phrases = [
            "I want to track my order",
            "Where is my package?",
            "When will my shipment arrive?",
            "I need a refund",
            "How do I return an item?",
            "I want my money back",
        ]
        return [make_conv(("user", phrases[i % len(phrases)])) for i in range(n)]

    @patch("arksim.scenario.goal_discovery.pipeline.build_embedding_service")
    @patch("arksim.scenario.goal_discovery.pipeline.LLM")
    def test_discover_returns_result(
        self, mock_llm_cls: MagicMock, mock_embedder_cls: MagicMock
    ) -> None:
        from arksim.scenario.goal_discovery.pipeline import GoalDiscoveryPipeline

        convs = self._make_convs(30)
        fake_emb = self._fake_embeddings(30)

        # Stub embedder
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = fake_emb
        mock_embedder_cls.return_value = mock_embedder

        # Stub LLM: name call returns valid JSON; no merge
        name_json = '{"name": "Track Order", "description": "User tracks order.", "intent_type": "informational"}'
        mock_llm = MagicMock()
        mock_llm.call_async = AsyncMock(return_value=name_json)
        mock_llm.call.return_value = '{"groups": [[0], [1]]}'
        mock_llm_cls.return_value = mock_llm

        pipeline = GoalDiscoveryPipeline(
            clustering_method="kmeans",
            k_range=(2, 3),
            min_words=1,
            merge_similar=False,
        )
        result = pipeline.discover(convs)

        assert isinstance(result, GoalDiscoveryResult)
        assert result.method == "goal_discovery"
        assert result.n_input == 30
        assert len(result.goals) >= 1

    @patch("arksim.scenario.goal_discovery.pipeline.build_embedding_service")
    @patch("arksim.scenario.goal_discovery.pipeline.LLM")
    def test_discover_empty_input(
        self, mock_llm_cls: MagicMock, mock_embedder_cls: MagicMock
    ) -> None:
        from arksim.scenario.goal_discovery.pipeline import GoalDiscoveryPipeline

        pipeline = GoalDiscoveryPipeline(min_words=1)
        # All conversations have no user turns
        convs = [make_conv(("assistant", "Hello")) for _ in range(5)]
        result = pipeline.discover(convs)
        assert result.goals == []
        assert result.n_input == 5

    @patch("arksim.scenario.goal_discovery.pipeline.build_embedding_service")
    @patch("arksim.scenario.goal_discovery.pipeline.LLM")
    def test_discover_goals_sorted_by_prevalence(
        self, mock_llm_cls: MagicMock, mock_embedder_cls: MagicMock
    ) -> None:
        from arksim.scenario.goal_discovery.pipeline import GoalDiscoveryPipeline

        convs = self._make_convs(20)
        fake_emb = self._fake_embeddings(20)

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = fake_emb
        mock_embedder_cls.return_value = mock_embedder

        name_json = (
            '{"name": "Goal", "description": "desc", "intent_type": "informational"}'
        )
        mock_llm = MagicMock()
        mock_llm.call_async = AsyncMock(return_value=name_json)
        mock_llm_cls.return_value = mock_llm

        pipeline = GoalDiscoveryPipeline(
            clustering_method="kmeans", k_range=(2, 2), min_words=1, merge_similar=False
        )
        result = pipeline.discover(convs)

        prevalences = [g.prevalence for g in result.goals]
        assert prevalences == sorted(prevalences, reverse=True)

    def test_invalid_clustering_method_raises(self) -> None:
        from arksim.scenario.goal_discovery.pipeline import GoalDiscoveryPipeline

        with pytest.raises(ValueError, match="Unknown clustering_method"):
            GoalDiscoveryPipeline(clustering_method="bad_method", min_words=1)

    def test_discover_single_turn_returns_empty(self) -> None:
        from arksim.scenario.goal_discovery.pipeline import GoalDiscoveryPipeline

        pipeline = GoalDiscoveryPipeline(
            clustering_method="kmeans", k_range=(2, 3), min_words=1
        )
        convs = [make_conv(("user", "I need help with my order"))]
        result = pipeline.discover(convs)
        assert result.goals == []
        assert result.n_input == 1

    @patch("arksim.scenario.goal_discovery.pipeline.build_embedding_service")
    @patch("arksim.scenario.goal_discovery.pipeline.LLM")
    def test_merge_preserves_clusters_omitted_by_llm(
        self, mock_llm_cls: MagicMock, mock_embedder_cls: MagicMock
    ) -> None:
        from arksim.scenario.goal_discovery.pipeline import GoalDiscoveryPipeline

        convs = self._make_convs(30)
        fake_emb = self._fake_embeddings(30)

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = fake_emb
        mock_embedder_cls.return_value = mock_embedder

        name_json = (
            '{"name": "Goal", "description": "desc", "intent_type": "informational"}'
        )
        mock_llm = MagicMock()
        mock_llm.call_async = AsyncMock(return_value=name_json)
        # LLM only mentions cluster 0 in groups, omitting cluster 1
        mock_llm.call.return_value = '{"groups": [[0]]}'
        mock_llm_cls.return_value = mock_llm

        pipeline = GoalDiscoveryPipeline(
            clustering_method="kmeans", k_range=(2, 2), min_words=1, merge_similar=True
        )
        result = pipeline.discover(convs)
        assert len(result.goals) == 2

    @patch("arksim.scenario.goal_discovery.pipeline.build_embedding_service")
    @patch("arksim.scenario.goal_discovery.pipeline.LLM")
    def test_merge_ignores_oob_indices_from_llm(
        self, mock_llm_cls: MagicMock, mock_embedder_cls: MagicMock
    ) -> None:
        from arksim.scenario.goal_discovery.pipeline import GoalDiscoveryPipeline

        convs = self._make_convs(30)
        fake_emb = self._fake_embeddings(30)

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = fake_emb
        mock_embedder_cls.return_value = mock_embedder

        name_json = (
            '{"name": "Goal", "description": "desc", "intent_type": "informational"}'
        )
        mock_llm = MagicMock()
        mock_llm.call_async = AsyncMock(return_value=name_json)
        # OOB index 99 should be ignored, not raise IndexError
        mock_llm.call.return_value = '{"groups": [[0, 99], [1]]}'
        mock_llm_cls.return_value = mock_llm

        pipeline = GoalDiscoveryPipeline(
            clustering_method="kmeans", k_range=(2, 2), min_words=1, merge_similar=True
        )
        result = pipeline.discover(convs)
        assert len(result.goals) == 2


# ── clusterer.best_k ──────────────────────────────────────────────────────────


class TestBestK:
    def test_corpus_smaller_than_min_k_does_not_raise(self) -> None:
        from arksim.scenario.goal_discovery.clusterer import best_k

        rng = np.random.default_rng(0)
        # 4 samples but k_range starts at 5; previously caused sklearn ValueError
        emb = rng.standard_normal((4, 8)).astype(np.float32)
        k = best_k(emb, k_range=(5, 30))
        assert 2 <= k <= 3  # clamped to n-1=3


# ── routes_arkdock_discovery._parse_record ────────────────────────────────────


class TestParseRecord:
    def test_messages_key_routes_to_conversations_record(self) -> None:
        from arksim.ui.api.routes_arkdock_discovery import _parse_record

        record = {
            "id": "c1",
            "messages": [{"role": "user", "content": "Where is my package?"}],
        }
        conv = _parse_record(record)
        assert conv.first_user_turn() == "Where is my package?"

    def test_no_messages_key_routes_to_flat_record(self) -> None:
        from arksim.ui.api.routes_arkdock_discovery import _parse_record

        record = {
            "user_question": "What do I need before buying?",
            "summarized_answers": "Here is what you need.",
        }
        conv = _parse_record(record)
        assert conv.first_user_turn() == "What do I need before buying?"


# ── is_negative_emotion ───────────────────────────────────────────────────────


class TestIsNegativeEmotion:
    def test_detects_frustration(self) -> None:
        from arksim.scenario.goal_discovery.preprocessing import is_negative_emotion

        assert is_negative_emotion("I am so frustrated with this service")

    def test_detects_profanity(self) -> None:
        from arksim.scenario.goal_discovery.preprocessing import is_negative_emotion

        assert is_negative_emotion("This is fucking broken")

    def test_detects_terrible(self) -> None:
        from arksim.scenario.goal_discovery.preprocessing import is_negative_emotion

        assert is_negative_emotion("The quality is terrible")

    def test_clean_text_returns_false(self) -> None:
        from arksim.scenario.goal_discovery.preprocessing import is_negative_emotion

        assert not is_negative_emotion("I would like to track my order please")

    def test_no_false_positive_on_worst_case(self) -> None:
        from arksim.scenario.goal_discovery.preprocessing import is_negative_emotion

        # "worst" is a negative word but let's make sure normal text does not match
        assert not is_negative_emotion("I want to check my order status")

    def test_negative_emotion_count_on_goal_cluster(self) -> None:
        from arksim.scenario.goal_discovery.models import GoalCluster

        cluster = GoalCluster(
            id="goal_000",
            name="Test",
            description="",
            exemplars=[],
            size=10,
            prevalence=1.0,
            negative_emotion_count=3,
        )
        assert cluster.negative_emotion_count == 3


# ── Arkdock artifact conversion ───────────────────────────────────────────────


class TestArkdockArtifacts:
    def _make_result(self) -> object:
        from arksim.scenario.goal_discovery.models import (
            GoalCluster,
            GoalDiscoveryResult,
        )

        goals = [
            GoalCluster(
                id="goal_000",
                name="Track Order Status",
                description="User wants to check their order status.",
                exemplars=["where is my order", "track my shipment"],
                size=80,
                prevalence=0.8,
                negative_emotion_count=5,
            ),
            GoalCluster(
                id="goal_001",
                name="Cancel an Order",
                description="User wants to cancel a purchase.",
                exemplars=["I want to cancel this fucking order", "cancel my order"],
                size=20,
                prevalence=0.2,
                negative_emotion_count=8,
            ),
        ]
        return GoalDiscoveryResult(
            goals=goals,
            method="goal_discovery",
            n_input=100,
            metadata={"n_clustered": 100, "n_noise": 0},
        )

    def test_approved_attributes_maps_goals(self) -> None:
        from arksim.scenario.goal_discovery.arkdock import to_arkdock_artifacts

        artifacts = to_arkdock_artifacts(self._make_result())
        attrs = artifacts["approved_attributes"]
        assert len(attrs) == 2
        assert attrs[0]["concept_label"] == "Track Order Status"
        assert attrs[0]["attribute_category"] == "goal"
        assert attrs[0]["metrics"]["support_turns"] == 80
        assert attrs[0]["metrics"]["negative_emotion_count"] == 5

    def test_failure_topics_only_negative(self) -> None:
        from arksim.scenario.goal_discovery.arkdock import to_arkdock_artifacts

        artifacts = to_arkdock_artifacts(self._make_result())
        ft = artifacts["failure_topics"]
        # failure_topics is wrapped as {"knowledge_topics": [...]} for arkdock-python
        assert isinstance(ft, dict)
        topics = ft["knowledge_topics"]
        # Both clusters have negative_emotion_count > 0
        assert len(topics) == 2
        names = {t["concept_label"] for t in topics}
        assert "Cancel an Order" in names

    def test_dialogue_rules_is_empty_list(self) -> None:
        from arksim.scenario.goal_discovery.arkdock import to_arkdock_artifacts

        artifacts = to_arkdock_artifacts(self._make_result())
        assert artifacts["dialogue_rules"] == []

    def test_discovery_summary_counts(self) -> None:
        from arksim.scenario.goal_discovery.arkdock import to_arkdock_artifacts

        artifacts = to_arkdock_artifacts(self._make_result())
        summary = artifacts["discovery_summary"]
        assert summary["conversation_count"] == 100
        assert summary["approved_attributes"] == 2
        assert summary["failure_like_turns"] == 13  # 5 + 8
        assert summary["user_turn_count"] == 100

    def test_config_maps_to_pipeline(self) -> None:
        from arksim.scenario.goal_discovery.arkdock import ArkdockDiscoveryConfig

        cfg = ArkdockDiscoveryConfig(approved_top_k=15, min_support=5)
        pipeline = cfg.to_pipeline()
        assert pipeline.k_range == (2, 15)
        assert pipeline.min_cluster_size == 5

    def test_config_defaults(self) -> None:
        from arksim.scenario.goal_discovery.arkdock import ArkdockDiscoveryConfig

        cfg = ArkdockDiscoveryConfig()
        assert cfg.approved_top_k == 20
        assert cfg.min_support == 3
        assert cfg.clustering_method == "kmeans"
