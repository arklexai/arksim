# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
import logging

import numpy as np
from pydantic import BaseModel

from arksim.llms.chat.llm import LLM
from arksim.scenario.goal_discovery.clusterer import (
    best_k,
    hdbscan_cluster,
    kmeans_cluster,
    select_exemplars,
)
from arksim.scenario.goal_discovery.embedder import build_embedding_service
from arksim.scenario.goal_discovery.models import (
    ConversationInput,
    GoalCluster,
    GoalDiscoveryResult,
)
from arksim.scenario.goal_discovery.preprocessing import (
    clean_text,
    contains_profanity,
    extract_first_turns,
    is_negative_emotion,
)
from arksim.scenario.goal_discovery.prompts import (
    CLUSTER_NAMING_PROMPT,
    CLUSTER_NAMING_SYSTEM,
    MERGE_SIMILAR_GOALS_PROMPT,
    MERGE_SIMILAR_GOALS_SYSTEM,
)

logger = logging.getLogger(__name__)


class _ClusterName(BaseModel):
    name: str
    description: str
    intent_type: str


class _MergeGroups(BaseModel):
    groups: list[list[int]]


class GoalDiscoveryPipeline:
    """Discover user goals via embedding clustering and LLM cluster naming.

    Pipeline:
        1. Extract first user turns from each conversation.
        2. Embed turns with a local sentence-transformer model.
        3. Cluster embeddings (HDBSCAN or K-Means).
        4. For each cluster, select representative exemplars.
        5. Ask the LLM to name each cluster concurrently.
        6. Optionally merge near-duplicate goal names.

    Args:
        embedding_model: sentence-transformers model name.
        clustering_method: "hdbscan" or "kmeans".
        min_cluster_size: HDBSCAN minimum cluster size.
        k_range: (min_k, max_k) silhouette sweep range for K-Means auto-selection.
        exemplar_count: Number of representative messages per cluster for naming.
        min_words: Minimum word count to accept a first user turn.
        llm_model: Chat model for cluster naming and merging.
        llm_provider: Provider for the chat LLM (e.g. "openai", "anthropic").
        merge_similar: Whether to run a second LLM pass to merge near-duplicate names.
        naming_concurrency: Max concurrent LLM calls during cluster naming.
    """

    def __init__(
        self,
        embedding_model: str | None = None,
        embedding_provider: str = "openai",
        clustering_method: str = "hdbscan",
        min_cluster_size: int = 10,
        k_range: tuple[int, int] = (5, 30),
        exemplar_count: int = 6,
        min_words: int = 3,
        llm_model: str = "gpt-4o-mini",
        llm_provider: str = "openai",
        merge_similar: bool = True,
        naming_concurrency: int = 8,
        max_input: int | None = None,
        filter_exemplars: bool = True,
    ) -> None:
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.clustering_method = clustering_method
        self.min_cluster_size = min_cluster_size
        self.k_range = k_range
        self.exemplar_count = exemplar_count
        self.min_words = min_words
        self.llm_model = llm_model
        self.llm_provider = llm_provider
        self.merge_similar = merge_similar
        self.naming_concurrency = naming_concurrency
        self.max_input = max_input
        self.filter_exemplars = filter_exemplars

    def discover(self, conversations: list[ConversationInput]) -> GoalDiscoveryResult:
        """Run the full LLM-light goal discovery pipeline.

        Args:
            conversations: Input conversations from any source schema.

        Returns:
            GoalDiscoveryResult with discovered goal clusters.
        """
        logger.info("LLM-light discovery: %d input conversations", len(conversations))

        # Step 1: Extract and clean first user turns
        indexed_turns = extract_first_turns(conversations, min_words=self.min_words)
        if not indexed_turns:
            logger.warning(
                "No qualifying first user turns found. Returning empty result."
            )
            return GoalDiscoveryResult(
                goals=[], method="goal_discovery", n_input=len(conversations)
            )

        if self.max_input and len(indexed_turns) > self.max_input:
            indexed_turns = list(indexed_turns[: self.max_input])
            logger.info("Capped input to %d turns (max_input)", self.max_input)

        indices, texts = zip(*indexed_turns, strict=False)
        texts = [clean_text(t) for t in texts]
        logger.info("Extracted %d qualifying first turns", len(texts))

        # Step 2: Embed
        embedder = build_embedding_service(
            self.embedding_provider, self.embedding_model
        )
        resolved_embedding_model = embedder.model_name
        embeddings = embedder.embed(list(texts))
        logger.info("Embedded %d turns (dim=%d)", len(texts), embeddings.shape[1])

        # Step 3: Cluster
        labels, cluster_ids = self._cluster(embeddings)
        logger.info("Found %d clusters (noise excluded)", len(cluster_ids))

        if not cluster_ids:
            logger.warning(
                "Clustering produced no clusters. Try reducing min_cluster_size."
            )
            return GoalDiscoveryResult(
                goals=[], method="goal_discovery", n_input=len(conversations)
            )

        # Step 4: Select exemplars per cluster, dropping under-sized ones.
        # K-Means assigns every point to a cluster regardless of size, so we
        # apply min_cluster_size here for both methods.
        exemplar_text_filter = contains_profanity if self.filter_exemplars else None
        cluster_exemplars: dict[int, list[str]] = {}
        cluster_sizes: dict[int, int] = {}
        cluster_neg_counts: dict[int, int] = {}
        for cid in cluster_ids:
            mask = labels == cid
            size = int(np.sum(mask))
            if size < self.min_cluster_size:
                continue
            cluster_texts = [texts[i] for i, m in enumerate(mask.tolist()) if m]
            exemplars = select_exemplars(
                embeddings,
                labels,
                texts,
                cid,
                n=self.exemplar_count,
                text_filter=exemplar_text_filter,
            )
            cluster_exemplars[cid] = exemplars
            cluster_sizes[cid] = size
            cluster_neg_counts[cid] = sum(
                1 for t in cluster_texts if is_negative_emotion(t)
            )
        cluster_ids = sorted(cluster_sizes)
        if not cluster_ids:
            logger.warning(
                "All clusters fell below min_cluster_size=%d. "
                "Try reducing min_cluster_size or providing more conversations.",
                self.min_cluster_size,
            )
            return GoalDiscoveryResult(
                goals=[], method="goal_discovery", n_input=len(conversations)
            )
        logger.info(
            "%d clusters passed min_cluster_size=%d filter",
            len(cluster_ids),
            self.min_cluster_size,
        )

        # Step 5: Name clusters concurrently via LLM
        llm = LLM(model=self.llm_model, provider=self.llm_provider)
        coro = self._name_clusters_async(llm, cluster_ids, cluster_exemplars)
        try:
            asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                named = pool.submit(asyncio.run, coro).result()
        except RuntimeError:
            named = asyncio.run(coro)

        # Step 6: Optionally merge near-duplicates
        if self.merge_similar and len(named) > 1:
            named = self._merge_similar(
                llm, named, cluster_sizes, cluster_exemplars, cluster_neg_counts
            )

        total_clustered = sum(cluster_sizes[cid] for cid in named)
        goals: list[GoalCluster] = []
        for i, (cid, name_result) in enumerate(named.items()):
            goals.append(
                GoalCluster(
                    id=f"goal_{i:03d}",
                    name=name_result.name,
                    description=name_result.description,
                    exemplars=cluster_exemplars[cid],
                    size=cluster_sizes[cid],
                    prevalence=cluster_sizes[cid] / max(total_clustered, 1),
                    negative_emotion_count=cluster_neg_counts.get(cid, 0),
                )
            )

        goals.sort(key=lambda g: g.prevalence, reverse=True)
        logger.info("Discovery complete: %d goal clusters", len(goals))

        return GoalDiscoveryResult(
            goals=goals,
            method="goal_discovery",
            n_input=len(conversations),
            metadata={
                "clustering_method": self.clustering_method,
                "n_clustered": total_clustered,
                "n_noise": int(np.sum(labels == -1)),
                "embedding_model": resolved_embedding_model,
            },
        )

    def _cluster(self, embeddings: np.ndarray) -> tuple[np.ndarray, list[int]]:
        """Run the configured clustering method and return (labels, cluster_ids)."""
        if self.clustering_method == "hdbscan":
            labels = hdbscan_cluster(embeddings, min_cluster_size=self.min_cluster_size)
        elif self.clustering_method == "kmeans":
            k = best_k(embeddings, k_range=self.k_range)
            logger.info("K-Means auto-selected k=%d", k)
            labels, _ = kmeans_cluster(embeddings, k=k)
        else:
            raise ValueError(
                f"Unknown clustering_method: {self.clustering_method!r}. "
                "Use 'hdbscan' or 'kmeans'."
            )

        cluster_ids = sorted(cid for cid in set(labels.tolist()) if cid != -1)
        return labels, cluster_ids

    async def _name_clusters_async(
        self,
        llm: LLM,
        cluster_ids: list[int],
        cluster_exemplars: dict[int, list[str]],
    ) -> dict[int, _ClusterName]:
        """Name all clusters concurrently, respecting naming_concurrency."""
        semaphore = asyncio.Semaphore(self.naming_concurrency)

        async def name_one(cid: int) -> tuple[int, _ClusterName]:
            exemplars = cluster_exemplars[cid]
            bullet_list = "\n".join(f"- {e}" for e in exemplars)
            prompt = CLUSTER_NAMING_PROMPT.format(
                n=len(exemplars), exemplars=bullet_list
            )
            messages = [
                {"role": "system", "content": CLUSTER_NAMING_SYSTEM},
                {"role": "user", "content": prompt},
            ]
            async with semaphore:
                raw = await llm.call_async(messages)

            try:
                data = json.loads(raw)
                result = _ClusterName(**data)
            except Exception:
                logger.warning("Failed to parse cluster name for cluster %d", cid)
                result = _ClusterName(
                    name=f"Goal Cluster {cid}",
                    description="Cluster naming failed.",
                    intent_type="informational",
                )
            return cid, result

        pairs = await asyncio.gather(*[name_one(cid) for cid in cluster_ids])
        return dict(pairs)

    def _merge_similar(
        self,
        llm: LLM,
        named: dict[int, _ClusterName],
        cluster_sizes: dict[int, int],
        cluster_exemplars: dict[int, list[str]],
        cluster_neg_counts: dict[int, int],
    ) -> dict[int, _ClusterName]:
        """Ask the LLM to group near-duplicate goal names and merge them."""
        cids = list(named.keys())
        goal_names = [named[cid].name for cid in cids]
        bullet_list = "\n".join(f"{i}. {n}" for i, n in enumerate(goal_names))
        prompt = MERGE_SIMILAR_GOALS_PROMPT.format(n=len(goal_names), goals=bullet_list)
        messages = [
            {"role": "system", "content": MERGE_SIMILAR_GOALS_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        try:
            raw = llm.call(messages)
            merge_result = _MergeGroups(**json.loads(raw))
        except Exception:
            logger.warning("Merge pass failed; returning un-merged clusters.")
            return named

        merged: dict[int, _ClusterName] = {}
        seen_indices: set[int] = set()
        for group in merge_result.groups:
            if not group:
                continue
            valid_group = [i for i in group if 0 <= i < len(cids)]
            if not valid_group:
                continue
            primary_idx = max(valid_group, key=lambda i: cluster_sizes[cids[i]])
            primary_cid = cids[primary_idx]

            for idx in valid_group:
                if idx != primary_idx:
                    other_cid = cids[idx]
                    cluster_exemplars[primary_cid] = (
                        cluster_exemplars[primary_cid] + cluster_exemplars[other_cid]
                    )[: self.exemplar_count]
                    cluster_sizes[primary_cid] += cluster_sizes[other_cid]
                    cluster_neg_counts[primary_cid] += cluster_neg_counts.get(
                        other_cid, 0
                    )

            merged[primary_cid] = named[primary_cid]
            seen_indices.update(valid_group)

        for idx, cid in enumerate(cids):
            if idx not in seen_indices:
                merged[cid] = named[cid]

        return merged
