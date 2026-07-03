# SPDX-License-Identifier: Apache-2.0
"""Utilities for integrating goal discovery with the Arkdock backend.

Converts a GoalDiscoveryResult into the artifact shape that
attribute_discovery_run.artifacts expects (spec §17.9), and maps the
discovery_config object from the Go dispatch request to LLMLightGoalDiscovery
constructor arguments.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from arksim.scenario.goal_discovery.llm_light import LLMLightGoalDiscovery
from arksim.scenario.goal_discovery.models import GoalDiscoveryResult
from arksim.scenario.goal_discovery.preprocessing import is_negative_emotion


class ArkdockDiscoveryConfig(BaseModel):
    """Subset of LLMLightGoalDiscovery knobs exposed via discovery_config.

    Field names mirror the arkdock-python SDK's DiscoverAttributesInput so the
    Go backend can pass discovery_config through unchanged.
    """

    approved_top_k: int = Field(
        default=20,
        description="Maximum number of goal clusters to surface (k_range upper bound).",
    )
    min_support: int = Field(
        default=3,
        description="Minimum number of conversations per cluster (min_cluster_size).",
    )
    max_input: int | None = Field(
        default=None,
        description="Cap on the number of first user turns to embed. None = no cap.",
    )
    clustering_method: str = Field(
        default="kmeans",
        description="'kmeans' or 'hdbscan'.",
    )
    merge_similar: bool = Field(
        default=True,
        description="Whether to run a second LLM pass to merge near-duplicate goal names.",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Chat model for cluster naming and merging.",
    )
    llm_provider: str = Field(
        default="openai",
        description="Provider for the chat LLM.",
    )
    embedding_provider: str = Field(
        default="openai",
        description="'openai' or 'sentence-transformers'.",
    )
    embedding_model: str | None = Field(
        default=None,
        description="Embedding model override. None picks the provider default.",
    )

    def to_llm_light(self) -> LLMLightGoalDiscovery:
        """Build a LLMLightGoalDiscovery instance from this config."""
        return LLMLightGoalDiscovery(
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model,
            clustering_method=self.clustering_method,
            min_cluster_size=self.min_support,
            k_range=(2, self.approved_top_k),
            llm_model=self.llm_model,
            llm_provider=self.llm_provider,
            merge_similar=self.merge_similar,
            max_input=self.max_input,
        )


def to_arkdock_artifacts(result: GoalDiscoveryResult) -> dict[str, Any]:
    """Convert a GoalDiscoveryResult to the attribute_discovery_run artifacts shape.

    Maps the four artifact keys from spec §17.9:
      - approved_attributes: one entry per GoalCluster (attribute_category="goal")
      - failure_topics:      clusters with negative_emotion_count > 0
      - dialogue_rules:      empty list (not yet supported by LLM light)
      - discovery_summary:   run-level statistics from GoalDiscoveryResult.metadata
    """
    total_neg = sum(g.negative_emotion_count for g in result.goals)
    n_clustered: int = result.metadata.get(
        "n_clustered", sum(g.size for g in result.goals)
    )

    approved_attributes = [
        {
            "concept_label": g.name,
            "attribute_category": "goal",
            "attribute_type": g.description,
            "metrics": {
                "support_turns": g.size,
                "contrast_score": round(g.prevalence, 4),
                "negative_emotion_count": g.negative_emotion_count,
            },
            "examples": g.exemplars,
        }
        for g in result.goals
    ]

    failure_topics = [
        {
            "concept_label": g.name,
            "metrics": {"support_turns": g.negative_emotion_count},
            # Prefer exemplars that themselves express negative emotion; fall back
            # to the first available exemplar if none pass the filter.
            "examples": (
                [ex for ex in g.exemplars if is_negative_emotion(ex)][:3]
                or g.exemplars[:1]
            ),
        }
        for g in result.goals
        if g.negative_emotion_count > 0
    ]

    discovery_summary = {
        "conversation_count": result.n_input,
        "user_turn_count": n_clustered,
        "failure_like_turns": total_neg,
        "success_like_turns": max(0, n_clustered - total_neg),
        "raw_candidates": n_clustered,
        "deduplicated_candidates": n_clustered,
        "approved_attributes": len(result.goals),
    }

    return {
        "approved_attributes": approved_attributes,
        "failure_topics": failure_topics,
        "dialogue_rules": [],
        "discovery_summary": discovery_summary,
    }
