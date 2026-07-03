# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

GoalSource = Literal["clustering", "provided_intent", "llm_extraction"]


@dataclass
class ExtractionFact:
    """A single structured fact extracted from one conversation by the LLM.

    Fields mirror the 3a spec: attribute, value, confidence, provenance_span,
    and source_id (which conversation this came from).
    """

    attribute: str
    value: str
    confidence: float
    provenance_span: str
    source_id: str


@dataclass
class ConversationExtractionResult:
    """All facts extracted from a single conversation."""

    source_id: str
    facts: list[ExtractionFact]


@dataclass
class GoalCluster:
    """A discovered user goal category."""

    id: str
    name: str
    description: str
    exemplars: list[str]
    size: int
    prevalence: float
    centroid: np.ndarray | None = None
    confidence: float = 1.0
    source: GoalSource = "clustering"
    negative_emotion_count: int = 0


@dataclass
class GoalDiscoveryResult:
    """Output of a goal discovery run."""

    goals: list[GoalCluster]
    method: Literal["llm_light", "llm_heavy"]
    n_input: int
    metadata: dict = field(default_factory=dict)

    def to_goal_list(self) -> list[dict]:
        """Serialize to a list compatible with downstream Stage 2 goal input."""
        return [
            {
                "goal": g.name,
                "description": g.description,
                "exemplars": g.exemplars,
                "prevalence": g.prevalence,
                "size": g.size,
                "confidence": g.confidence,
                "source": g.source,
                "negative_emotion_count": g.negative_emotion_count,
            }
            for g in self.goals
        ]


USER_ROLES: frozenset[str] = frozenset({"user", "human", "customer"})

# MAA meta fields carried into ConversationInput.meta
_MAA_META_FIELDS: tuple[str, ...] = (
    "intent",
    "page_type",
    "item_id",
    "store_id",
    "customer_type",
    "session_id",
    "mcvis_id",
    "svoc_id",
    "device_type",
    "language_code",
    "image_id",
    "action_id",
    "input_type",
    "model_name",
    "zip_code",
)


@dataclass
class ConversationInput:
    """A single conversation from any source schema."""

    turns: list[dict]
    meta: dict = field(default_factory=dict)

    def first_user_turn(self, min_words: int = 0) -> str | None:
        """Return the first user turn that meets a minimum word count.

        Args:
            min_words: Skip turns with fewer than this many words.
                       Set to 0 to accept any non-empty turn.
        """
        for turn in self.turns:
            if turn.get("role", "").lower() not in USER_ROLES:
                continue
            content = turn.get("content", "").strip()
            if not content:
                continue
            if min_words and len(content.split()) < min_words:
                continue
            return content
        return None

    @classmethod
    def from_maa_record(cls, record: dict) -> ConversationInput:
        """Build a ConversationInput from a single MAA chat history row.

        The MAA schema is a flat record (one row = one user question +
        one assistant response). The user_question becomes the user turn;
        summarized_answers becomes the assistant turn.

        reformulated_user_question — the assistant-rewritten version of the
        question, already normalised for retrieval — is stored in
        meta["reformulated_question"] so extract_first_turns() can prefer it
        over the raw user_question for embedding.
        """
        turns: list[dict] = []
        if record.get("user_question"):
            turns.append({"role": "user", "content": record["user_question"]})
        if record.get("summarized_answers"):
            turns.append({"role": "assistant", "content": record["summarized_answers"]})

        meta: dict = {
            k: record[k] for k in _MAA_META_FIELDS if record.get(k) is not None
        }
        if record.get("reformulated_user_question"):
            meta["reformulated_question"] = record["reformulated_user_question"]

        return cls(turns=turns, meta=meta)

    @classmethod
    def from_conversations_record(cls, record: dict) -> ConversationInput:
        """Build a ConversationInput from a conversations-sample.json record.

        Schema:
          id, created_at, modified_at, user_rating  — carried into meta
          messages  — list of {role, content, intent?} turns
          category, tags  — optional top-level fields, carried into meta

        The intent from the first user message is stored in meta["intent"]
        so extract_first_turns() and the dry-run preview can read it.
        """
        turns: list[dict] = [
            {k: v for k, v in msg.items() if v is not None}
            for msg in record.get("messages") or []
        ]

        meta: dict = {}
        for key in (
            "id",
            "created_at",
            "modified_at",
            "user_rating",
            "category",
            "tags",
        ):
            val = record.get(key)
            if val is not None:
                meta[key] = val

        # Surface the first user-turn intent at the top level for easy lookup.
        for turn in turns:
            if turn.get("role", "").lower() in USER_ROLES and turn.get("intent"):
                meta.setdefault("intent", turn["intent"])
                break

        return cls(turns=turns, meta=meta)
