# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
import logging

from arksim.llms.chat.llm import LLM
from arksim.scenario.goal_discovery.models import (
    ConversationExtractionResult,
    ConversationInput,
    ExtractionFact,
)
from arksim.scenario.goal_discovery.prompts import EXTRACTION_PROMPT, EXTRACTION_SYSTEM

logger = logging.getLogger(__name__)


class ConversationExtractor:
    """Per-conversation LLM structured extraction (pipeline step 3a).

    For each conversation the LLM reads the full dialogue and returns a list
    of typed facts: {attribute, value, confidence, provenance_span, source_id}.
    All conversations are processed concurrently up to `concurrency` in-flight
    requests at once.

    Args:
        llm_model: Chat model to use for extraction.
        llm_provider: Provider for the chat LLM (e.g. "openai", "anthropic").
        concurrency: Max concurrent LLM calls.
        attributes: List of attribute definitions to extract. Each entry is a
            dict with "name" and "description" keys. Defaults to
            DEFAULT_ATTRIBUTES when not provided.
    """

    DEFAULT_ATTRIBUTES: list[dict[str, str]] = [
        {
            "name": "goal",
            "description": "The primary thing the user is trying to accomplish",
        },
        {
            "name": "product_mentioned",
            "description": "Any specific product, model number, or brand mentioned",
        },
        {
            "name": "issue_type",
            "description": (
                "If troubleshooting: the category of problem "
                "(e.g. leaking, not cooling, won't start)"
            ),
        },
        {
            "name": "urgency",
            "description": "How urgent the user seems: low, medium, or high",
        },
    ]

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        llm_provider: str = "openai",
        concurrency: int = 16,
        attributes: list[dict[str, str]] | None = None,
    ) -> None:
        self.llm_model = llm_model
        self.llm_provider = llm_provider
        self.concurrency = concurrency
        self.attributes = (
            attributes if attributes is not None else self.DEFAULT_ATTRIBUTES
        )

    def extract_all(
        self,
        conversations: list[ConversationInput],
        source_id_key: str | None = "session_id",
    ) -> list[ConversationExtractionResult]:
        """Run per-conversation LLM extraction over all conversations.

        Args:
            conversations: Input conversations.
            source_id_key: Key in conversation.meta to use as source_id.
                Falls back to the positional index when the key is absent or
                source_id_key is None.

        Returns:
            One ConversationExtractionResult per input conversation, in the
            same order as the input list.
        """
        return asyncio.run(self._extract_all_async(conversations, source_id_key))

    async def _extract_all_async(
        self,
        conversations: list[ConversationInput],
        source_id_key: str | None,
    ) -> list[ConversationExtractionResult]:
        semaphore = asyncio.Semaphore(self.concurrency)
        llm = LLM(model=self.llm_model, provider=self.llm_provider)
        tasks = [
            self._extract_one(idx, conv, llm, semaphore, source_id_key)
            for idx, conv in enumerate(conversations)
        ]
        return list(await asyncio.gather(*tasks))

    async def _extract_one(
        self,
        idx: int,
        conv: ConversationInput,
        llm: LLM,
        semaphore: asyncio.Semaphore,
        source_id_key: str | None,
    ) -> ConversationExtractionResult:
        source_id = str(conv.meta.get(source_id_key, idx) if source_id_key else idx)

        conversation_text = "\n".join(
            f"{t.get('role', 'user').capitalize()}: {t.get('content', '').strip()}"
            for t in conv.turns
            if t.get("content", "").strip()
        )
        attribute_lines = "\n".join(
            f"- {a['name']}: {a['description']}" for a in self.attributes
        )

        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {
                "role": "user",
                "content": EXTRACTION_PROMPT.format(
                    conversation=conversation_text,
                    attributes=attribute_lines,
                ),
            },
        ]

        async with semaphore:
            raw = await llm.call_async(messages)

        facts: list[ExtractionFact] = []
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    facts.append(
                        ExtractionFact(
                            attribute=str(item.get("attribute", "")),
                            value=str(item.get("value", "")),
                            confidence=float(item.get("confidence", 0.0)),
                            provenance_span=str(item.get("provenance_span", "")),
                            source_id=source_id,
                        )
                    )
        except Exception:
            logger.warning("Failed to parse extraction result for conversation %d", idx)

        return ConversationExtractionResult(source_id=source_id, facts=facts)
