# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter, defaultdict

from arksim.llms.chat.llm import LLM
from arksim.scenario.goal_discovery.extractor import ConversationExtractor
from arksim.scenario.goal_discovery.models import (
    ConversationExtractionResult,
    ConversationInput,
    GoalCluster,
    GoalDiscoveryResult,
)
from arksim.scenario.goal_discovery.preprocessing import is_negative_emotion
from arksim.scenario.goal_discovery.prompts import (
    CANONICALIZE_GOALS_PROMPT,
    CANONICALIZE_GOALS_SYSTEM,
    MERGE_SIMILAR_GOALS_PROMPT,
    MERGE_SIMILAR_GOALS_SYSTEM,
)

logger = logging.getLogger(__name__)


class LLMHeavyGoalDiscovery:
    """Discover user goals via per-conversation LLM extraction and canonicalization.

    Pipeline:
        Stage 1 — per-conversation extraction:
            For each conversation the LLM extracts a structured "goal" fact.
            All conversations are processed concurrently up to `concurrency`
            in-flight requests.

        Stage 2 — batch canonicalization:
            Raw extracted goals are batched (up to `batch_size` per prompt)
            and sent to the LLM for normalization into a canonical taxonomy.
            Near-duplicates across batches are merged in a final pass.

    Args:
        llm_model: Chat model for extraction and canonicalization.
        llm_provider: Provider for the chat LLM (e.g. "openai", "anthropic").
        concurrency: Max concurrent LLM calls during stage 1.
        batch_size: Number of raw goals packed into one canonicalization prompt.
        goal_attribute: Which attribute name from stage 1 to treat as the goal.
        extractor_attributes: Custom attribute list for stage 1. Defaults to
            ConversationExtractor.DEFAULT_ATTRIBUTES.
    """

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        llm_provider: str = "openai",
        concurrency: int = 16,
        batch_size: int = 50,
        goal_attribute: str = "goal",
        extractor_attributes: list[dict[str, str]] | None = None,
    ) -> None:
        self.llm_model = llm_model
        self.llm_provider = llm_provider
        self.concurrency = concurrency
        self.batch_size = batch_size
        self.goal_attribute = goal_attribute
        self.extractor_attributes = extractor_attributes

    def discover(self, conversations: list[ConversationInput]) -> GoalDiscoveryResult:
        """Run the full LLM-heavy goal discovery pipeline.

        Args:
            conversations: Input conversations from any source schema.

        Returns:
            GoalDiscoveryResult with discovered goal clusters.
        """
        logger.info("LLM-heavy discovery: %d input conversations", len(conversations))

        extractor = ConversationExtractor(
            llm_model=self.llm_model,
            llm_provider=self.llm_provider,
            concurrency=self.concurrency,
            attributes=self.extractor_attributes,
        )
        extraction_results = extractor.extract_all(conversations)
        logger.info(
            "Stage 1 complete: extracted facts from %d conversations",
            len(extraction_results),
        )

        return self.canonicalize(extraction_results, n_input=len(conversations))

    def canonicalize(
        self,
        extraction_results: list[ConversationExtractionResult],
        n_input: int | None = None,
    ) -> GoalDiscoveryResult:
        """Run stage 2 over pre-computed extraction results.

        Useful when stage 1 was run separately (e.g. to inspect intermediate
        facts before canonicalization).

        Args:
            extraction_results: Output of ConversationExtractor.extract_all().
            n_input: Original conversation count. Defaults to len(extraction_results).

        Returns:
            GoalDiscoveryResult with discovered goal clusters.
        """
        if n_input is None:
            n_input = len(extraction_results)

        # Collect (source_id, raw_goal, exemplar) for each goal fact
        goal_items: list[tuple[str, str, str]] = []
        for result in extraction_results:
            for fact in result.facts:
                if fact.attribute == self.goal_attribute and fact.value.strip():
                    exemplar = fact.provenance_span.strip() or fact.value
                    goal_items.append((result.source_id, fact.value.strip(), exemplar))

        if not goal_items:
            logger.warning("No goal facts found in extraction results.")
            return GoalDiscoveryResult(goals=[], method="llm_heavy", n_input=n_input)

        logger.info("Stage 2: canonicalizing %d raw goal facts", len(goal_items))

        # Batch canonicalize (parallel LLM calls, one per batch)
        batches = [
            goal_items[i : i + self.batch_size]
            for i in range(0, len(goal_items), self.batch_size)
        ]
        batch_results = asyncio.run(self._canonicalize_batches_async(batches))

        # Assign a canonical name string to each goal_item
        assigned_names: list[str] = []
        desc_map: dict[str, str] = {}

        for batch, result in zip(batches, batch_results, strict=False):
            canonicals: list[str] = result.get("canonical") or []
            descriptions: list[str] = result.get("descriptions") or []
            assignments: list[int] = result.get("assignments") or list(
                range(len(batch))
            )

            for name, desc in zip(canonicals, descriptions, strict=False):
                desc_map.setdefault(name, desc)

            for local_i, (_, raw_goal, _) in enumerate(batch):
                if canonicals:
                    idx = int(assignments[local_i]) if local_i < len(assignments) else 0
                    idx = max(0, min(idx, len(canonicals) - 1))
                    assigned_names.append(canonicals[idx])
                else:
                    assigned_names.append(raw_goal)

        # Merge near-duplicate canonical names that appeared across different batches
        unique_canonicals = list(dict.fromkeys(assigned_names))
        if len(batches) > 1 and len(unique_canonicals) > 1:
            logger.info(
                "Stage 2 merge: %d unique canonicals across %d batches",
                len(unique_canonicals),
                len(batches),
            )
            remap = self._final_merge(unique_canonicals, assigned_names)
        else:
            remap = {name: name for name in unique_canonicals}

        final_assigned = [remap.get(n, n) for n in assigned_names]

        final_desc_map: dict[str, str] = {}
        for name, desc in desc_map.items():
            final_desc_map.setdefault(remap.get(name, name), desc)

        # Aggregate: group exemplar texts and count by canonical name
        groups: dict[str, list[str]] = defaultdict(list)
        neg_counts: dict[str, int] = defaultdict(int)
        for (_, raw_goal, exemplar), final_name in zip(
            goal_items, final_assigned, strict=False
        ):
            groups[final_name].append(exemplar)
            if is_negative_emotion(exemplar) or is_negative_emotion(raw_goal):
                neg_counts[final_name] += 1

        total = len(goal_items)
        goals: list[GoalCluster] = []
        for i, (canonical_name, exemplar_list) in enumerate(
            sorted(groups.items(), key=lambda x: -len(x[1]))
        ):
            seen: set[str] = set()
            deduped: list[str] = []
            for ex in exemplar_list:
                if ex not in seen and len(deduped) < 6:
                    seen.add(ex)
                    deduped.append(ex)

            goals.append(
                GoalCluster(
                    id=f"goal_{i:03d}",
                    name=canonical_name,
                    description=final_desc_map.get(canonical_name, ""),
                    exemplars=deduped,
                    size=len(exemplar_list),
                    prevalence=len(exemplar_list) / max(total, 1),
                    source="llm_extraction",
                    negative_emotion_count=neg_counts.get(canonical_name, 0),
                )
            )

        n_with_goals = len(
            {
                r.source_id
                for r in extraction_results
                if any(f.attribute == self.goal_attribute for f in r.facts)
            }
        )
        logger.info("Stage 2 complete: %d canonical goal clusters", len(goals))
        return GoalDiscoveryResult(
            goals=goals,
            method="llm_heavy",
            n_input=n_input,
            metadata={
                "n_goal_facts": total,
                "n_canonical": len(goals),
                "n_conversations_with_goals": n_with_goals,
            },
        )

    async def _canonicalize_batches_async(
        self, batches: list[list[tuple[str, str, str]]]
    ) -> list[dict]:
        semaphore = asyncio.Semaphore(self.concurrency)
        llm = LLM(model=self.llm_model, provider=self.llm_provider)
        tasks = [
            self._canonicalize_one_batch(batch, llm, semaphore) for batch in batches
        ]
        return list(await asyncio.gather(*tasks))

    async def _canonicalize_one_batch(
        self,
        batch: list[tuple[str, str, str]],
        llm: LLM,
        semaphore: asyncio.Semaphore,
    ) -> dict:
        goals_text = "\n".join(
            f"{i + 1}. {raw_goal}" for i, (_, raw_goal, _) in enumerate(batch)
        )
        messages = [
            {"role": "system", "content": CANONICALIZE_GOALS_SYSTEM},
            {
                "role": "user",
                "content": CANONICALIZE_GOALS_PROMPT.format(
                    n=len(batch), goals=goals_text
                ),
            },
        ]
        async with semaphore:
            raw = await llm.call_async(messages)

        try:
            return json.loads(raw)
        except Exception:
            logger.warning("Failed to parse canonicalize batch; using raw goals.")
            unique = list(dict.fromkeys(rg for _, rg, _ in batch))
            return {
                "canonical": unique,
                "descriptions": [""] * len(unique),
                "assignments": [unique.index(rg) for _, rg, _ in batch],
            }

    def _final_merge(
        self, unique_canonicals: list[str], assigned_names: list[str]
    ) -> dict[str, str]:
        """Merge near-duplicate canonical names via LLM. Returns old->new name map."""
        freq = Counter(assigned_names)
        # Most frequent name first so it survives as primary after merge
        sorted_canonicals = sorted(unique_canonicals, key=lambda n: -freq[n])

        llm = LLM(model=self.llm_model, provider=self.llm_provider)
        bullet_list = "\n".join(
            f"{i}. {name}" for i, name in enumerate(sorted_canonicals)
        )
        messages = [
            {"role": "system", "content": MERGE_SIMILAR_GOALS_SYSTEM},
            {
                "role": "user",
                "content": MERGE_SIMILAR_GOALS_PROMPT.format(
                    n=len(sorted_canonicals), goals=bullet_list
                ),
            },
        ]

        try:
            raw = llm.call(messages)
            groups: list[list[int]] = json.loads(raw).get("groups", [])
        except Exception:
            logger.warning("Final merge failed; keeping canonicals separate.")
            return {name: name for name in unique_canonicals}

        remap: dict[str, str] = {}
        for group in groups:
            if not group:
                continue
            primary = sorted_canonicals[group[0]]
            for idx in group:
                remap[sorted_canonicals[idx]] = primary
        return remap
