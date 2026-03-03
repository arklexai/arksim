# SPDX-License-Identifier: Apache-2.0
"""Central registry of all evaluation and simulation prompts.

Provides a single lookup for every prompt used in the arksim
pipeline, organized by human-readable category.
"""

from __future__ import annotations

from dataclasses import dataclass

from arksim.evaluator.utils.prompts import (
    agent_behavior_failure_system_prompt,
    agent_behavior_failure_user_prompt,
    coherence_system_prompt,
    coherence_user_prompt,
    faithfulness_system_prompt,
    faithfulness_user_prompt,
    find_unique_errors_prompt,
    goal_completion_system_prompt,
    goal_completion_user_prompt,
    helpfulness_system_prompt,
    helpfulness_user_prompt,
    relevance_system_prompt,
    relevance_user_prompt,
    verbosity_system_prompt,
    verbosity_user_prompt,
)
from arksim.simulation_engine.utils.prompts import (
    DEFAULT_SIMULATED_USER_PROMPT_TEMPLATE,
)


@dataclass(frozen=True)
class PromptEntry:
    """A single prompt with its name and text."""

    name: str
    text: str


@dataclass(frozen=True)
class PromptCategory:
    """A named group of related prompts."""

    category: str
    description: str
    prompts: tuple[PromptEntry, ...]


PROMPT_REGISTRY: tuple[PromptCategory, ...] = (
    PromptCategory(
        category="helpfulness",
        description="Evaluates how effectively the assistant addresses the user's needs",
        prompts=(
            PromptEntry(name="system_prompt", text=helpfulness_system_prompt),
            PromptEntry(name="user_prompt", text=helpfulness_user_prompt),
        ),
    ),
    PromptCategory(
        category="coherence",
        description="Evaluates logical flow and consistency of responses",
        prompts=(
            PromptEntry(name="system_prompt", text=coherence_system_prompt),
            PromptEntry(name="user_prompt", text=coherence_user_prompt),
        ),
    ),
    PromptCategory(
        category="verbosity",
        description="Evaluates response length appropriateness",
        prompts=(
            PromptEntry(name="system_prompt", text=verbosity_system_prompt),
            PromptEntry(name="user_prompt", text=verbosity_user_prompt),
        ),
    ),
    PromptCategory(
        category="relevance",
        description="Evaluates how on-topic the response is",
        prompts=(
            PromptEntry(name="system_prompt", text=relevance_system_prompt),
            PromptEntry(name="user_prompt", text=relevance_user_prompt),
        ),
    ),
    PromptCategory(
        category="faithfulness",
        description="Evaluates accuracy relative to provided knowledge",
        prompts=(
            PromptEntry(name="system_prompt", text=faithfulness_system_prompt),
            PromptEntry(name="user_prompt", text=faithfulness_user_prompt),
        ),
    ),
    PromptCategory(
        category="goal_completion",
        description="Evaluates whether the user's goal was achieved",
        prompts=(
            PromptEntry(name="system_prompt", text=goal_completion_system_prompt),
            PromptEntry(name="user_prompt", text=goal_completion_user_prompt),
        ),
    ),
    PromptCategory(
        category="agent_behavior_failure",
        description="Detects agent behavior failures per turn",
        prompts=(
            PromptEntry(
                name="system_prompt",
                text=agent_behavior_failure_system_prompt,
            ),
            PromptEntry(
                name="user_prompt",
                text=agent_behavior_failure_user_prompt,
            ),
        ),
    ),
    PromptCategory(
        category="unique_error_detection",
        description="Deduplicates and identifies unique errors across conversations",
        prompts=(PromptEntry(name="prompt", text=find_unique_errors_prompt),),
    ),
    PromptCategory(
        category="simulated_user",
        description="System prompt template for the simulated user during conversations",
        prompts=(
            PromptEntry(
                name="default_template",
                text=DEFAULT_SIMULATED_USER_PROMPT_TEMPLATE,
            ),
        ),
    ),
)


def get_categories() -> list[str]:
    """Return sorted list of all category names."""
    return sorted(cat.category for cat in PROMPT_REGISTRY)


def get_prompts_by_category(
    category: str | None = None,
) -> list[PromptCategory]:
    """Return prompt categories, optionally filtered.

    Args:
        category: If provided, return only the matching
            category (case-insensitive). None returns all.

    Returns:
        List of matching PromptCategory objects.
    """
    if category is None:
        return list(PROMPT_REGISTRY)
    target = category.lower()
    return [c for c in PROMPT_REGISTRY if c.category == target]
