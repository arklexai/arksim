# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from arksim.scenario.goal_discovery.models import (
    ConversationInput,
    GoalCluster,
    GoalDiscoveryResult,
)
from arksim.scenario.goal_discovery.pipeline import GoalDiscoveryPipeline
from arksim.scenario.goal_discovery.preprocessing import (
    clean_text,
    contains_profanity,
    extract_first_turns,
    is_negative_emotion,
    sample_conversations,
)

__all__ = [
    "ConversationInput",
    "GoalCluster",
    "GoalDiscoveryResult",
    "GoalDiscoveryPipeline",
    "clean_text",
    "contains_profanity",
    "extract_first_turns",
    "is_negative_emotion",
    "sample_conversations",
]

# from_maa_record is a classmethod on ConversationInput, not a standalone export.
# Usage: ConversationInput.from_maa_record(record)
