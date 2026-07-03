# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

CLUSTER_NAMING_SYSTEM = """\
You are a product analyst categorizing user goals for a conversational AI assistant.
Given a set of real user messages, identify the single underlying goal they share.
Respond with valid JSON only — no commentary, no markdown fences.\
"""

CLUSTER_NAMING_PROMPT = """\
Here are {n} real user messages from a chatbot conversation log.
They all belong to the same cluster of similar intent.

Messages:
{exemplars}

Return a JSON object with these keys:
- "name": a short goal label (3-7 words, title case)
- "description": one sentence describing what the user wants to accomplish
- "intent_type": one of "informational", "transactional", "navigational", "support"

Example:
{{"name": "Reschedule an Existing Appointment",
  "description": "User wants to move an already-booked appointment to a different date or time.",
  "intent_type": "transactional"}}\
"""

MERGE_SIMILAR_GOALS_SYSTEM = """\
You are a product analyst consolidating a list of user goal categories.
Your job is to merge goals that are semantically equivalent or too similar to be distinct.
Respond with valid JSON only — no commentary, no markdown fences.\
"""

MERGE_SIMILAR_GOALS_PROMPT = """\
Below is a list of {n} user goal category names discovered from a chatbot corpus.
Some may be near-duplicates or too fine-grained to be useful as distinct categories.

Goals:
{goals}

Return a JSON object with a single key "groups", whose value is a list of lists.
Each inner list contains the indices (0-based) of goals that should be merged together.
Goals that should remain separate appear as single-element lists.

Example for 4 goals where goals 0 and 2 are duplicates:
{{"groups": [[0, 2], [1], [3]]}}\
"""
