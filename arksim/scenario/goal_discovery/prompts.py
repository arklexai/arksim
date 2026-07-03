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

EXTRACTION_SYSTEM = """\
You extract structured facts from customer-service conversations.

For each fact return a JSON object with exactly these keys:
  "attribute"       - the attribute name (from the list provided)
  "value"           - the extracted value, concise and normalised
  "confidence"      - float 0.0-1.0, how certain you are
  "provenance_span" - the exact substring from the conversation that supports this fact

Return a JSON array of fact objects.
Omit any attribute you cannot find evidence for.
Return only the JSON array — no commentary, no markdown fences.\
"""

EXTRACTION_PROMPT = """\
Conversation:
{conversation}

Extract facts for these attributes:
{attributes}

Return a JSON array of fact objects.\
"""

CANONICALIZE_GOALS_SYSTEM = """\
You are a product analyst normalizing raw user goal descriptions into a clean taxonomy.
Respond with valid JSON only — no commentary, no markdown fences.\
"""

CANONICALIZE_GOALS_PROMPT = """\
Below are {n} raw goal descriptions extracted from customer conversations.
Many are duplicates or near-duplicates phrased differently.

Goals:
{goals}

Deduplicate and normalize them into a canonical taxonomy.
Return a JSON object with these keys:
- "canonical": list of distinct canonical goal names (3-7 words, title case)
- "descriptions": list of one-sentence descriptions, one per canonical goal (same order)
- "assignments": list of {n} integers (0-based), one per input line,
                 each being the index of its canonical goal in "canonical"

Example for 5 inputs that reduce to 2 canonical goals:
{{"canonical": ["Track Order Status", "Cancel an Order"],
  "descriptions": ["User wants to check the current status or location of their order.",
                   "User wants to cancel a purchase they have already placed."],
  "assignments": [0, 0, 1, 0, 1]}}\
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
