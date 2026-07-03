# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import random
import re

from arksim.scenario.goal_discovery.models import ConversationInput


def extract_first_turns(
    conversations: list[ConversationInput],
    min_words: int = 3,
    reformulated_key: str | None = None,
) -> list[tuple[int, str]]:
    """Extract (original_index, text) from each conversation.

    When reformulated_key is set (e.g. "reformulated_question" for MAA data),
    the value at meta[reformulated_key] is used instead of the raw first user
    turn — it is already normalised and makes better embedding input.
    Falls back to the raw first user turn if the key is absent or too short.

    Conversations with no qualifying text are dropped.

    Args:
        conversations: Input conversations.
        min_words: Skip turns with fewer than this many words.
        reformulated_key: Optional meta key to prefer over the raw turn.

    Returns:
        List of (original_index, turn_text) tuples.
    """
    result: list[tuple[int, str]] = []
    for i, conv in enumerate(conversations):
        text: str | None = None

        if reformulated_key:
            reformulated = conv.meta.get(reformulated_key, "")
            if isinstance(reformulated, str):
                reformulated = reformulated.strip()
                if reformulated and len(reformulated.split()) >= min_words:
                    text = reformulated

        if text is None:
            text = conv.first_user_turn(min_words=min_words)

        if text:
            result.append((i, text))
    return result


# Common profanity — whole-word match only (avoids "assassin", "classic", etc.)
_PROFANITY_PATTERN = re.compile(
    r"\b("
    r"fuck|fucking|fucked|fucker|"
    r"shit|shitting|bullshit|"
    r"ass|asshole|"
    r"bitch|bitching|"
    r"cunt|"
    r"damn|damned|"
    r"bastard|"
    r"piss|pissed|"
    r"cock|dick|"
    r"crap"
    r")\b",
    re.IGNORECASE,
)


def contains_profanity(text: str) -> bool:
    """Return True if text contains at least one profane word."""
    return bool(_PROFANITY_PATTERN.search(text))


# Negative sentiment words — frustration, anger, or strong dissatisfaction.
_NEGATIVE_SENTIMENT_PATTERN = re.compile(
    r"\b("
    r"frustrated|frustrating|frustration|"
    r"angry|anger|furious|furiously|outraged|outrageous|infuriated|"
    r"terrible|terribly|horrible|horribly|awful|awfully|"
    r"ridiculous|unacceptable|disgraceful|appalling|"
    r"pathetic|useless|incompetent|"
    r"disappointed|disappointing|disappointment|"
    r"disgusted|disgusting|"
    r"upset|annoyed|irritated|irate|"
    r"hate|hated|"
    r"worst|"
    r"scam|fraud|fraudulent|"
    r"rude|rudely|rudeness"
    r")\b",
    re.IGNORECASE,
)


def is_negative_emotion(text: str) -> bool:
    """Return True if text expresses frustration, anger, or strong dissatisfaction."""
    return contains_profanity(text) or bool(_NEGATIVE_SENTIMENT_PATTERN.search(text))


# PII placeholder tokens injected by upstream redaction systems.
# These are stripped before embedding so they do not skew similarity.
_PII_PATTERN = re.compile(
    r"\[(?:REDACTED|PII|EMAIL|PHONE|NAME|ADDRESS|SSN|DOB|ZIP|ID)\]"
    r"|\*{2,}",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """Lightly normalise a user turn before embedding.

    Strips PII placeholder tokens (e.g. [REDACTED], ***), collapses
    whitespace, and strips leading/trailing space.
    Does not lowercase so that embeddings preserve capitalisation signals.
    """
    text = _PII_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sample_conversations(
    conversations: list[ConversationInput],
    n: int,
    seed: int = 42,
) -> list[ConversationInput]:
    """Return up to n conversations sampled without replacement."""
    if n >= len(conversations):
        return conversations
    rng = random.Random(seed)
    return rng.sample(conversations, n)
