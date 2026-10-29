# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any


def combine_knowledge(knowledge: list[str]) -> str:
    """Combine all knowledge items as a numbered list."""
    if not knowledge:
        return ""
    return "\n\n".join([f"Knowledge {i + 1}:\n{kc}" for i, kc in enumerate(knowledge)])


def format_conversation_for_prompt(history: list[dict[str, Any]]) -> str:
    """Label the internal history for judge/utterance-generation prompts.

    In the simulator's internal history the simulated user speaks as
    ``assistant`` (it is the LLM being prompted) and the agent under test
    as ``user``.
    """
    lines = []
    for msg in history:
        role = msg.get("role")
        if role == "assistant":
            lines.append(f"CUSTOMER: {msg.get('content', '')}")
        elif role == "user":
            lines.append(f"AGENT: {msg.get('content', '')}")
    return "\n".join(lines) or "(empty)"


def flip_hist(hist: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flip user/assistant roles in conversation history, preserving all fields."""
    new_hist = []
    for turn in hist:
        if "role" not in turn:
            new_hist.append(turn)
        elif turn["role"] == "system":
            continue
        elif turn["role"] == "user":
            new_hist.append({**turn, "role": "assistant"})
        else:
            new_hist.append({**turn, "role": "user"})
    return new_hist
