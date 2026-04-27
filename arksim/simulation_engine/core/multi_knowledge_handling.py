# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import random
from enum import Enum

from pydantic import BaseModel

from arksim.llms.chat import LLM
from arksim.llms.chat.base.usage import usage_label
from arksim.simulation_engine.utils.prompts import USER_INTENT_DECISION_PROMPT

logger = logging.getLogger(__name__)

# How much recent conversation to show the intent-decider LLM
MAX_TURNS_FOR_INTENT = 6  # last 3 exchanges (user+assistant)
MAX_CHARS_PER_TURN = 200  # truncate each message for privacy and brevity


def combine_knowledge(knowledge: list[str]) -> str:
    """Combine all knowledge items as a numbered list (COMBINE_ALL strategy)."""
    if not knowledge:
        return ""
    return "\n\n".join([f"Knowledge {i + 1}:\n{kc}" for i, kc in enumerate(knowledge)])


def pick_one_for_turn(
    knowledge: list[str],
    used_indices: set[int] | None = None,
    rng: random.Random | None = None,
) -> tuple[str, set[int]]:
    """Pick one knowledge item for this turn, avoiding already-used indices (Method 1 per-turn).

    For each turn a different knowledge is chosen when possible. When all have been used,
    the set of used indices is reset so we cycle through again without repeating until then.

    Args:
        knowledge: List of knowledge content strings.
        used_indices: Indices already used this conversation (modified in place and returned).
        rng: Optional RNG for reproducibility.

    Returns:
        (selected_content, updated_used_indices).
    """
    if not knowledge:
        return "", set()
    r = rng or random
    used = used_indices if used_indices is not None else set()
    # If all indices used, reset so we don't repeat until we've used all
    if len(used) >= len(knowledge):
        used = set()
    available = [i for i in range(len(knowledge)) if i not in used]
    idx = r.choice(available)
    used = used | {idx}
    return knowledge[idx].strip(), used


class UserIntentResult(str, Enum):
    ASK = "ask"
    ANSWER = "answer"


class UserIntent(BaseModel):
    """Structured intent prediction from LLM."""

    thought: str
    result: UserIntentResult


async def decide_ask_or_answer(llm: LLM, history: list[dict]) -> str:
    """Prior agentic step: decide if this turn is ask question or provide answer. Returns 'ask' or 'answer'."""
    # Format recent messages (skip system)
    recent = [msg for msg in history[1:] if msg.get("role") in ("user", "assistant")][
        -MAX_TURNS_FOR_INTENT:
    ]  # last 3 exchanges

    def _speaker_label(role: str) -> str:
        # In this simulator, 'user' is the customer and 'assistant' is the agent.
        # Make this explicit for the intent-deciding LLM.
        if role == "user":
            return "CUSTOMER"
        if role == "assistant":
            return "AGENT"
        return role.upper() if role else ""

    convo_text = "\n".join(
        f"{_speaker_label(m.get('role', ''))}: {m.get('content', '')[:MAX_CHARS_PER_TURN]}"
        for m in recent
    )
    prompt = USER_INTENT_DECISION_PROMPT.format(
        convo_text=convo_text or "(just started)"
    )
    try:
        with usage_label(component="multi_knowledge"):
            parsed = await llm.call_async(  # type: ignore[call-arg]
                [{"role": "user", "content": prompt}],
                schema=UserIntent,
            )

        intent: UserIntent = parsed

        # Debug-only, PII-safe logging (truncate content, no raw messages)
        logger.debug(
            "UserIntent decision: result=%s, thought_preview=%s, convo_chars=%d",
            intent.result.value,
            intent.thought[:100].replace("\n", " ") if intent.thought else "",
            len(convo_text),
        )
        return intent.result.value
    except Exception:
        logger.debug("UserIntent decision: default to ask", exc_info=True)
        return "ask"  # default to ask


# -------- Turn-level strategy helpers --------


async def prior_agentic_turn_knowledge(
    llm: LLM,
    history: list[dict],
    knowledge_content: str | list[str],
    state: dict,
) -> tuple[str, dict]:
    """Prior-agentic turn strategy.

    Each turn:
    - Ask the LLM if the user should ASK or ANSWER next (based on history).
    - If ANSWER: return no knowledge for this turn.
    - If ASK:
        * If multiple knowledge items: pick one with pick_one_for_turn, tracking used indices.
        * If single string: return it (stripped).
    """
    if not knowledge_content:
        return "", state

    decision = await decide_ask_or_answer(llm, history)
    if decision != UserIntentResult.ASK.value:
        return "", state

    # When asking, inject knowledge as user_message
    if isinstance(knowledge_content, list):
        used = state.get("used_indices_pick_first") or set()
        turn_knowledge, used = pick_one_for_turn(knowledge_content, used_indices=used)
        state["used_indices_pick_first"] = used
        return turn_knowledge, state
    if isinstance(knowledge_content, str):
        return knowledge_content.strip(), state
    return "", state


async def pick_one_turn_knowledge(
    llm: LLM,  # unused, kept for a uniform signature
    history: list[dict],  # unused
    knowledge_content: str | list[str],
    state: dict,
) -> tuple[str, dict]:
    """Pick-one turn strategy: always show one knowledge item this turn.

    - If multiple items: rotate via pick_one_for_turn, tracking used indices.
    - If single string: always return it (stripped).
    """
    del llm, history  # not needed for this strategy

    if not knowledge_content:
        return "", state

    if isinstance(knowledge_content, list) and knowledge_content:
        used = state.get("used_indices") or set()
        turn_knowledge, used = pick_one_for_turn(knowledge_content, used_indices=used)
        state["used_indices"] = used
        return turn_knowledge, state

    if isinstance(knowledge_content, str):
        return knowledge_content.strip(), state
    return "", state


async def combine_all_turn_knowledge(
    llm: LLM,  # unused
    history: list[dict],  # unused
    knowledge_content: str | list[str],
    state: dict,
) -> tuple[str, dict]:
    """Combine-all strategy: return the same combined knowledge every turn."""
    del llm, history  # not needed for this strategy

    if not knowledge_content:
        return "", state

    if isinstance(knowledge_content, list):
        return combine_knowledge(knowledge_content), state
    if isinstance(knowledge_content, str):
        return knowledge_content.strip(), state
    return "", state


# Default turn strategy used by the simulator.
# To switch methods, change this binding to one of:
# - prior_agentic_turn_knowledge
# - pick_one_turn_knowledge
# - combine_all_turn_knowledge
TURN_KNOWLEDGE_FN = prior_agentic_turn_knowledge
