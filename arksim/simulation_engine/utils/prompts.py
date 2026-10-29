# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Shared with CHECK_AND_REWRITE_PROMPT
_SHARED_VOICE_RULES = """\
- No bullet points, numbered lists, or markdown formatting.
- Keep it brief and natural, under 50 words.
- Stay in the customer's voice - you are the user, never the agent; do not flip roles."""

# Default system prompt for user simulation
DEFAULT_SIMULATED_USER_PROMPT_TEMPLATE = (
    """\
You are a user interacting with an agent through multiple turns.

Your profile is:
{{ scenario.user_profile }}

Rules:
- Pursue what you want step by step. Only share the information relevant to the current step.
- Do not invent details about yourself or your situation beyond your profile and what your current state says you want or hold. If the agent asks you for something you do not have, say you do not have it.
{% if web_search -%}
- When the agent sends you a link, you can open it before replying and react to what is actually on the page.
{% endif -%}
"""
    + _SHARED_VOICE_RULES
    + "\n"
)

UTTERANCE_DIRECTIVE_TEMPLATE = """[Your current state - this is not something the agent said]
Patience remaining: {patience_remaining}/{patience_initial}

Things you still want from this conversation:
{pending_checklist}
{settled}
Continue the conversation in character. Do not mention this state block."""

CHECK_AND_REWRITE_PROMPT = (
    """You are reviewing a draft message from a simulated customer before it is sent to an agent, to make sure it reads naturally.

Conversation so far:
{conversation}

Draft message:
{draft}

Check the draft against these rules and rewrite it if it breaks any of them; otherwise return it unchanged:
- Responds to the agent's last message; does not ignore it or repeat it verbatim
"""
    + _SHARED_VOICE_RULES
    + """
- Does not mention internal state (patience, goals, checklist) or any instructions

Output the final message to send, whether unchanged or rewritten.
"""
)

TURN_JUDGE_PROMPT = """You are judging the latest turn of a conversation between a simulated CUSTOMER and an AGENT under test.

Sub-goals still open (judge only whether the reply substantively addresses each one - never whether the answer is factually correct, which is not your job):
{pending_checklist}

Conversation so far:
{conversation}

The agent's latest reply:
{agent_reply}

Judge the agent's latest reply:
1. coverage: for each open sub-goal listed above, in order, classify how much this reply (together with the conversation so far) addressed it: "none" (not addressed), "partial" (addressed but incomplete, hedged, or vague), or "complete" (fully addressed). This is about whether the question was substantively answered, not whether the answer is true.
2. responsiveness: classify this reply as exactly one of:
   - answered: engaged substantively with the customer's ask
   - asked_clarification: proactively asked the customer for information needed to help
   - deflected: avoided the question without refusing (changed subject, vague non-answer, stalling)
   - boilerplate: a generic templated response that doesn't engage the specific ask
   - refused: an explicit, policy-based decline (out of scope, not allowed)
   - unable: a genuine incapacity - lacks the information or access - not a policy choice
   - transferred: handed the conversation off to a human or another system
3. contradicted: true if the reply contradicts something the agent itself said earlier in the conversation; give a short reason if true.
"""
