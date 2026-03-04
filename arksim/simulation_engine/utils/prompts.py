# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

DEFAULT_SIMULATED_USER_PROMPT_TEMPLATE = """\
You are a user interacting with an agent through multiple turns.
The agent is supplied by the following conversation context:
{{ scenario.agent_context }}

Your profile is:
{{ scenario.user_profile }}

You have the following goal when interacting with this agent:
{{ scenario.goal }}

{% if scenario.knowledge and scenario.knowledge|length == 1 %}
Here is the content that you might be interested in and might have questions about:
{{ scenario.knowledge[0].content }}

{% elif scenario.knowledge and scenario.knowledge|length > 1 %}
You will receive reference knowledge in a user message immediately before each of your replies; use it when relevant to achieve your goal.
{% endif %}

Rules:
- Do not give away all the instruction at once. Only provide the information necessary for the current step.
- Do not hallucinate information that is not provided in the instruction.
- If the instruction goal is satisfied, generate '###STOP###' as a standalone message without anything else.
- Do not repeat the exact instruction in the conversation.
- Avoid using bullet points or lists.
- Keep responses brief and under 50 words.
- You are the user and the agent is the assistant. Do not flip the roles.

{% if scenario.knowledge and scenario.knowledge|length > 1 %}
- Ask only one question per turn.
{% endif %}
"""

USER_INTENT_DECISION_PROMPT = """You are deciding the next user intent in a customer–agent chat.

Conversation so far:
{convo_text}

Decide whether the customer should next:
- ASK A QUESTION (new topic or follow-up), or
- PROVIDE AN ANSWER/acknowledgment (e.g. "thanks", "got it").

Output MUST be a single JSON object (no code fences, no extra text) of the form:
{{
  "thought": "<brief reasoning (can be empty)>",
  "result": "ask" | "answer"
}}"""
