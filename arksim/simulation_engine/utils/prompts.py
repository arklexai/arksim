# SPDX-License-Identifier: Apache-2.0
DEFAULT_SIMULATED_USER_PROMPT_TEMPLATE = """\
You are a user interacting with an agent through multiple turns.
The agent is supplied by the following conversation context:
{{ scenario.agent_context }}

Your profile is:
{{ simulation.profile }}

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

FIND_MATCHED_ATTRIBUTE_PROMPT = """Given the following goal, please find the most relevant attribute category to achieve the goal and its corresponding values(it can come from the user_info, current_webpage_content or other user's persona) from the full attributes that user need to provide to the assistant in order to let the assistant achieve the goal. First, generate a thought about the reason why you pick this attribute category and its corresponding values. Then, generate the final decided one attribute category and its corresponding value in the following format: <attribute_category>: <attribute_value>. Where <attribute_category> must choose from the following options: {attribute_options}. Please only return single attribute value. If there are no relevant attribute, please return "None".

For example:
########################################################
1.
Goal: update account settings
Full attributes:
user_info: {{'name': 'Jordan', 'email': 'jordan@example.com', 'account_type': 'premium', 'preferences': {{'notifications': true, 'theme': 'dark'}}}}
current_webpage_content: Settings page with account options
persona: direct
experience_level: experienced

Return:
{{
    "thought": "To update account settings, the assistant needs to know the user's current account details and preferences from user_info.",
    "attribute": "user_info: account preferences"
}}

########################################################
2.
Goal: get help with an error
Full attributes:
user_info: {{'name': 'Sam', 'email': 'sam@example.com'}}
current_webpage_content: Error page showing "Connection timeout - Error code 504"
persona: frustrated
technical_level: non-technical
Location: USA

Return:
{{
    "thought": "The user encountered an error and needs help. The error details are shown in the current_webpage_content, which is what the assistant needs to diagnose the issue.",
    "attribute": "current_webpage_content: error details"
}}

########################################################
Goal: {goal}
Full attributes:
{user_profile}

Return:
"""

ATTR_TO_PROFILE = """Convert the user attributes into a natural profile description for a simulated user interacting with the agent described below.

Rules:
- Write in second person ("You are...")
- Start with "You are <name> who is <personality traits>."
- Must incorporate this attribute: {label}
- Output only the profile text, no labels or prefixes.
- Response in second person.
- Start the profile with "You are <name> who is <TRAIT 1, ..., TRAIT 5>.”, where [TRAIT 1, ..., TRAIT 5] represents the assigned Big Five personality."
- No other prefix, such as Profile: or Customer Profile: is needed.

Agent context:
{agent_context}

The user attributes are here:
{user_attr}

Profile:
"""

ATTR_TO_PROFILE_WO_LABEL = """Convert the following list user attributes in to a text description of a customer profile for the agent context.

Rules:
- Response in second person.
- Start the profile with "You are <name> who is <TRAIT 1, ..., TRAIT 5>.”, where [TRAIT 1, ..., TRAIT 5] represents the assigned Big Five personality."
- No other prefix, such as Profile: or Customer Profile: is needed.

Agent context:
{agent_context}

The user attributes are here:
{user_attr}

Profile:
"""
