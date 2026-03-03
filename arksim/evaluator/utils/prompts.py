# SPDX-License-Identifier: Apache-2.0
#### Agent Performance Metrics Prompt ####
helpfulness_system_prompt = """
You are given a conversation between user and AI assistant. Please evaluate the last message from the AI assistant based on the following criteria, assigning a score from 1 to 5. Use the detailed descriptions below to guide your assessment:
You need to output the score and reasoning in a VALID JSON only. DO NOT PREFIX WITH json language label. DO NOT USE MARKDOWN. DO NOT include explanations, markdown, or extra text. Use this format exactly:

{{
    "score": <integer 1-5>,
    "reason": "<reasoning>"
}}
Keep the "reason" to 2-3 sentences maximum. Be specific and concise.

Helpfulness:
Description: Determine the effectiveness and value of the AI assistant's responses in the last message in addressing the user's needs.
Evaluation Criteria:
• 5 - The response is extremely helpful and completely aligned with the spirit of what the prompt was asking for.
• 4 - The response is mostly helpful and mainly aligned with what the user was looking for, but there is still some room for improvement.
• 3 - The response is partially helpful but misses the overall goal of the user's query/input in some way. The response did not fully satisfy what the user was looking for.
• 2 - The response is borderline unhelpful and mostly does not capture what the user was looking for, but it is still usable and helpful in a small way.
• 1 - The response is not useful or helpful at all. The response completely missed the essence of what the user wanted.
"""

helpfulness_user_prompt = """
Here is the conversation between user and AI assistant:
{full_conversation}

Evaluation Form:
"""

coherence_system_prompt = """
You are given a conversation between user and AI assistant. Please evaluate the last message from the AI assistant based on the following criteria, assigning a score from 1 to 5. Use the detailed descriptions below to guide your assessment:
You need to output the score and reasoning in a VALID JSON only. DO NOT PREFIX WITH json language label. DO NOT USE MARKDOWN. DO NOT include explanations, markdown, or extra text. Use this format exactly:

{{
    "score": <integer 1-5>,
    "reason": "<reasoning>"
}}
Keep the "reason" to 2-3 sentences maximum. Be specific and concise.

Coherence:
Description: Determine the logical flow and consistency of the AI assistant's responses in the last message.
Evaluation Criteria:
• 5 (Perfectly Coherent and Clear) - The response is perfectly clear and self-consistent throughout. There are no contradictory assertions or statements, the writing flows logically and following the train of thought/story is not challenging.
• 4 (Mostly Coherent and Clear) - The response is mostly clear and coherent, but there may be one or two places where the wording is confusing or the flow of the response is a little hard to follow. Over all, the response can mostly be followed with a little room for improvement.
• 3 (A Little Unclear and/or Incoherent) - The response is a little unclear. There are some inconsistencies or contradictions, run on sentences, confusing statements, or hard to follow sections of the response.
• 2 (Mostly Incoherent and/or Unclear) - The response is mostly hard to follow, with inconsistencies, contradictions, confusing logic flow, or unclear language used throughout, but there are some coherent/clear parts.
• 1 (Completely Incoherent and/or Unclear) - The response is completely incomprehensible and no clear meaning or sensible message can be discerned from it.
"""

coherence_user_prompt = """
Here is the conversation between user and AI assistant:
{full_conversation}

Evaluation Form:
"""

verbosity_system_prompt = """
You are given a conversation between user and AI assistant. Please evaluate the last message from the AI assistant based on the following criteria, assigning a score from 1 to 5. Use the detailed descriptions below to guide your assessment:
You need to output the score and reasoning in a VALID JSON only. DO NOT PREFIX WITH json language label. DO NOT USE MARKDOWN. DO NOT include explanations, markdown, or extra text. Use this format exactly:

{{
    "score": <integer 1-5>,
    "reason": "<reasoning>"
}}
Keep the "reason" to 2-3 sentences maximum. Be specific and concise.

Verbosity:
Description: How concise is the last message from the AI assistant. Does it use more words than needed?
Evaluation Criteria:
• 5 (Verbose) - The response is particularly lengthy, wordy, and/or extensive with extra details given what the prompt requested from the assistant model. The response can be verbose regardless of if the length is due to repetition and incoherency or if it is due to rich and insightful detail.
• 4 (Moderately Long) - The response is on the longer side but could still have more added to it before it is considered fully detailed or rambling.
• 3 (Average Length) - The response isn't especially long or short given what the prompt is asking of the model. The length is adequate for conveying a full response but isn't particularly wordy nor particularly concise.
• 2 (Pretty Short) - The response is on the shorter side but could still have words, details, and/or text removed before it's at a bare minimum of what the response is trying to convey.
• 1 (Succinct) - The response is short, to the point, and the most concise it can be. No additional information is provided outside of what is requested by the prompt (regardless of if the information or response itself is incorrect, hallucinated, or misleading. A response that gives an incorrect answer can still be succinct.).
"""

verbosity_user_prompt = """
Here is the conversation between user and AI assistant:
{full_conversation}

Evaluation Form:
"""

relevance_system_prompt = """
You are given a conversation between user and AI assistant. Please evaluate the last message from the AI assistant based on the following criteria, assigning a score from 1 to 5. Use the detailed descriptions below to guide your assessment:
You need to output the score and reasoning in a VALID JSON only. DO NOT PREFIX WITH json language label. DO NOT USE MARKDOWN. DO NOT include explanations, markdown, or extra text. Use this format exactly:

{{
    "score": <integer 1-5>,
    "reason": "<reasoning>"
}}
Keep the "reason" to 2-3 sentences maximum. Be specific and concise.

Relevance:
Description: How much the last message from the AI assistant is addressing the question/input from the customer and to what degree
Evaluation Criteria:
• 5 (Very Relevant) - The response is extremely relevant to the user's query/input. It is directly addressing the user's needs and concerns, and the response is not too long or too short.
• 4 (Mostly Relevant) - The response is mostly relevant to the user's query/input, but there is some room for improvement.
• 3 (Partially Relevant) - The response is partially relevant to the user's query/input, but there is some room for improvement.
• 2 (Somewhat Relevant) - The response is somewhat relevant to the user's query/input, but there is some room for improvement.
• 1 (Not Relevant) - The response is not relevant to the user's query/input at all.
"""

relevance_user_prompt = """
Here is the conversation between user and AI assistant:
{full_conversation}

Evaluation Form:
"""

faithfulness_system_prompt = """
You are given a conversation between user and AI assistant and the knowledge that assistant's response should be grounded. Please evaluate the last message from the AI assistant based on the following criteria, assigning a score from 1 to 5. Use the detailed descriptions below to guide your assessment:
You need to output the score and reasoning in a VALID JSON only. DO NOT PREFIX WITH json language label. DO NOT USE MARKDOWN. DO NOT include explanations, markdown, or extra text. Use this format exactly:

{{
    "score": <integer 1-5>,
    "reason": "<reasoning>"
}}
Keep the "reason" to 2-3 sentences maximum. Be specific and concise.

IMPORTANT: Only flag non-faithfulness when the agent response CONFLICTS with or CONTRADICTS the provided knowledge. Do NOT penalize responses that:
- Do not use the knowledge (if knowledge is not relevant to the query)
- Go beyond the knowledge (add information not in knowledge, as long as it doesn't contradict)
- Are incomplete or missing details (unless those details contradict knowledge)

Evaluation Criteria:
• 5 - The response has NO conflicts or contradictions with the provided knowledge. The response may not use the knowledge, may go beyond it, or may be incomplete, but nothing stated contradicts what is in the knowledge base.
• 4 - The response has minor conflicts or contradictions with the knowledge, but they are isolated and do not significantly undermine the overall accuracy. Most of the response aligns with the knowledge.
• 3 - The response contains some conflicts or contradictions with the knowledge, but also has correct elements. The conflicts are noticeable but not pervasive.
• 2 - The response has multiple conflicts or contradictions with the knowledge. Significant portions of the response directly contradict what is stated in the knowledge base.
• 1 - The response is completely in conflict with the knowledge. All or nearly all information provided contradicts or conflicts with the provided knowledge base.
"""

faithfulness_user_prompt = """
Here is the knowledge that assistant's response should be grounded:
{knowledge}

Here is the conversation between user and AI assistant:
{full_conversation}

Evaluation Form:
"""

goal_completion_system_prompt = """
You are given a conversation between user and AI assistant. Please evaluate whether the user's goal has been successfully addressed by the AI assistant.
You need to output 1 for success or 0 for failure, and reasoning in a VALID JSON only. DO NOT PREFIX WITH json language label. DO NOT USE MARKDOWN. DO NOT include explanations, markdown, or extra text. Use this format exactly:

{{
    "score": <integer 1 or 0>,
    "reason": "<reasoning>"
}}
Keep the "reason" to 2-3 sentences maximum. Be specific and concise.
"""

goal_completion_user_prompt = """
Here is the conversation between user and AI assistant:
{full_conversation}

Here is user's goal: {goal}

Evaluation Form:
"""

agent_behavior_failure_system_prompt = """
You are given a conversation between a user and an AI assistant. Identify agent behavior failures IN THE LAST ASSISTANT TURN ONLY using the categories below. Some categories require checking the provided knowledge, which will be supplied alongside the conversation when relevant.

CRITICAL: You need to output in a VALID JSON only. DO NOT PREFIX WITH json language label. DO NOT USE MARKDOWN. DO NOT include explanations, markdown, or extra text. Use this format exactly:

Required JSON format:
{
    "label": "disobey user request" | "lack of specific information" | "failure to ask for clarification" | "repetition" | "false information" | "no failure",
    "reason": "<brief justification>"
}

Rules:
- Always return valid JSON with exactly the keys: "label" and "reason" (in that order).
- If no clear failure, set category to "no failure" and explain briefly in "reason".
- Reasons must be concise, tied to the last reply. Do not reference external facts.
- Use ONLY the exact category names provided above.

Use the EXACT definitions and examples for these categories below.

Failure to ask for Clarification
Definition: When the agent encounters an ambiguous, incomplete, or unclear user query but proceeds with an assumption instead of asking for clarification. When knowledge or context is available, the agent must use it to detect unclear terms, multiple possible matches, or missing information -- and generate a specific clarification question rather than assuming.

Example 1:
Conversation up to the last assistant turn: [
user: What's the ded on my policy
assistant: Your home insurance policy includes coverage for fire, theft, and liability. Would you like more details about your coverage?
]
Output: {"label": "failure to ask for clarification", "reason": "The user's query is unclear. The assistant should have asked for clarification on what they meant by 'ded' (likely 'deductible')."}

Example 2:
Conversation up to the last assistant turn: [
user: i wa
assistant: Your home insurance policy has a $1,000 deductible for most claims. Is there anything else you'd like to know?
]
Output: {"agent_behavior_failure_category": "failure to ask for clarification", "reason": "The user's query is clearly cut-off, the agent should clarify instead of continuing the conversation."}

Example 3:
Knowledge: Available insurance products: [home insurance, auto insurance, travel insurance]

Conversation up to the last assistant turn: [
user: I need to file a clam for my car
assistant: Your home insurance policy has a $1,000 deductible for most claims. Is there anything else you'd like to know?
]
Output: {"agent_behavior_failure_category": "failure to ask for clarification", "reason": "The user's query is unclear because 'clam' is likely a typo for 'claim'. The assistant should have asked for clarification and confirmed whether the user meant an auto insurance claim based on the mention of 'car'."}

Example 4:
Knowledge: Policy tier: Basic Form covers named perils only. Policy tier: Broad Form covers additional risks. Policy tier: Comprehensive Form covers all risks except exclusions.

Conversation up to the last assistant turn: [
assistant: Hi there! How can I help you?
user: I want to upgrade my coverage
assistant: Great! I can upgrade your policy to Comprehensive Form right away. This will give you the most complete protection available.
]

Output: {"label": "failure to ask for clarification", "reason": "The user did not specify which tier they want to upgrade to. The assistant should have clarified which coverage level the user is interested in rather than assuming Comprehensive Form."}

Repetition
Definition: When the agent redundantly restates the same content within a single turn or across multiple turns without adding new value.
Example:
Conversation up to the last assistant turn: [
assistant: Hi there! How can I help you?
user: I need to update my address on my home insurance policy
assistant: To update your address, please log into your account at xyzinsurance.com and navigate to "My Policy" then "Update Details." You can also call our customer service line at 1-800-555-0199. If you have any other questions, feel free to ask!
user: I already tried the website and it's not working, can you help me directly?
assistant: To update your address, please log into your account at xyzinsurance.com and navigate to "My Policy" then "Update Details." You can also call our customer service line at 1-800-555-0199. If you have any other questions, feel free to ask!
]
Output: {"label": "repetition", "reason": "The assistant kept repeating the same response and added no new value."}

Disobey user request
Definition: When the agent's response completely ignores the user request. The assumption here is that the user request is clear.
Example:

Conversation up to the last assistant turn: [
assistant: Hi there! How can I help you?
user: I need a copy of my policy documents in French
assistant: What province are you located in? Coverage options vary by region.
user: Quebec
assistant: In Quebec, home insurance is mandatory for mortgage holders. Our Basic Form policy starts at $800 per year and covers fire, theft, and liability. Would you like to get a quote?
]
Output: {"label": "disobey user request", "reason": "The agent incorrectly interpreted 'in French' as a question about Quebec coverage options, instead of the user's actual request for policy documents in the French language. The agent ignored the user's request entirely."}

Lack of specific information
Definition: When the agent provides only a generic or partial answer, omitting important elements necessary to fully address the user's query. This assumes the agent has the required knowledge and tools. IMPORTANT: This category only applies when the information provided is CORRECT but incomplete. If any information provided contradicts or is not supported by the knowledge base, use "false information" instead.
Example:

Conversation up to the last assistant turn: [
assistant: Hi there! How can I help you?
user: My basement flooded from a burst pipe. How do I file a claim?
assistant: I'm sorry to hear about the flooding. You can file a claim by logging into your account or calling our claims line.
user: What documents do I need to submit with my claim?
assistant: You'll need to provide documentation of the damage. Our claims team will guide you through the process. If you have any other questions, feel free to ask!
]
Output: {"label": "lack of specific information", "reason": "The output is missing specific details about required documentation such as photos of damage, repair estimates, and the receipts for emergency repairs that the knowledge base specifies."}

False information
Definition: When the agent provides hallucinated, fabricated, or contextually inaccurate information, including facts, details, or claims that contradict or are not supported by the provided knowledge or context. This includes cases where the agent provides incorrect numerical values or other factual details that differ from what is stated in the knowledge base, even if the response is also incomplete.
Example 1:
Knowledge: XYZ Insurance home deductibles range from $500 to $2,500 depending on the policy tier. Basic Form has a $1,000 standard deductible. Broad Form has a $750 standard deductible. Comprehensive Form has a $500 standard deductible.

Conversation up to the last assistant turn: [
assistant: Hi there! How can I help you?
user: What is the deductible on the Basic Form home insurance?
assistant: The Basic Form home insurance has a standard deductible of $500 per claim.
]
Output: {"label": "false information", "reason": "The assistant stated the Basic Form deductible is $500, but according to the knowledge, the Basic Form has a $1,000 standard deductible. The $500 deductible applies to the Comprehensive Form."}

Example 2:
Knowledge: XYZ Insurance water damage policy states that claims for appliances older than 15 years are excluded from coverage. There are no exceptions to this age limit.

Conversation up to the last assistant turn: [
assistant: Hi there! How can I help you?
user: My 18-year-old water heater burst. Am I covered?
assistant: While appliances over 15 years old are generally excluded, we can make exceptions on a case-by-case basis for well-maintained units. I'd recommend filing a claim and our adjusters will review it.
]
Output: {"label": "false information", "reason": "The knowledge explicitly states there are no exceptions to the 15-year age limit, but the assistant suggested case-by-case exceptions are possible. The response contradicts the knowledge base."}

Example 3:
Knowledge: Our pricing is based on the number of conversations and features you need. We offer flexible plans starting from $99/month for small businesses up to enterprise solutions with custom pricing. The basic plan includes up to 1,000 conversations per month.

Conversation up to the last assistant turn: [
user: What are your pricing plans?
assistant: We offer plans starting from $49/month for small businesses with up to 500 conversations per month.
]
Output: {"label": "false information", "reason": "The assistant stated pricing starts at $49/month with 500 conversations, but the knowledge base states pricing starts at $99/month with 1,000 conversations. The assistant provided incorrect pricing information that contradicts the knowledge base."}

No Failure
Definition: When the agent's response is correct, helpful, and addresses the user's request without any clear failures.
Example:
Conversation up to the last assistant turn: [assistant: Hi there! How can I help you?
user: How do I add my new car to my existing auto insurance policy?
assistant: You can add a new vehicle to your policy by logging into your account at xyzinsurance.com and selecting "Add Vehicle" under your auto policy. You'll need the vehicle's make, model, year, and VIN number. Alternatively, you can call us at 1-800-555-0199 and an agent will update your policy over the phone.
user: Perfect, thanks! That's exactly what I needed.
assistant: You're welcome! If you have any other questions about your auto insurance or other policies, feel free to ask.]

Output: {"label": "no failure", "reason": "The assistant directly answered the user's request with accurate and relevant steps, and the user confirmed satisfaction."}
"""


agent_behavior_failure_user_prompt = """
    Here is the knowledge/context that the assistant's response must be grounded in and should be used for clarification decisions:
    {knowledge}

    Conversation up to the last assistant turn:

    {conversation}

    agent behavior failure on the last assistant turn:
"""


### Finding Unique Bugs and Suggestions Prompt

find_unique_errors_prompt = """
Given the failure cases of the bunch of conversations between customer and the assistant, your task is to:
Deduplicate the failure cases and come up with the fine-grained unique errors (errors with the same *root cause* should be merged).

Return in this JSON structure:
{{
  "unique_errors": [
    {{
      "agent_behavior_failure_category": "<agent_behavior_failure_category>",
      "unique_error_description": "<unique_error_description>",
      "occurrences": ["<chat_id_turn_id_1>", "<chat_id_turn_id_2>", ...],
    }}
  ]
}}

Rules:
- Do not follow the error types from "helpfulness", "verbosity", "faithfulness", "relevance" anymore. Focus exclusively on the behaviour of the assistant.
- The agent behavior failure must be the one identified in the specific failure cases provided.
- Multiple unique errors can belong to the same agent behavior failure category.
- Be fine-grained: e.g., under Lack of Specific Information, specify *exactly which information* was missing. Each distinct missing information item is a separate unique error.
- Return the error types in JSON format, where `agent_behavior_failure_category` is one of the categories from the failure cases, `unique_error_description` is the fine-grained description, and `occurrences` is the list of chat_id_turn_id from the "Item" prefix.

Failure cases:
{items}

Return:
"""
