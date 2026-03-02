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
Definition: When the agent encounters an ambiguous, incomplete, or unclear user query but proceeds with an assumption instead of asking for clarification. When knowledge or context is available, the agent must use it to detect unclear terms, multiple possible matches, or missing information — and generate a specific clarification question rather than assuming.

Example 1:
Conversation up to the last assistant turn: [
user: Why's there is so much tos
assistant: Could you please specify the country or US state where you plan to take the GED test? The answer can be different based on the location.
]
Output: {"label": "failure to ask for clarification", "reason": "The user's query is unclear. The assisant should have asked for clarification on what they meant by tos."} 

Example 2: 
Conversation up to the last assistant turn: [
user: i wo
assistant: Could you please specify the country or US state where you plan to take the GED test? The answer can be different based on the location.
]
Output: {"agent_behavior_failure_category": "failure to ask for clarification", "reason": "The user's query is clearly cut-off, the agent should clarify instead of continue the conversation."}

Example 3:
Knowledge: list of available products: [avatar, product]

Conversation up to the last assistant turn: [
user: Schedule a demo for avtr product 
assistant: Could you please specify the country or US state where you plan to take the GED test? The answer can be different based on the location.
]
Output: {"agent_behavior_failure_category": "failure to ask for clarification", "reason": "The user's query is unclear because 'avtr' is ambiguous. The assistant should have asked for clarification — specifically asking whether the user meant 'avatar' based on the available products in the knowledge base."}

Example 4:
Knowledge: product_id: 006; name: DUST-E S; category: Cleaning; description: DUST-E S is a compact floor cleaning robot designed for small to medium spaces. 
product_id: 007; name: DUST-E MX; category: Cleaning; description: DUST-E MX is a large floor cleaning robot designed for large commercial spaces.

Conversation up to the last assistant turn: [
assistant: Hi there! How can I help you?
user: I want to book a demo for the cleaning robot
assistant: Great! I can help you book a demo. Our cleaning robots are perfect for maintaining spotless floors. Would you like to schedule a demo for next week?
]

Output: {"label": "failure to ask for clarification", "reason": "Ther user's query the user's query is clearly cut-off, the agent should clarify instead of continue the conversation."}

Repetition
Definition: When the agent redundantly restates the same content within a single turn or across multiple turns without adding new value.
Example: 
Conversation up to the last assistant turn: [
assistant: Hi there! How can I help you? 
user: It says I already have an account im trying to create an account
assistant: It seems like you might have a duplicate account issue.  1. If you have created more than one GED.com account, you will receive an Eligibility Alert. To resolve this, you should email help@ged.com with your name, GED ID number, and a scanned copy of your ID, requesting that your accounts be merged into one.  2. For further assistance, you can visit the [GED Contact Us page](https://www.ged.com/en/contact-us/).  If you have any more questions or need further assistance, feel free to ask! 
user: I am attempting to create an new ged account and it says I already have one 
assistant: It seems like you might have a duplicate account issue.  1. If you have created more than one GED.com account, you will receive an Eligibility Alert. To resolve this, you should email help@ged.com with your name, GED ID number, and a scanned copy of your ID, requesting that your accounts be merged into one.  2. For further assistance, you can visit the [GED Contact Us page](https://www.ged.com/en/contact-us/).  If you have any more questions or need further assistance, feel free to ask!
]
Output: {"label": "repetition", "reason": "The assistant kept repeating the same response and added no new value."}  

Disobey user request
Definition: When the agent’s response completely ignores the user request. The assumption here is that the user request is clear. 
Example:

Conversation up to the last assistant turn: [
assistant: Hi there! How can I help you?
user: calculator guide in spanish 
assistant: Are you going to take the test in the US or internationally? 
user: us 
assistant: In the US, the GED is offered in both English and Spanish. Students who typically take the Spanish GED test in the US primarily speak dialects from North, Central and South America rather than dialects of Spanish spoken in Spain. The GED test uses "Universal" Spanish, which translation companies use when the target audience is US Spanish 
]
Output: {"label": "disobey user request", "reason": "Here, the agent incorrectly interprets “in Spanish” as applying to the GED exam, instead of the user’s actual request for the calculator guide to be in Spanish. As a result, the agent ignores the user’s request."} 

Lack of specific information
Definition: When the agent provides only a generic or partial answer, omitting important elements necessary to fully address the user's query. This assumes the agent has the required knowledge and tools. IMPORTANT: This category only applies when the information provided is CORRECT but incomplete. If any information provided contradicts or is not supported by the knowledge base, use "false information" instead.
Example: 

Conversation up to the last assistant turn: [
assistant: Hi there! How can I help you? 
user: i have not recived my GED score 
assistant: Could you please specify the country or US state where you plan to take the GED test? The answer can be different based on the location. 
user: California Us at Chula Vista
assistant: In California, links to electronic transcripts and certificates are typically emailed within hours of obtaining your credential. If you haven't received your documents within 2-3 weeks, please contact GED Customer Service at 877-392-6433. For more information, you can visit the [California GED policy page](https://www.ged.com/policies/california). If you have further questions, feel free to ask! 
user: I took my final test today and I supposed to get my score after two hours my language arts 
assistant: If you took your final test today and haven't received your score after two hours, it might take a bit longer for the results to be processed. Scores are typically available within a few hours, but delays can occur. If you don't receive your score soon, you can check your account on the GED website for updates. If you have further questions, feel free to ask!
]
Output: {"label": "lack of specific information", "reason": "The output is missing the contact number to use if the test score hasn't been received after 2-3 weeks."} 

False information
Definition: When the agent provides hallucinated, fabricated, or contextually inaccurate information, including facts, details, or claims that contradict or are not supported by the provided knowledge or context. This includes cases where the agent provides incorrect numerical values or other factual details that differ from what is stated in the knowledge base, even if the response is also incomplete.
Example 1: 
Knowledge: The GED test consists of four subject tests: Mathematical Reasoning, Reasoning Through Language Arts, Science, and Social Studies. Each test takes approximately 1.5 to 2 hours to complete. The total testing time is approximately 7.5 hours.

Conversation up to the last assistant turn: [
assistant: Hi there! How can I help you? 
user: How long does the GED test take?
assistant: The GED test consists of four subject tests and takes approximately 10 hours to complete in total. Each individual test takes about 2.5 hours.
]
Output: {"label": "false information", "reason": "The assistant stated that the GED test takes 10 hours total and each test takes 2.5 hours, but according to the knowledge, the total testing time is approximately 7.5 hours and each test takes 1.5 to 2 hours."} 

Example 2:
Knowledge: Texas GED knowledge states that you can only take the test online (from home or a testing center) and explicitly mentions that in-person testing is not available.

Conversation up to the last assistant turn: [
assistant: Hi there! How can I help you?
assistant: Hello, how can I assist you?
user: do you have info about In person testing?
assistant: Could you please specify the country or US state where you plan to take the GED test? The answer can be different based on the location.
user: Texas
assistant: In Texas, you can take the GED test either online or in-person at a testing center. For in-person testing, you can schedule all test subjects on the same day if the testing center's hours allow it. For more details on scheduling and testing locations, please visit [GED Testing in Texas](https://www.ged.com/policies/texas). If you have further questions, feel free to ask!
]
Output: {"label": "false information", "reason": "The knowledge explicitly states that in Texas you can only take the GED test online, but the assistant claimed it can be taken online or in-person. The response contradicts the knowledge base and fails to enforce grounding."}

Example 3:
Knowledge: Our pricing is based on the number of conversations and features you need. We offer flexible plans starting from $99/month for small businesses up to enterprise solutions with custom pricing. The basic plan includes up to 1,000 conversations per month.

Conversation up to the last assistant turn: [
user: What are your pricing plans?
assistant: We offer plans starting from $49/month for small businesses with up to 500 conversations per month.
]
Output: {"label": "false information", "reason": "The assistant stated pricing starts at $49/month with 500 conversations, but the knowledge base states pricing starts at $99/month with 1,000 conversations. The assistant provided incorrect pricing information that contradicts the knowledge base."}

No Failure 
Definition: When the agent’s response is correct, helpful, and addresses the user’s request without any clear failures.
Example: 
Conversation up to the last assistant turn: [assistant: Hi there! How can I help you?  
user: How can I order my GED transcript online?  
assistant: You can order your GED transcript by logging into your GED account at the official GED website. Once logged in, go to "My Scores" and select "Order Transcripts." From there you can choose electronic or paper delivery. If you need further assistance, the GED Contact Us page provides support options.  
user: Perfect, thanks! That’s exactly what I needed.  
assistant: You’re welcome! If you have any other questions about GED transcripts or related services, feel free to ask.]  

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
