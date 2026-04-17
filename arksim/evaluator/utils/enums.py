# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from enum import Enum


# Evaluation Status
class EvaluationStatus(str, Enum):
    EVALUATION_FAILED = "Evaluation Failed"
    DONE = "Done"
    PARTIAL_FAILURE = "Partial Failure"
    FAILED = "Failed"


class AgentMetrics(str, Enum):
    """Enum for agent metric types."""

    HELPFULNESS = "helpfulness"
    COHERENCE = "coherence"
    VERBOSITY = "verbosity"
    RELEVANCE = "relevance"
    FAITHFULNESS = "faithfulness"
    USER_GOAL_COMPLETION = "user_goal_completion"
    GOAL_COMPLETION = "goal_completion"  # deprecated: use USER_GOAL_COMPLETION
    AGENT_BEHAVIOR_FAILURE = "agent_behavior_failure"
    TOOL_CALL_BEHAVIOR_FAILURE = "tool_call_behavior_failure"


class EvaluationOutcomes(str, Enum):
    """Enum for evaluation outcomes."""

    EVALUATION_RUN_FAILED = "evaluation_run_failed"
    SKIPPED_GOOD_PERFORMANCE = "skipped_good_performance"
    AGENT_API_ERROR = "agent_api_error"
    NO_FAILURE = "no failure"


# Define nested enum as a class attribute after EvaluationOutcomes is defined
class AgentBehaviorFailureType(str, Enum):
    """Enum for agent behavior failure types (nested in EvaluationOutcomes)."""

    LACK_OF_SPECIFIC_INFORMATION = "lack of specific information"
    FAILURE_TO_ASK_FOR_CLARIFICATION = "failure to ask for clarification"
    DISOBEY_USER_REQUEST = "disobey user request"
    REPETITION = "repetition"
    FALSE_INFORMATION = "false information"
    UNSAFE_ACTION = "unsafe action"
    UNSAFE_STATE = "unsafe state"
    CONSTRAINT_VIOLATION = "constraint_violation"
    NO_FAILURE = "no failure"


# Attach as nested class
EvaluationOutcomes.AgentBehaviorFailureType = AgentBehaviorFailureType


# Mapping of agent behavior failure categories to severity levels.
AGENT_BEHAVIOR_FAILURE_SEVERITY = {
    "unsafe action": "critical",
    "unsafe state": "critical",
    "false information": "critical",
    "disobey user request": "high",
    "constraint_violation": "high",
    "lack of specific information": "medium",
    "failure to ask for clarification": "medium",
    "repetition": "low",
}
