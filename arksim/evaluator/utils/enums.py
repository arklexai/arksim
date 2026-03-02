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
    GOAL_COMPLETION = "goal_completion"
    AGENT_BEHAVIOR_FAILURE = "agent_behavior_failure"
    UNIQUE_BUGS = "unique_bugs"


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
    NO_FAILURE = "no failure"


# Attach as nested class
EvaluationOutcomes.AgentBehaviorFailureType = AgentBehaviorFailureType


# Mapping of agent behavior failure categories to severity levels.
AGENT_BEHAVIOR_FAILURE_SEVERITY = {
    "false information": "critical",
    "disobey user request": "high",
    "lack of specific information": "medium",
    "failure to ask for clarification": "medium",
    "repetition": "low",
}
