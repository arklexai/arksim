# SPDX-License-Identifier: Apache-2.0
"""Pure translation between the EvalHub job contract and arksim.

Only in-memory transforms live here: no filesystem, network, or LLM calls. That
keeps this module unit-testable without credentials or a running EvalHub. The
I/O-bound pieces (reading the mounted model secret, running the simulation,
reading artifact files) live in :mod:`arksim_evalhub.adapter`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

from evalhub.adapter import EvaluationResult, ModelConfig
from pydantic import BaseModel, ConfigDict, Field, model_validator

from arksim import A2AConfig, AgentConfig, ChatCompletionsConfig, Scenarios
from arksim.constants import DEFAULT_MODEL, DEFAULT_PROVIDER

if TYPE_CHECKING:
    from typing_extensions import Self

    from arksim.evaluator.entities import Evaluation

# Conversation-level scores reduced to scalar MLflow metrics. These are public,
# stable fields on arksim's ConversationEvaluation.
_CONVERSATION_SCORES = (
    "overall_agent_score",
    "goal_completion_score",
    "turn_success_ratio",
)

# Body keys the adapter owns; the target model and message history must not be
# overridden by caller-supplied request_body.
_RESERVED_BODY_KEYS = {"model", "messages"}


class ArksimJobParameters(BaseModel):
    """Typed view of the EvalHub ``JobSpec.parameters`` freeform JSON.

    EvalHub passes all benchmark-specific config through ``parameters``. This
    model is the contract arksim expects to find there. ``extra="forbid"`` turns
    a misspelled key into an immediate, actionable error rather than a setting
    that is silently ignored.

    Note the two distinct model roles. ``JobSpec.model`` (handled separately) is
    the *target agent under test*. The ``simulator_*`` and ``evaluator_*`` fields
    below are arksim's own LLMs: one drives the simulated user, the other judges
    the resulting transcripts.
    """

    model_config = ConfigDict(extra="forbid")

    # -- Scenarios (required) -------------------------------------------------
    scenarios: Scenarios = Field(
        ...,
        description=(
            "arksim scenarios. Accepts a full {schema_version, scenarios} object "
            "or a bare list of scenario objects."
        ),
    )

    # -- How arksim drives the target agent (JobSpec.model.url) ---------------
    protocol: Literal["chat_completions", "a2a"] = Field(
        default="chat_completions",
        description=(
            "How arksim talks to the target agent at JobSpec.model.url. "
            "'chat_completions' for an OpenAI-compatible endpoint, 'a2a' for an "
            "A2A agent server."
        ),
    )
    target_agent_name: str = Field(
        default="target-agent",
        description="Display name for the target agent in arksim output.",
    )
    request_body: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extra chat_completions body fields merged into each request "
            "(e.g. {'temperature': 0}). Ignored when protocol='a2a'. Must not "
            "contain 'model' or 'messages'."
        ),
    )
    extra_headers: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Extra HTTP headers sent to the target agent. An 'Authorization' "
            "header here overrides the resolved api-key."
        ),
    )
    target_api_key_env: str | None = Field(
        default=None,
        description=(
            "Env var to read the target api-key from when no EvalHub model secret "
            "is mounted. Intended for local runs only."
        ),
    )

    # -- Simulator LLM (generates the simulated user) -------------------------
    simulator_model: str = Field(
        default=DEFAULT_MODEL,
        description=f"LLM that plays the simulated user (arksim default: {DEFAULT_MODEL}).",
    )
    simulator_provider: str | None = Field(
        default=DEFAULT_PROVIDER, description="Provider for simulator_model."
    )
    num_conversations_per_scenario: int = Field(
        default=2, ge=1, description="Conversations simulated per scenario."
    )
    max_turns: int = Field(
        default=5, ge=1, description="Maximum turns per conversation."
    )
    num_workers: Annotated[int, Field(ge=1)] | Literal["auto"] = Field(
        default=50,
        description="Parallel simulation/evaluation workers: a positive int, or 'auto'.",
    )

    # -- Evaluator (judge) LLM + metric selection -----------------------------
    evaluator_model: str = Field(
        default=DEFAULT_MODEL,
        description=f"LLM that judges transcripts (arksim default: {DEFAULT_MODEL}).",
    )
    evaluator_provider: str | None = Field(
        default=DEFAULT_PROVIDER, description="Provider for evaluator_model."
    )
    metrics_to_run: list[str] | None = Field(
        default=None,
        description=(
            "Built-in metric names to run; None uses arksim's defaults. Options: "
            "helpfulness, coherence, verbosity, relevance, faithfulness, "
            "goal_completion, agent_behavior_failure."
        ),
    )
    numeric_thresholds: dict[str, float] | None = Field(
        default=None,
        description=(
            "Optional minimum scores keyed by metric name "
            "(e.g. {'overall_score': 0.6})."
        ),
    )
    qualitative_failure_labels: dict[str, list[str]] | None = Field(
        default=None,
        description=(
            "Optional failing labels per qualitative metric keyed by metric name "
            "(e.g. {'agent_behavior_failure': ['violated']})."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _wrap_bare_scenario_list(cls, data: object) -> object:
        """Allow ``scenarios`` to be a bare list, not only a Scenarios object."""
        if isinstance(data, dict) and isinstance(data.get("scenarios"), list):
            data = {
                **data,
                "scenarios": {"schema_version": "v1", "scenarios": data["scenarios"]},
            }
        return data

    @model_validator(mode="after")
    def _validate(self) -> Self:
        """Reject configurations that would fail late or silently mislead."""
        if not self.scenarios.scenarios:
            raise ValueError("'scenarios' must contain at least one scenario")
        overridden = _RESERVED_BODY_KEYS & set(self.request_body)
        if overridden:
            raise ValueError(
                f"request_body must not override reserved keys: {sorted(overridden)}; "
                "the target model and message history are set by the adapter"
            )
        return self


def build_agent_config(
    model: ModelConfig,
    params: ArksimJobParameters,
    api_key: str | None,
) -> AgentConfig:
    """Map an EvalHub target model + parameters into an arksim ``AgentConfig``.

    The target endpoint comes from ``model.url``; ``model.name`` becomes the
    chat-completions ``model`` body field. The resolved api key (already read
    from the mounted secret by the caller) is attached as a bearer token.
    ``ModelConfig`` already guarantees ``url`` and ``name`` are non-empty.
    """
    headers: dict[str, str] = dict(params.extra_headers)
    if api_key:
        headers.setdefault("Authorization", f"Bearer {api_key}")

    if params.protocol == "a2a":
        return AgentConfig(
            agent_type="a2a",
            agent_name=params.target_agent_name,
            api_config=A2AConfig(endpoint=model.url, headers=headers or None),
        )

    body: dict[str, Any] = {
        "model": model.name,
        "messages": [],
        **params.request_body,
    }
    return AgentConfig(
        agent_type="chat_completions",
        agent_name=params.target_agent_name,
        api_config=ChatCompletionsConfig(
            endpoint=model.url,
            headers=headers or None,
            body=body,
        ),
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def aggregate_metrics(evaluation: Evaluation) -> list[EvaluationResult]:
    """Reduce per-conversation arksim scores to EvalHub scalar metrics.

    EvalHub records scalar metrics; per-conversation and per-turn detail surface
    as MLflow artifacts instead (see ``adapter._collect_artifacts``). Pass/fail
    is applied by EvalHub via the provider's ``primary_score``/``pass_criteria``,
    so this function stays purely descriptive.
    """
    convos = evaluation.conversations
    n = len(convos)
    results = [
        EvaluationResult(
            metric_name="num_conversations",
            metric_value=n,
            metric_type="int",
            num_samples=n,
        )
    ]
    if n == 0:
        return results
    for name in _CONVERSATION_SCORES:
        results.append(
            EvaluationResult(
                metric_name=name,
                metric_value=_mean([getattr(c, name) for c in convos]),
                metric_type="float",
                num_samples=n,
            )
        )
    return results


def compute_overall_score(evaluation: Evaluation) -> float | None:
    """Mean overall agent score across conversations, or None when empty."""
    convos = evaluation.conversations
    if not convos:
        return None
    return _mean([c.overall_agent_score for c in convos])
