# SPDX-License-Identifier: Apache-2.0
"""End-to-end example: export an arksim run to a (self-hosted) Langfuse.

This script builds a small, self-contained arksim ``Scenarios`` /
``Simulation`` / ``Evaluation`` in memory (no LLM or agent keys needed) and
pushes them to Langfuse as a dataset, a dataset run of traces, and scores.

To export a *real* run instead, replace ``build_sample_*`` below with:

    from arksim import run_simulation, run_evaluation, Scenarios
    scenarios = Scenarios.load("scenarios.json")
    simulation = await run_simulation(settings, scenarios=scenarios)
    evaluation = run_evaluation(eval_settings, simulation=simulation,
                                scenarios=scenarios)

Usage:

    pip install 'arksim[langfuse]'
    export LANGFUSE_HOST=http://localhost:3000
    export LANGFUSE_PUBLIC_KEY=pk-lf-...
    export LANGFUSE_SECRET_KEY=sk-lf-...
    python examples/langfuse/export_to_langfuse.py
"""

from __future__ import annotations

import os

from arksim.evaluator.base_metric import QualResult, QuantResult
from arksim.evaluator.entities import (
    ConversationEvaluation,
    Evaluation,
    TurnEvaluation,
)
from arksim.integrations.langfuse import export_to_langfuse
from arksim.scenario.entities import (
    ExpectedToolCall,
    KnowledgeItem,
    Scenario,
    Scenarios,
    ToolCallsAssertion,
)
from arksim.simulation_engine.entities import (
    Conversation,
    Message,
    SimulatedUserPrompt,
    Simulation,
)
from arksim.simulation_engine.tool_types import ToolCall, ToolCallSource


def build_sample_scenarios() -> Scenarios:
    return Scenarios(
        schema_version="v1",
        scenarios=[
            Scenario(
                scenario_id="cancel-order",
                user_id="cust-001",
                goal="Cancel order #42 and get a refund confirmation.",
                agent_context="You are a customer support agent for an online store.",
                knowledge=[
                    KnowledgeItem(content="Order #42 is eligible for cancellation.")
                ],
                user_profile="A busy customer who wants a quick resolution.",
                assertions=[
                    ToolCallsAssertion(
                        type="tool_calls",
                        expected=[
                            ExpectedToolCall(
                                name="cancel_order",
                                arguments={"order_id": 42},
                                arg_match_mode="partial",
                            )
                        ],
                        match_mode="contains",
                    )
                ],
            ),
            Scenario(
                scenario_id="store-hours",
                user_id="cust-002",
                goal="Find out the weekend opening hours.",
                agent_context="You are a customer support agent for an online store.",
                user_profile="A polite first-time customer.",
            ),
        ],
    )


def build_sample_simulation() -> Simulation:
    return Simulation(
        schema_version="v1.1",
        simulator_version="v1",
        conversations=[
            Conversation(
                conversation_id="conv-cancel-order",
                scenario_id="cancel-order",
                conversation_history=[
                    Message(
                        turn_id=0,
                        role="simulated_user",
                        content="Hi, I need to cancel order 42.",
                    ),
                    Message(
                        turn_id=0,
                        role="assistant",
                        content="Sure, let me pull that up.",
                    ),
                    Message(
                        turn_id=1,
                        role="simulated_user",
                        content="Thanks, please cancel it.",
                    ),
                    Message(
                        turn_id=1,
                        role="assistant",
                        content="Order #42 has been cancelled and refunded.",
                        tool_calls=[
                            ToolCall(
                                id="call-1",
                                name="cancel_order",
                                arguments={"order_id": 42},
                                result='{"status": "cancelled", "refund": true}',
                                source=ToolCallSource.OTEL_TRACE,
                            )
                        ],
                    ),
                ],
                simulated_user_prompt=SimulatedUserPrompt(
                    simulated_user_prompt_template="(template omitted)", variables={}
                ),
            ),
            Conversation(
                conversation_id="conv-store-hours",
                scenario_id="store-hours",
                conversation_history=[
                    Message(
                        turn_id=0,
                        role="simulated_user",
                        content="What are your weekend hours?",
                    ),
                    Message(
                        turn_id=0,
                        role="assistant",
                        content="We're open 10am-6pm on weekends.",
                    ),
                ],
                simulated_user_prompt=SimulatedUserPrompt(
                    simulated_user_prompt_template="(template omitted)", variables={}
                ),
            ),
        ],
    )


def build_sample_evaluation() -> Evaluation:
    return Evaluation(
        schema_version="v1.1",
        generated_at="2026-07-07T00:00:00Z",
        evaluator_version="v1",
        evaluation_id="eval-demo",
        simulation_id="sim-demo",
        conversations=[
            ConversationEvaluation(
                conversation_id="conv-cancel-order",
                goal_completion_score=1.0,
                goal_completion_reason="Order cancelled and refund confirmed.",
                turn_success_ratio=1.0,
                overall_agent_score=0.95,
                evaluation_status="done",
                turn_scores=[
                    TurnEvaluation(
                        turn_id=0,
                        scores=[
                            QuantResult(
                                name="helpfulness",
                                value=4.0,
                                reason="Acknowledged request.",
                            )
                        ],
                        turn_score=4.0,
                        turn_behavior_failure="skipped_good_performance",
                        turn_behavior_failure_reason="",
                    ),
                    TurnEvaluation(
                        turn_id=1,
                        scores=[
                            QuantResult(
                                name="helpfulness", value=5.0, reason="Resolved fully."
                            ),
                            QuantResult(name="coherence", value=5.0, reason="Clear."),
                        ],
                        turn_score=5.0,
                        turn_behavior_failure="skipped_good_performance",
                        turn_behavior_failure_reason="",
                        qual_scores=[QualResult(name="tone", value="professional")],
                    ),
                ],
            ),
            ConversationEvaluation(
                conversation_id="conv-store-hours",
                goal_completion_score=1.0,
                goal_completion_reason="Provided the hours.",
                turn_success_ratio=1.0,
                overall_agent_score=1.0,
                evaluation_status="done",
                turn_scores=[
                    TurnEvaluation(
                        turn_id=0,
                        scores=[
                            QuantResult(
                                name="helpfulness", value=5.0, reason="Direct answer."
                            )
                        ],
                        turn_score=5.0,
                        turn_behavior_failure="skipped_good_performance",
                        turn_behavior_failure_reason="",
                    ),
                ],
            ),
        ],
        unique_errors=[],
    )


def main() -> None:
    missing = [
        var
        for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
        if not os.environ.get(var)
    ]
    if missing:
        raise SystemExit(
            "Missing env vars: "
            + ", ".join(missing)
            + "\nSet LANGFUSE_HOST (e.g. http://localhost:3000), "
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY first."
        )

    scenarios = build_sample_scenarios()
    simulation = build_sample_simulation()
    evaluation = build_sample_evaluation()

    # host/keys are read from LANGFUSE_* env vars by the SDK when omitted.
    result = export_to_langfuse(
        scenarios,
        simulation,
        evaluation,
        dataset_name="arksim-demo",
        run_name="arksim-demo-run",
        host=os.environ.get("LANGFUSE_HOST"),
    )

    print("Export complete.")
    print(result.format())
    print(
        "\nOpen your Langfuse UI and look for:\n"
        "  - Dataset 'arksim-demo' with 2 items (cancel-order, store-hours)\n"
        "  - Dataset run 'arksim-demo-run' with 2 linked traces\n"
        "  - Scores on each trace (overall_agent_score, goal_completion, ...) "
        "and per-turn scores (helpfulness, coherence, tone)."
    )


if __name__ == "__main__":
    main()
