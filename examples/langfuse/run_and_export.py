# SPDX-License-Identifier: Apache-2.0
"""Real end-to-end run: simulate -> evaluate -> export to Langfuse.

Unlike ``export_to_langfuse.py`` (which builds sample data), this script runs
an actual arksim simulation against a live agent, evaluates it, and exports
the result to Langfuse. Everything is driven by environment variables.

Self-contained OpenAI setup (agent-under-test is a second OpenAI model):

    pip install 'arksim[langfuse]'

    # Drives the synthetic user + evaluator AND the agent-under-test:
    export OPENAI_API_KEY=sk-...

    # Langfuse target (self-hosted example):
    export LANGFUSE_HOST=http://localhost:3000
    export LANGFUSE_PUBLIC_KEY=pk-lf-...
    export LANGFUSE_SECRET_KEY=sk-lf-...

    python examples/langfuse/run_and_export.py

Optional overrides (sensible defaults shown):

    export ARKSIM_MODEL=gpt-5.1            # sim/eval LLM (provider=openai)
    export ARKSIM_PROVIDER=openai
    export AGENT_MODEL=gpt-4o-mini         # agent-under-test model
    export AGENT_ENDPOINT=https://api.openai.com/v1/chat/completions
    export AGENT_API_KEY=$OPENAI_API_KEY   # auth for the agent endpoint
"""

from __future__ import annotations

import asyncio
import os
import tempfile

from arksim import (
    AgentConfig,
    ChatCompletionsConfig,
    EvaluationInput,
    Scenarios,
    SimulationInput,
    run_evaluation,
    run_simulation,
)
from arksim.integrations.langfuse import export_to_langfuse
from arksim.scenario.entities import (
    ExpectedToolCall,
    KnowledgeItem,
    Scenario,
    ToolCallsAssertion,
)


def build_scenarios() -> Scenarios:
    return Scenarios(
        schema_version="v1",
        scenarios=[
            Scenario(
                scenario_id="cancel-order",
                user_id="cust-001",
                goal="Cancel order #42 and confirm the refund.",
                agent_context=(
                    "You are a customer support agent for an online store. "
                    "You can cancel orders and issue refunds."
                ),
                knowledge=[
                    KnowledgeItem(content="Order #42 is eligible for cancellation.")
                ],
                user_profile="A busy customer who wants a quick resolution.",
                assertions=[
                    ToolCallsAssertion(
                        type="tool_calls",
                        expected=[
                            ExpectedToolCall(
                                name="cancel_order", arguments={"order_id": 42}
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


def build_agent_config() -> AgentConfig:
    """A chat-completions agent-under-test (OpenAI by default)."""
    endpoint = os.environ.get(
        "AGENT_ENDPOINT", "https://api.openai.com/v1/chat/completions"
    )
    agent_model = os.environ.get("AGENT_MODEL", "gpt-4o-mini")
    # AGENT_API_KEY falls back to OPENAI_API_KEY. The ${ENV_VAR} syntax is
    # resolved by arksim at request time, so the key is never hard-coded.
    auth_var = "AGENT_API_KEY" if os.environ.get("AGENT_API_KEY") else "OPENAI_API_KEY"
    return AgentConfig(
        agent_name="support-bot",
        agent_type="chat_completions",
        api_config=ChatCompletionsConfig(
            endpoint=endpoint,
            headers={
                "Authorization": f"Bearer ${{{auth_var}}}",
                "Content-Type": "application/json",
            },
            body={"model": agent_model},
        ),
    )


def require_env() -> None:
    missing = [
        v
        for v in ("OPENAI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
        if not os.environ.get(v)
    ]
    if missing:
        raise SystemExit(
            "Missing env vars: "
            + ", ".join(missing)
            + "\nSee the module docstring for the full list."
        )


def main() -> None:
    require_env()
    scenarios = build_scenarios()

    with tempfile.TemporaryDirectory(prefix="arksim-lf-") as workdir:
        sim_settings = SimulationInput(
            agent_config=build_agent_config(),
            model=os.environ.get("ARKSIM_MODEL", "gpt-5.1"),
            provider=os.environ.get("ARKSIM_PROVIDER", "openai"),
            num_conversations_per_scenario=1,
            max_turns=4,
            output_file_path=os.path.join(workdir, "simulation.json"),
        )

        print("Running simulation...")
        simulation = asyncio.run(run_simulation(sim_settings, scenarios=scenarios))

        print("Running evaluation...")
        eval_settings = EvaluationInput(
            model=os.environ.get("ARKSIM_MODEL", "gpt-5.1"),
            provider=os.environ.get("ARKSIM_PROVIDER", "openai"),
            output_dir=os.path.join(workdir, "evaluation"),
            generate_html_report=False,
        )
        evaluation = run_evaluation(
            eval_settings, simulation=simulation, scenarios=scenarios
        )

        print("Exporting to Langfuse...")
        result = export_to_langfuse(
            scenarios,
            simulation,
            evaluation,
            dataset_name="arksim-live",
            run_name="arksim-live-run",
            host=os.environ.get("LANGFUSE_HOST"),
        )

    print("\nExport complete.")
    print(result.format())


if __name__ == "__main__":
    main()
