# SPDX-License-Identifier: Apache-2.0
"""ArkSim quality gate pytest test.

Copy to tests/test_agent_quality.py in your repo.

Install:
    pip install arksim pytest pytest-asyncio
"""

import pytest

# TODO: import your agent class — it must subclass BaseAgent.
#
# Option A — subclass BaseAgent directly in your agent module:
#
#   from arksim.simulation_engine.agent.base import BaseAgent
#
#   class MyAgent(BaseAgent):
#       async def get_chat_id(self) -> str:
#           return str(id(self))
#
#       async def execute(self, user_query: str, **kwargs) -> str:
#           # Your agent logic here
#           ...
#
# Option B — thin adapter around an existing agent:
#
#   class MyAgentAdapter(BaseAgent):
#       def __init__(self, agent_config):
#           super().__init__(agent_config)
#           self.agent = MyExistingAgent()
#
#       async def get_chat_id(self) -> str:
#           return str(id(self))
#
#       async def execute(self, user_query: str, **kwargs) -> str:
#           return await self.agent.process(user_query)
#
from my_agent import MyAgent  # TODO: replace with your agent class

from arksim.config import AgentConfig, CustomConfig
from arksim.evaluator import (
    EvaluationInput,
    check_numeric_thresholds,
    check_qualitative_failure_labels,
    run_evaluation,
)
from arksim.simulation_engine import SimulationInput, run_simulation

# ── Quality gate thresholds ────────────────────────────────────────────────────

SCORE_THRESHOLD = 0.7

NUMERIC_THRESHOLDS = {
    "goal_completion": 0.8,
    "faithfulness": 3.5,
}

QUALITATIVE_FAILURE_LABELS = {
    "agent_behavior_failure": ["false information", "disobey user request"],
}


# ── Test ───────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_quality() -> None:
    # ── Simulate ──────────────────────────────────────────────────────────────
    agent_config = AgentConfig(
        agent_type="custom",
        agent_name="my-agent",
        custom_config=CustomConfig(agent_class=MyAgent),
    )

    simulation = await run_simulation(
        SimulationInput(
            agent_config=agent_config,
            scenario_file_path="./arksim/scenarios.json",
            num_conversations_per_scenario=3,
            max_turns=5,
            output_file_path="./arksim/results/simulation.json",
        )
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    evaluation = run_evaluation(
        EvaluationInput(
            output_dir="./arksim/results/evaluation",
            metrics_to_run=[
                "faithfulness",
                "helpfulness",
                "coherence",
                "relevance",
                "goal_completion",
                "agent_behavior_failure",
            ],
            generate_html_report=True,
            model="gpt-4o",
            provider="openai",
            num_workers=10,
        ),
        simulation=simulation,
    )

    # ── Assert quality gates ───────────────────────────────────────────────────
    # Overall score gate
    failed = [
        f"  {c.conversation_id}: {c.overall_agent_score:.2f} < {SCORE_THRESHOLD}"
        for c in evaluation.conversations
        if c.overall_agent_score < SCORE_THRESHOLD
    ]
    assert not failed, "Overall score threshold failed:\n" + "\n".join(failed)

    # Per-metric gate
    assert check_numeric_thresholds(
        evaluation,
        numeric_thresholds=NUMERIC_THRESHOLDS,
    ), "Numeric threshold check failed — see arksim/results/evaluation for details"

    # Qualitative gate
    assert check_qualitative_failure_labels(
        evaluation,
        qualitative_failure_labels=QUALITATIVE_FAILURE_LABELS,
    ), (
        "Qualitative failure label check failed — see arksim/results/evaluation for details"
    )
