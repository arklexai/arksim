"""
Example script that runs the full pipeline (simulation → evaluation)
programmatically, with custom metrics passed to the evaluator.

Usage:
    python run_pipeline.py
"""

import asyncio
import os

from arksim.config import AgentConfig
from arksim.evaluator import Evaluator, EvaluationParams
from arksim.llms.chat import LLM
from arksim.scenario import Scenarios
from arksim.simulation_engine import Simulator, SimulationParams
from arksim.utils.html_report.generate_html_report import (
    HtmlReportParams,
    generate_html_report,
)
from custom_metrics import ConversionMetric, UpsellBehaviorMetric


async def main() -> None:
    # ── Paths ──────────────────────────────────────────────────────
    agent_setup_dir = os.path.dirname(__file__)
    results_dir = os.path.join(agent_setup_dir, "results")

    scenario_file_path = os.path.join(agent_setup_dir, "scenarios.json")
    agent_config_path = os.path.join(agent_setup_dir, "agent_config.json")
    simulation_output_file = os.path.join(results_dir, "simulation", "simulation.json")
    evaluation_output_dir = os.path.join(results_dir, "evaluation")

    model = "gpt-5.1"
    max_turns = 5

    # ── Step 1: Run Simulation ─────────────────────────────────────
    print("=" * 60)
    print("Step 1/2: Running Simulation...")
    print("=" * 60)

    scenario_output = Scenarios.load(scenario_file_path)
    agent_config = AgentConfig.load(agent_config_path)
    llm = LLM(model=model)

    simulation_params = SimulationParams(
        num_convos_per_scenario=1,
        max_turns=max_turns,
        num_workers="auto",
        output_file_path=simulation_output_file,
    )

    simulator = Simulator(
        agent_config=agent_config,
        simulator_params=simulation_params,
        llm=llm,
    )
    simulation_output = await simulator.simulate(scenario_output)
    await simulator.save()

    # ── Step 2: Evaluate (with custom metrics) ─────────────────────
    print("\n" + "=" * 60)
    print("Step 2/2: Evaluating Results...")
    print("=" * 60)

    evaluator_params = EvaluationParams(
        output_dir=evaluation_output_dir,
        num_workers="auto",
        custom_metrics=[
            ConversionMetric(),
            # Add more custom metrics here...
        ],
        custom_qualitative_metrics=[
            UpsellBehaviorMetric(),
            # Add more custom qualitative metrics here...
        ],
    )

    evaluator = Evaluator(evaluator_params, llm=llm)
    evaluator_output = await asyncio.to_thread(evaluator.evaluate, simulation_output)
    evaluator.display_evaluation_summary()
    evaluator.save_results()

    # Generate HTML report
    html_output_path = os.path.join(evaluation_output_dir, "final_report.html")
    report_params = HtmlReportParams(
        simulation=simulation_output,
        evaluation=evaluator_output,
        scenarios=scenario_output,
        output_path=html_output_path,
        chat_id_to_label=evaluator.chat_id_to_label,
    )
    await asyncio.to_thread(generate_html_report, report_params)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"Results saved to: {results_dir}")
    print(f"HTML report: {html_output_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
