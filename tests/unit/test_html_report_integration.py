# SPDX-License-Identifier: Apache-2.0
"""Integration test for HTML report generation.

Uses importlib to load modules directly from file paths, bypassing
the arksim.__init__.py chains that pull in heavy deps
(langchain, azure, etc.) that may not be installed in the test env.

All sys.modules patching is done inside a module-scoped fixture so that
monkeypatch restores the original entries automatically after the test
module finishes — no manual bookkeeping required.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from _pytest.monkeypatch import MonkeyPatch
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path constant and raw loader (no sys.modules side-effects at import time)
# ---------------------------------------------------------------------------
_ARKSIM_ROOT = Path(__file__).resolve().parents[2] / "arksim"


def _load_module(name: str, filepath: Path) -> types.ModuleType:
    """Load a Python module from filepath into sys.modules[name]."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Minimal Pydantic stubs used as arksim.scenario replacements
# (pure class definitions — no sys.modules manipulation)
# ---------------------------------------------------------------------------
class _KnowledgeItem(BaseModel):
    content: str


class _Scenario(BaseModel):
    scenario_id: str
    goal: str = ""
    user_profile: str = ""
    knowledge: list[_KnowledgeItem] = Field(default_factory=list)
    origin: dict[str, Any] = Field(default_factory=dict)


class _Scenarios(BaseModel):
    schema_version: str = "1.0"
    scenarios: list[_Scenario] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def _module_mp() -> Generator[MonkeyPatch, None, None]:
    """Module-scoped MonkeyPatch that restores sys.modules after all tests."""
    with MonkeyPatch.context() as mp:
        yield mp


@pytest.fixture(scope="module")
def html_report_env(_module_mp: MonkeyPatch) -> SimpleNamespace:
    """Load the HTML report stack with isolated sys.modules via monkeypatch.

    monkeypatch.setitem records the original value (or absence) and restores
    it automatically when the module scope exits — no manual bookkeeping.
    """
    mp = _module_mp

    def load(name: str, rel: str) -> types.ModuleType:
        spec = importlib.util.spec_from_file_location(name, _ARKSIM_ROOT / rel)
        mod = importlib.util.module_from_spec(spec)
        mp.setitem(sys.modules, name, mod)
        spec.loader.exec_module(mod)
        return mod

    # 1. Lightweight deps
    load("arksim.evaluator.utils.enums", "evaluator/utils/enums.py")
    load("arksim.evaluator.base_metric", "evaluator/base_metric.py")

    # 1b. Load usage module directly so simulation_engine/entities.py can
    #     import TokenUsage without triggering the heavy llms.chat chain.
    usage_mod = load("arksim.llms.chat.base.usage", "llms/chat/base/usage.py")

    # 1c. Load tool_types directly so evaluator/entities.py can import ToolCall
    #     without triggering the simulation_engine.__init__ → simulator → a2a chain.
    tool_types_mod = load(
        "arksim.simulation_engine.tool_types", "simulation_engine/tool_types.py"
    )
    # Expose ToolCall on the package stub so relative package resolution works.
    sim_pkg_early = types.ModuleType("arksim.simulation_engine")
    sim_pkg_early.ToolCall = tool_types_mod.ToolCall
    mp.setitem(sys.modules, "arksim.simulation_engine", sim_pkg_early)

    # 1d. Load scenario entities and core.state directly so
    #     simulation_engine/entities.py can import ConversationLedger without
    #     triggering the scenario.__init__ generation chain.
    scenario_entities_real = load("arksim.scenario.entities", "scenario/entities.py")
    scenario_pkg_early = types.ModuleType("arksim.scenario")
    for _name in ("BehaviorParams", "GoalChecklistItem", "Scenario"):
        setattr(scenario_pkg_early, _name, getattr(scenario_entities_real, _name))
    mp.setitem(sys.modules, "arksim.scenario", scenario_pkg_early)
    load("arksim.simulation_engine.core.state", "simulation_engine/core/state.py")

    # 2. Entities
    entities_mod = load("arksim.evaluator.entities", "evaluator/entities.py")

    # 3. Simulation engine entities
    sim_entities_mod = load(
        "arksim.simulation_engine.entities", "simulation_engine/entities.py"
    )

    # 4. Stub arksim.simulation_engine package
    sim_pkg = types.ModuleType("arksim.simulation_engine")
    sim_pkg.Conversation = sim_entities_mod.Conversation
    sim_pkg.Simulation = sim_entities_mod.Simulation
    sim_pkg.combine_knowledge = MagicMock()
    mp.setitem(sys.modules, "arksim.simulation_engine", sim_pkg)

    # 5. Stub arksim.scenario (and arksim.scenario.entities needed by evaluator.__init__)
    scenario_entities_mod = types.ModuleType("arksim.scenario.entities")
    scenario_entities_mod.AssertionType = MagicMock()
    scenario_entities_mod.ToolCallsAssertion = MagicMock()
    scenario_entities_mod.Scenario = _Scenario
    scenario_entities_mod.Scenarios = _Scenarios
    mp.setitem(sys.modules, "arksim.scenario.entities", scenario_entities_mod)

    scenario_mod = types.ModuleType("arksim.scenario")
    scenario_mod.Scenario = _Scenario
    scenario_mod.Scenarios = _Scenarios
    scenario_mod.entities = scenario_entities_mod
    mp.setitem(sys.modules, "arksim.scenario", scenario_mod)

    # 6. Stub heavy deps
    mp.setitem(sys.modules, "arksim.llms", MagicMock())
    mp.setitem(sys.modules, "arksim.llms.chat", MagicMock())

    # 6b. Stub arksim.evaluator package so generate_html_report.py can import
    #     submodules without running arksim/evaluator/__init__.py (which pulls
    #     in the full evaluator + simulator + a2a chain).
    evaluator_pkg = types.ModuleType("arksim.evaluator")
    mp.setitem(sys.modules, "arksim.evaluator", evaluator_pkg)

    # Load builtin_metrics directly — it only needs LLM (already a MagicMock),
    # base_metric (already loaded), and lightweight utils.
    load("arksim.evaluator.utils.constants", "evaluator/utils/constants.py")
    load("arksim.evaluator.utils.prompts", "evaluator/utils/prompts.py")
    load("arksim.evaluator.utils.schema", "evaluator/utils/schema.py")
    load("arksim.evaluator.builtin_metrics", "evaluator/builtin_metrics.py")

    # 7. Load generate_html_report
    gen_report_mod = load(
        "arksim.utils.html_report.generate_html_report",
        "utils/html_report/generate_html_report.py",
    )

    return SimpleNamespace(
        Evaluation=entities_mod.Evaluation,
        ConversationEvaluation=entities_mod.ConversationEvaluation,
        TurnEvaluation=entities_mod.TurnEvaluation,
        QuantResult=entities_mod.QuantResult,
        UniqueError=entities_mod.UniqueError,
        Occurrence=entities_mod.Occurrence,
        Conversation=sim_entities_mod.Conversation,
        Simulation=sim_entities_mod.Simulation,
        Message=sim_entities_mod.Message,
        SimulatedUserPrompt=sim_entities_mod.SimulatedUserPrompt,
        TokenUsage=usage_mod.TokenUsage,
        HtmlReportParams=gen_report_mod.HtmlReportParams,
        generate_html_report=gen_report_mod.generate_html_report,
    )


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def _build_test_data(env: SimpleNamespace) -> tuple[Any, Any, dict[str, str]]:
    """Build minimal test data for HTML report generation."""
    conversations = env.Simulation(
        schema_version="1.0",
        simulator_version="test",
        conversations=[
            env.Conversation(
                conversation_id="conv-uuid-1",
                scenario_id="sc-1",
                conversation_history=[
                    env.Message(
                        turn_id=0, role="simulated_user", content="What is the weather?"
                    ),
                    env.Message(
                        turn_id=0, role="assistant", content="It is sunny today."
                    ),
                    env.Message(turn_id=1, role="simulated_user", content="Thanks!"),
                    env.Message(turn_id=1, role="assistant", content="You're welcome!"),
                ],
                simulated_user_prompt=env.SimulatedUserPrompt(
                    simulated_user_prompt_template="You are a user.",
                    variables={"goal": "Get weather info"},
                ),
            ),
            env.Conversation(
                conversation_id="conv-uuid-2",
                scenario_id="sc-2",
                conversation_history=[
                    env.Message(
                        turn_id=0,
                        role="simulated_user",
                        content="Tell me about insurance",
                    ),
                    env.Message(
                        turn_id=0, role="assistant", content="Sure, let me explain."
                    ),
                ],
                simulated_user_prompt=env.SimulatedUserPrompt(
                    simulated_user_prompt_template="You are a user.",
                    variables={"goal": "Learn about insurance"},
                ),
            ),
        ],
    )

    evaluation_results = env.Evaluation(
        schema_version="1.0",
        generated_at="2025-01-01T00:00:00Z",
        evaluator_version="v1",
        evaluation_id="eval-uuid-1",
        simulation_id="sim-uuid-1",
        conversations=[
            env.ConversationEvaluation(
                conversation_id="conv-uuid-1",
                goal_completion_score=1.0,
                goal_completion_reason="Completed",
                turn_success_ratio=1.0,
                overall_agent_score=1.0,
                evaluation_status="Done",
                turn_scores=[
                    env.TurnEvaluation(
                        turn_id=0,
                        scores=[
                            env.QuantResult(
                                name="helpfulness", value=4.0, reason="Good"
                            ),
                            env.QuantResult(name="coherence", value=4.0, reason="Good"),
                        ],
                        turn_score=4.0,
                        turn_behavior_failure="no failure",
                        turn_behavior_failure_reason="All good",
                    ),
                    env.TurnEvaluation(
                        turn_id=1,
                        scores=[
                            env.QuantResult(
                                name="helpfulness", value=4.5, reason="Great"
                            ),
                            env.QuantResult(
                                name="coherence", value=4.5, reason="Great"
                            ),
                        ],
                        turn_score=4.5,
                        turn_behavior_failure="no failure",
                        turn_behavior_failure_reason="All good",
                    ),
                ],
            ),
            env.ConversationEvaluation(
                conversation_id="conv-uuid-2",
                goal_completion_score=0.3,
                goal_completion_reason="Incomplete",
                turn_success_ratio=0.5,
                overall_agent_score=0.45,
                evaluation_status="Failed",
                turn_scores=[
                    env.TurnEvaluation(
                        turn_id=0,
                        scores=[
                            env.QuantResult(
                                name="helpfulness", value=2.0, reason="Lacks detail"
                            ),
                        ],
                        turn_score=2.0,
                        turn_behavior_failure="lack of specific information",
                        turn_behavior_failure_reason="Did not provide details",
                    ),
                ],
            ),
        ],
        unique_errors=[
            env.UniqueError(
                unique_error_id="uid-1",
                behavior_failure_category="lack of specific information",
                unique_error_description="Agent fails to provide detailed insurance information",
                severity="medium",
                occurrences=[
                    env.Occurrence(conversation_id="conv-uuid-2", turn_id=0),
                ],
            ),
        ],
    )

    chat_id_to_label = {
        "conv-uuid-1": "Conversation 1",
        "conv-uuid-2": "Conversation 2",
    }

    return conversations, evaluation_results, chat_id_to_label


def _render(env: SimpleNamespace) -> str:
    """Generate the report for the standard test data and return its HTML."""
    convos, eval_results, label_map = _build_test_data(env)
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        output_path = f.name
    params = env.HtmlReportParams(
        simulation=convos,
        evaluation=eval_results,
        output_path=output_path,
        chat_id_to_label=label_map,
    )
    return Path(env.generate_html_report(params)).read_text()


def _embedded(html: str, name: str) -> Any:  # noqa: ANN401
    """Parse the JSON embedded in the template as ``name = {...};``."""
    start = html.index(f"{name} = ") + len(f"{name} = ")
    return json.JSONDecoder().raw_decode(html[start:])[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestHtmlReportIntegration:
    """Integration tests for generate_html_report with new evaluation schema."""

    def test_html_contains_chat_id_to_label(
        self, html_report_env: SimpleNamespace
    ) -> None:
        """Generated HTML embeds CHAT_ID_TO_LABEL JS constant."""
        convos, eval_results, label_map = _build_test_data(html_report_env)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        params = html_report_env.HtmlReportParams(
            simulation=convos,
            evaluation=eval_results,
            output_path=output_path,
            chat_id_to_label=label_map,
        )
        result_path = html_report_env.generate_html_report(params)
        html = Path(result_path).read_text()

        assert "CHAT_ID_TO_LABEL" in html
        assert '"conv-uuid-1": "Conversation 1"' in html
        assert '"conv-uuid-2": "Conversation 2"' in html

    def test_html_contains_occurrences_data(
        self, html_report_env: SimpleNamespace
    ) -> None:
        """Unique errors data includes occurrence dict keyed by conversation_id."""
        convos, eval_results, label_map = _build_test_data(html_report_env)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        params = html_report_env.HtmlReportParams(
            simulation=convos,
            evaluation=eval_results,
            output_path=output_path,
            chat_id_to_label=label_map,
        )
        result_path = html_report_env.generate_html_report(params)
        html = Path(result_path).read_text()

        assert "conv-uuid-2" in html
        assert "lack of specific information" in html

    def test_conversation_level_occurrence_has_no_turn(
        self, html_report_env: SimpleNamespace
    ) -> None:
        """A failed-assertion error lists its conversation with no turn or snippet."""
        convos, eval_results, label_map = _build_test_data(html_report_env)
        eval_results.unique_errors.append(
            html_report_env.UniqueError(
                unique_error_id="uid-2",
                behavior_failure_category="assertion failed",
                unique_error_description="Refunded without looking up the order",
                severity="high",
                occurrences=[html_report_env.Occurrence(conversation_id="conv-uuid-1")],
            )
        )

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        params = html_report_env.HtmlReportParams(
            simulation=convos,
            evaluation=eval_results,
            output_path=output_path,
            chat_id_to_label=label_map,
        )
        html = Path(html_report_env.generate_html_report(params)).read_text()

        assert '"conv-uuid-1": []' in html
        assert "turn_None" not in html

    def test_html_contains_chat_label_function(
        self, html_report_env: SimpleNamespace
    ) -> None:
        """Template includes the chatLabel() and openModal() JS helpers."""
        convos, eval_results, label_map = _build_test_data(html_report_env)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        params = html_report_env.HtmlReportParams(
            simulation=convos,
            evaluation=eval_results,
            output_path=output_path,
            chat_id_to_label=label_map,
        )
        result_path = html_report_env.generate_html_report(params)
        html = Path(result_path).read_text()

        assert "function chatLabel(chatId)" in html
        assert "function openModal(" in html

    def test_empty_label_map_produces_valid_html(
        self, html_report_env: SimpleNamespace
    ) -> None:
        """Empty chat_id_to_label still produces valid HTML."""
        convos, eval_results, _ = _build_test_data(html_report_env)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        params = html_report_env.HtmlReportParams(
            simulation=convos,
            evaluation=eval_results,
            output_path=output_path,
            chat_id_to_label={},
        )
        result_path = html_report_env.generate_html_report(params)
        html = Path(result_path).read_text()

        assert "CHAT_ID_TO_LABEL = {}" in html

    def test_no_unique_errors_produces_valid_html(
        self, html_report_env: SimpleNamespace
    ) -> None:
        """Report with no unique errors still generates valid HTML."""
        convos, eval_results, label_map = _build_test_data(html_report_env)
        eval_results.unique_errors = []

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        params = html_report_env.HtmlReportParams(
            simulation=convos,
            evaluation=eval_results,
            output_path=output_path,
            chat_id_to_label=label_map,
        )
        result_path = html_report_env.generate_html_report(params)
        html = Path(result_path).read_text()

        assert "CHAT_ID_TO_LABEL" in html
        assert "chatLabel" in html

    def test_html_has_platform_layout(self, html_report_env: SimpleNamespace) -> None:
        """Overview (Results, Metrics, Failure patterns) and Conversations tabs."""
        html = _render(html_report_env)

        assert 'data-tab="overview"' in html
        assert 'data-tab="conversations"' in html
        for section in ("resultsSection", "metricsSection", "patternsSection"):
            assert f'id="{section}"' in html
        assert "Conversation scoped" in html
        assert "Turn scoped" in html

    def test_metrics_summarised_by_scope(
        self, html_report_env: SimpleNamespace
    ) -> None:
        """Each metric carries its scope, mean, median, band and lowest conversations."""
        report = _embedded(_render(html_report_env), "EMBEDDED_FINAL_REPORT")
        metrics = {m["key"]: m for m in report["metrics"]}

        helpfulness = metrics["helpfulness"]
        assert helpfulness["scope"] == "turn"
        assert helpfulness["value"] == pytest.approx(3.125)
        assert helpfulness["median"] == pytest.approx(3.125)
        assert helpfulness["band"] == "Good"
        assert [lo["chat_id"] for lo in helpfulness["lowest"]] == [
            "conv-uuid-2",
            "conv-uuid-1",
        ]
        assert helpfulness["lowest"][1]["min_turn_id"] == 0

        goal = metrics["goal_completion"]
        assert goal["scope"] == "conversation"
        assert goal["score_max"] == 1.0
        assert goal["value"] == pytest.approx(0.65)

        abf = report["qual_metrics"][0]
        assert abf["key"] == "agent_behavior_failure"
        assert {lb["label"]: lb["count"] for lb in abf["labels"]} == {
            "no failure": 2,
            "lack of specific information": 1,
        }

    def test_conversation_rows_carry_metric_cells(
        self, html_report_env: SimpleNamespace
    ) -> None:
        """Conversations tab rows hold a per-metric mean and the lowest turn."""
        rows = _embedded(_render(html_report_env), "convosData")
        cell = rows[0]["metric_results"]["helpfulness"]
        assert cell["value"] == pytest.approx(4.25)
        assert (cell["min_turn_id"], cell["min_value"]) == (0, 4.0)
        assert rows[0]["turn_count"] == 2
        assert rows[1]["metric_results"]["agent_behavior_failure"]["labels"] == [
            {"value": "lack of specific information", "turn_ids": [0]}
        ]

    def test_failure_patterns_carry_source(
        self, html_report_env: SimpleNamespace
    ) -> None:
        """Patterns embed their source so the report can filter by it."""
        patterns = _embedded(_render(html_report_env), "uniqueErrorsData")
        assert patterns[0]["source"] == "behavior"
        assert "SEVERITY_ORDER" in _render(html_report_env)

    def test_token_usage_appears_in_report_data(
        self, html_report_env: SimpleNamespace
    ) -> None:
        """simulation_token_usage and evaluation_token_usage are embedded in FINAL_REPORT_DATA."""
        convos, eval_results, label_map = _build_test_data(html_report_env)

        convos.usage = html_report_env.TokenUsage(
            total_input_tokens=1234,
            total_output_tokens=567,
        )
        eval_results.usage = html_report_env.TokenUsage(
            total_input_tokens=890,
            total_output_tokens=321,
        )

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        params = html_report_env.HtmlReportParams(
            simulation=convos,
            evaluation=eval_results,
            output_path=output_path,
            chat_id_to_label=label_map,
        )
        result_path = html_report_env.generate_html_report(params)
        html = Path(result_path).read_text()

        assert "simulation_token_usage" in html
        assert "evaluation_token_usage" in html
        assert "1234" in html
        assert "567" in html
        assert "890" in html
        assert "321" in html

    def test_backward_compat_simulation_without_usage(
        self, html_report_env: SimpleNamespace
    ) -> None:
        """Simulation parsed from old JSON (no usage field) defaults usage to None."""
        old_sim_dict = {
            "schema_version": "1.0",
            "simulator_version": "test",
            "simulation_id": "sim-old-1",
            "generated_at": "2024-01-01T00:00:00Z",
            "conversations": [],
        }
        sim = html_report_env.Simulation.model_validate(old_sim_dict)
        assert sim.usage is None

    def test_backward_compat_simulation_null_usage_in_report(
        self, html_report_env: SimpleNamespace
    ) -> None:
        """Report embeds simulation_token_usage: null when simulation has no usage data."""
        convos, eval_results, label_map = _build_test_data(html_report_env)
        # usage is None by default — simulates an old simulation file
        assert convos.usage is None

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        params = html_report_env.HtmlReportParams(
            simulation=convos,
            evaluation=eval_results,
            output_path=output_path,
            chat_id_to_label=label_map,
        )
        result_path = html_report_env.generate_html_report(params)
        html = Path(result_path).read_text()

        assert '"simulation_token_usage": null' in html

    def test_backward_compat_evaluation_without_usage(
        self, html_report_env: SimpleNamespace
    ) -> None:
        """Evaluation parsed from old JSON (no usage field) defaults usage to None."""
        old_eval_dict = {
            "schema_version": "1.0",
            "generated_at": "2024-01-01T00:00:00Z",
            "evaluator_version": "v1",
            "evaluation_id": "eval-old-1",
            "simulation_id": "sim-old-1",
            "conversations": [],
            "unique_errors": [],
        }
        evaluation = html_report_env.Evaluation.model_validate(old_eval_dict)
        assert evaluation.usage is None

    def test_backward_compat_evaluation_null_usage_in_report(
        self, html_report_env: SimpleNamespace
    ) -> None:
        """Report embeds evaluation_token_usage: null when evaluation has no usage data."""
        convos, eval_results, label_map = _build_test_data(html_report_env)
        # usage is None by default — simulates an old evaluation file
        assert eval_results.usage is None

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        params = html_report_env.HtmlReportParams(
            simulation=convos,
            evaluation=eval_results,
            output_path=output_path,
            chat_id_to_label=label_map,
        )
        result_path = html_report_env.generate_html_report(params)
        html = Path(result_path).read_text()

        assert '"evaluation_token_usage": null' in html
