# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from arksim.evaluator.entities import (
    ErrorScenarioMapping,
    EvaluationParams,
    Occurrence,
    UniqueError,
)
from arksim.evaluator.error_scenarios import (
    _build_error_scenario_map,
    _sort_key,
    build_error_scenario_data,
)
from arksim.evaluator.evaluator import Evaluator
from arksim.scenario.entities import Scenario, Scenarios


def _make_error(
    error_id: str,
    occurrences: list[tuple[str, int]],
    severity: str = "high",
) -> UniqueError:
    return UniqueError(
        unique_error_id=error_id,
        behavior_failure_category="false information",
        unique_error_description=f"Error {error_id}",
        severity=severity,
        occurrences=[
            Occurrence(conversation_id=cid, turn_id=tid) for cid, tid in occurrences
        ],
    )


def _make_scenario(scenario_id: str) -> Scenario:
    return Scenario(
        scenario_id=scenario_id,
        user_id="user_1",
        goal="Test goal",
        agent_context="Test context",
        user_profile="Test profile",
    )


class TestSortKey:
    def test_unknown_severity_sorts_after_low(self) -> None:
        err_low = _make_error("err_low", [("c1", 0)], severity="low")
        err_unknown = _make_error("err_unknown", [("c1", 0)], severity="catastrophic")

        assert _sort_key(err_low) < _sort_key(err_unknown)

    def test_same_severity_sorts_by_descending_occurrence_count(self) -> None:
        err_few = _make_error("err_few", [("c1", 0)], severity="high")
        err_many = _make_error(
            "err_many", [("c1", 0), ("c2", 1), ("c3", 2)], severity="high"
        )

        assert _sort_key(err_many) < _sort_key(err_few)


class TestBuildErrorScenarioMap:
    def test_maps_errors_to_scenarios(self) -> None:
        conv_to_scenario = {
            "conv_1": "scenario_refund",
            "conv_2": "scenario_clarify",
            "conv_3": "scenario_refund",
        }
        errors = [
            _make_error("err_1", [("conv_1", 0), ("conv_3", 1)]),
            _make_error("err_2", [("conv_2", 0)]),
        ]
        result = _build_error_scenario_map(errors, conv_to_scenario)
        assert result["err_1"] == {"scenario_refund"}
        assert result["err_2"] == {"scenario_clarify"}

    def test_unknown_conversation_id_skipped(self) -> None:
        conv_to_scenario = {"conv_1": "scenario_a"}
        errors = [_make_error("err_1", [("conv_1", 0), ("conv_unknown", 1)])]
        result = _build_error_scenario_map(errors, conv_to_scenario)
        assert result["err_1"] == {"scenario_a"}

    def test_empty_errors_returns_empty(self) -> None:
        result = _build_error_scenario_map([], {"conv_1": "scenario_a"})
        assert result == {}

    def test_all_unknown_conversation_ids_drops_key(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        errors = [_make_error("err_1", [("conv_unknown", 0)])]
        with caplog.at_level("WARNING", logger="arksim.evaluator.error_scenarios"):
            result = _build_error_scenario_map(errors, {})
        assert "err_1" not in result
        assert any("err_1" in record.message for record in caplog.records)

    def test_error_with_empty_occurrences_dropped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        error = UniqueError(
            unique_error_id="err_empty",
            behavior_failure_category="false information",
            unique_error_description="Error with no occurrences",
            severity="high",
            occurrences=[],
        )
        with caplog.at_level("WARNING", logger="arksim.evaluator.error_scenarios"):
            result = _build_error_scenario_map([error], {"conv_1": "scenario_a"})
        assert "err_empty" not in result
        assert any("err_empty" in record.message for record in caplog.records)


class TestBuildErrorScenarioData:
    def test_computes_mappings_sorted_by_severity(self) -> None:
        scenarios = Scenarios(
            schema_version="1.0",
            scenarios=[_make_scenario("scenario_a"), _make_scenario("scenario_b")],
        )
        errors = [
            _make_error("err_medium", [("conv_2", 0)], severity="medium"),
            _make_error(
                "err_critical", [("conv_1", 0), ("conv_3", 1)], severity="critical"
            ),
        ]
        conv_to_scenario = {
            "conv_1": "scenario_a",
            "conv_2": "scenario_b",
            "conv_3": "scenario_a",
        }

        mappings, all_scenarios = build_error_scenario_data(
            unique_errors=errors,
            conv_to_scenario=conv_to_scenario,
            scenarios=scenarios,
        )

        assert len(mappings) == 2
        assert mappings[0].error_index == 1
        assert mappings[0].unique_error_id == "err_critical"
        assert mappings[0].severity == "critical"
        assert mappings[0].scenario_ids == ["scenario_a"]

        assert mappings[1].error_index == 2
        assert mappings[1].unique_error_id == "err_medium"
        assert mappings[1].severity == "medium"
        assert mappings[1].scenario_ids == ["scenario_b"]

        # All scenarios union
        all_ids = [s.scenario_id for s in all_scenarios]
        assert all_ids == ["scenario_a", "scenario_b"]

    def test_no_errors_returns_empty(self) -> None:
        scenarios = Scenarios(
            schema_version="1.0", scenarios=[_make_scenario("scenario_a")]
        )
        mappings, all_scenarios = build_error_scenario_data(
            unique_errors=[],
            conv_to_scenario={},
            scenarios=scenarios,
        )

        assert mappings == []
        assert all_scenarios == []

    def test_missing_scenario_gracefully_skipped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        scenarios = Scenarios(
            schema_version="1.0", scenarios=[_make_scenario("scenario_real")]
        )
        errors = [_make_error("err_ghost", [("conv_ghost", 0)], severity="high")]
        conv_to_scenario = {"conv_ghost": "scenario_ghost"}

        with caplog.at_level("DEBUG", logger="arksim.evaluator.error_scenarios"):
            mappings, all_scenarios = build_error_scenario_data(
                unique_errors=errors,
                conv_to_scenario=conv_to_scenario,
                scenarios=scenarios,
            )

        assert mappings == []
        assert all_scenarios == []

    def test_overlapping_scenarios_deduplicated(self) -> None:
        scenarios = Scenarios(
            schema_version="1.0",
            scenarios=[
                _make_scenario("scenario_a"),
                _make_scenario("scenario_b"),
                _make_scenario("scenario_c"),
            ],
        )
        errors = [
            _make_error("err_1", [("conv_1", 0), ("conv_2", 0)], severity="high"),
            _make_error("err_2", [("conv_2", 0), ("conv_3", 0)], severity="high"),
        ]
        conv_to_scenario = {
            "conv_1": "scenario_a",
            "conv_2": "scenario_b",
            "conv_3": "scenario_c",
        }

        mappings, all_scenarios = build_error_scenario_data(
            unique_errors=errors,
            conv_to_scenario=conv_to_scenario,
            scenarios=scenarios,
        )

        assert len(mappings) == 2
        all_ids = [s.scenario_id for s in all_scenarios]
        assert all_ids == ["scenario_a", "scenario_b", "scenario_c"]

    def test_single_error_single_scenario(self) -> None:
        scenarios = Scenarios(
            schema_version="1.0", scenarios=[_make_scenario("scenario_a")]
        )
        errors = [_make_error("err_1", [("conv_1", 0)], severity="critical")]
        conv_to_scenario = {"conv_1": "scenario_a"}

        mappings, all_scenarios = build_error_scenario_data(
            unique_errors=errors,
            conv_to_scenario=conv_to_scenario,
            scenarios=scenarios,
        )

        assert len(mappings) == 1
        assert mappings[0].error_index == 1
        assert mappings[0].scenario_ids == ["scenario_a"]
        assert len(all_scenarios) == 1

    def test_all_errors_unmapped_returns_empty(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """All errors map to scenario IDs that don't exist in the Scenarios object."""
        scenarios = Scenarios(
            schema_version="1.0", scenarios=[_make_scenario("scenario_real")]
        )
        errors = [
            _make_error("err_1", [("conv_1", 0)], severity="high"),
            _make_error("err_2", [("conv_2", 0)], severity="medium"),
        ]
        conv_to_scenario = {"conv_1": "scenario_ghost_a", "conv_2": "scenario_ghost_b"}

        with caplog.at_level("DEBUG", logger="arksim.evaluator.error_scenarios"):
            mappings, all_scenarios = build_error_scenario_data(
                unique_errors=errors,
                conv_to_scenario=conv_to_scenario,
                scenarios=scenarios,
            )

        assert mappings == []
        assert all_scenarios == []


class TestWriteFocusFiles:
    """Test that Evaluator.save_results writes focus files from error_scenario_mappings."""

    def test_save_results_writes_focus_files(self, tmp_path: Path) -> None:
        from arksim.evaluator.entities import (
            Evaluation,
        )

        params = EvaluationParams(output_dir=str(tmp_path))
        scenarios = Scenarios(
            schema_version="1.0",
            scenarios=[_make_scenario("scenario_a"), _make_scenario("scenario_b")],
        )
        evaluator = Evaluator(params=params, scenarios=scenarios)

        evaluator.evaluation_results = Evaluation(
            schema_version="v1.1",
            generated_at="2026-04-01T00:00:00Z",
            evaluator_version="v1",
            evaluation_id="eval-1",
            simulation_id="sim-1",
            conversations=[],
            unique_errors=[],
            error_scenario_mappings=[
                ErrorScenarioMapping(
                    error_index=1,
                    unique_error_id="err_1",
                    error_description="Wrong refund window",
                    severity="critical",
                    scenario_ids=["scenario_a"],
                ),
                ErrorScenarioMapping(
                    error_index=2,
                    unique_error_id="err_2",
                    error_description="Missed escalation",
                    severity="high",
                    scenario_ids=["scenario_b"],
                ),
            ],
        )

        evaluator.save_results()

        # evaluation.json written
        eval_path = os.path.join(str(tmp_path), "evaluation.json")
        assert os.path.exists(eval_path)

        # Focus files written
        focus_dir = os.path.join(str(tmp_path), "focus")
        assert os.path.exists(os.path.join(focus_dir, "error_1.json"))
        assert os.path.exists(os.path.join(focus_dir, "error_2.json"))
        assert os.path.exists(os.path.join(focus_dir, "all_failures.json"))

        # Verify error_1.json content
        with open(os.path.join(focus_dir, "error_1.json")) as f:
            data = json.load(f)
        assert data["schema_version"] == "1.0"
        assert len(data["scenarios"]) == 1
        assert data["scenarios"][0]["scenario_id"] == "scenario_a"

        # Verify all_failures.json has union
        with open(os.path.join(focus_dir, "all_failures.json")) as f:
            all_data = json.load(f)
        all_ids = {s["scenario_id"] for s in all_data["scenarios"]}
        assert all_ids == {"scenario_a", "scenario_b"}

    def test_save_results_no_focus_files_when_no_mappings(self, tmp_path: Path) -> None:
        from arksim.evaluator.entities import Evaluation

        params = EvaluationParams(output_dir=str(tmp_path))
        evaluator = Evaluator(params=params)

        evaluator.evaluation_results = Evaluation(
            schema_version="v1.1",
            generated_at="2026-04-01T00:00:00Z",
            evaluator_version="v1",
            evaluation_id="eval-1",
            simulation_id="sim-1",
            conversations=[],
            unique_errors=[],
        )

        evaluator.save_results()

        assert os.path.exists(os.path.join(str(tmp_path), "evaluation.json"))
        assert not os.path.exists(os.path.join(str(tmp_path), "focus"))

    def test_focus_file_io_failure_does_not_crash_save(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from unittest.mock import patch

        from arksim.evaluator.entities import Evaluation

        params = EvaluationParams(output_dir=str(tmp_path))
        scenarios = Scenarios(
            schema_version="1.0", scenarios=[_make_scenario("scenario_a")]
        )
        evaluator = Evaluator(params=params, scenarios=scenarios)

        evaluator.evaluation_results = Evaluation(
            schema_version="v1.1",
            generated_at="2026-04-01T00:00:00Z",
            evaluator_version="v1",
            evaluation_id="eval-1",
            simulation_id="sim-1",
            conversations=[],
            unique_errors=[],
            error_scenario_mappings=[
                ErrorScenarioMapping(
                    error_index=1,
                    unique_error_id="err_1",
                    error_description="Error",
                    severity="high",
                    scenario_ids=["scenario_a"],
                ),
            ],
        )

        # Patch _write_focus_files to raise after evaluation.json is saved
        with (
            patch.object(
                Evaluator, "_write_focus_files", side_effect=OSError("disk full")
            ),
            caplog.at_level(logging.ERROR),
        ):
            evaluator.save_results()

        # evaluation.json was still saved
        assert os.path.exists(os.path.join(str(tmp_path), "evaluation.json"))
        assert any(
            "Focus file writing failed" in record.message for record in caplog.records
        )


class TestDisplayTopUniqueErrorsWithMappings:
    def test_displays_scenario_ids_and_focus_path(
        self, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        from arksim.evaluator.entities import Evaluation

        params = EvaluationParams(output_dir=str(tmp_path))
        evaluator = Evaluator(params=params)
        evaluator.chat_id_to_label = {
            "conv_1": "Conversation 1",
            "conv_2": "Conversation 2",
        }

        errors = [_make_error("err_1", [("conv_1", 0), ("conv_2", 1)])]
        evaluator.evaluation_results = Evaluation(
            schema_version="v1.1",
            generated_at="2026-04-01T00:00:00Z",
            evaluator_version="v1",
            evaluation_id="eval-1",
            simulation_id="sim-1",
            conversations=[],
            unique_errors=errors,
            error_scenario_mappings=[
                ErrorScenarioMapping(
                    error_index=1,
                    unique_error_id="err_1",
                    error_description="Error err_1",
                    severity="high",
                    scenario_ids=["scenario_clarify", "scenario_refund"],
                ),
            ],
        )

        with caplog.at_level(logging.INFO):
            evaluator._display_top_unique_errors(errors)

        log_text = caplog.text
        assert "scenario_clarify" in log_text
        assert "scenario_refund" in log_text
        assert "focus/error_1.json" in log_text

    def test_no_mapping_for_displayed_error(
        self, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        """Error is displayed but has no matching mapping (e.g. all convs unknown)."""
        from arksim.evaluator.entities import Evaluation

        params = EvaluationParams(output_dir=str(tmp_path))
        evaluator = Evaluator(params=params)
        evaluator.chat_id_to_label = {"conv_1": "Conversation 1"}

        errors = [
            _make_error("err_1", [("conv_1", 0)]),
            _make_error("err_2", [("conv_1", 1)]),
        ]
        # Only err_1 has a mapping; err_2 was dropped during compute
        evaluator.evaluation_results = Evaluation(
            schema_version="v1.1",
            generated_at="2026-04-01T00:00:00Z",
            evaluator_version="v1",
            evaluation_id="eval-1",
            simulation_id="sim-1",
            conversations=[],
            unique_errors=errors,
            error_scenario_mappings=[
                ErrorScenarioMapping(
                    error_index=1,
                    unique_error_id="err_1",
                    error_description="Error err_1",
                    severity="high",
                    scenario_ids=["scenario_a"],
                ),
            ],
        )

        with caplog.at_level(logging.INFO):
            evaluator._display_top_unique_errors(errors)

        log_text = caplog.text
        assert "error_1.json" in log_text
        assert "error_2.json" not in log_text

    def test_display_summary_shows_rerun_hint(
        self, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        from arksim.evaluator.entities import (
            ConversationEvaluation,
            Evaluation,
            TurnEvaluation,
        )

        params = EvaluationParams(output_dir=str(tmp_path))
        evaluator = Evaluator(params=params)
        evaluator.chat_id_to_label = {"conv_1": "Conversation 1"}

        evaluator.evaluation_results = Evaluation(
            schema_version="v1.1",
            generated_at="2026-04-01T00:00:00Z",
            evaluator_version="v1",
            evaluation_id="eval-1",
            simulation_id="sim-1",
            conversations=[
                ConversationEvaluation(
                    conversation_id="conv_1",
                    goal_completion_score=0.5,
                    goal_completion_reason="Partial",
                    turn_success_ratio=0.5,
                    overall_agent_score=0.5,
                    evaluation_status="completed",
                    turn_scores=[
                        TurnEvaluation(
                            turn_id=0,
                            scores=[],
                            turn_score=3.0,
                            turn_behavior_failure="false_information",
                            turn_behavior_failure_reason="Wrong info",
                            unique_error_ids=["err_1"],
                        ),
                    ],
                ),
            ],
            unique_errors=[_make_error("err_1", [("conv_1", 0)])],
            error_scenario_mappings=[
                ErrorScenarioMapping(
                    error_index=1,
                    unique_error_id="err_1",
                    error_description="Error err_1",
                    severity="high",
                    scenario_ids=["scenario_a"],
                ),
            ],
        )
        evaluator.total_conversations = 1
        evaluator.total_turns = 1

        with caplog.at_level(logging.INFO):
            evaluator.display_evaluation_summary()

        log_text = caplog.text
        assert "FOCUS FILES FOR TARGETED RERUNS" in log_text
        assert "all_failures.json" in log_text


class TestErrorScenarioMappingSerialization:
    def test_error_scenario_mappings_included_in_model_dump(self) -> None:
        from arksim.evaluator.entities import Evaluation

        evaluation = Evaluation(
            schema_version="v1.1",
            generated_at="2026-04-01T00:00:00Z",
            evaluator_version="v1",
            evaluation_id="eval-1",
            simulation_id="sim-1",
            conversations=[],
            unique_errors=[],
            error_scenario_mappings=[
                ErrorScenarioMapping(
                    error_index=1,
                    unique_error_id="err_1",
                    error_description="Agent gave wrong refund amount",
                    severity="critical",
                    scenario_ids=["scenario_refund"],
                ),
            ],
        )

        data = evaluation.model_dump()
        assert len(data["error_scenario_mappings"]) == 1
        mapping = data["error_scenario_mappings"][0]
        assert mapping["unique_error_id"] == "err_1"
        assert mapping["scenario_ids"] == ["scenario_refund"]
        assert "file_path" not in mapping

    def test_error_scenario_mappings_defaults_to_empty(self) -> None:
        from arksim.evaluator.entities import Evaluation

        evaluation = Evaluation(
            schema_version="v1.1",
            generated_at="2026-04-01T00:00:00Z",
            evaluator_version="v1",
            evaluation_id="eval-1",
            simulation_id="sim-1",
            conversations=[],
            unique_errors=[],
        )

        data = evaluation.model_dump()
        assert data["error_scenario_mappings"] == []
