# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os

import pytest

from arksim.evaluator.entities import Occurrence, UniqueError
from arksim.evaluator.focus import (
    build_error_scenario_map,
    generate_focus_files,
)
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
        result = build_error_scenario_map(errors, conv_to_scenario)
        assert result["err_1"] == {"scenario_refund"}
        assert result["err_2"] == {"scenario_clarify"}

    def test_unknown_conversation_id_skipped(self) -> None:
        conv_to_scenario = {"conv_1": "scenario_a"}
        errors = [_make_error("err_1", [("conv_1", 0), ("conv_unknown", 1)])]
        result = build_error_scenario_map(errors, conv_to_scenario)
        assert result["err_1"] == {"scenario_a"}

    def test_empty_errors_returns_empty(self) -> None:
        result = build_error_scenario_map([], {"conv_1": "scenario_a"})
        assert result == {}

    def test_all_unknown_conversation_ids_drops_key(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        errors = [_make_error("err_1", [("conv_unknown", 0)])]
        with caplog.at_level("WARNING", logger="arksim.evaluator.focus"):
            result = build_error_scenario_map(errors, {})
        assert "err_1" not in result
        assert any("err_1" in record.message for record in caplog.records)


class TestGenerateFocusFiles:
    def test_writes_per_error_and_all_files(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        scenario_a = _make_scenario("scenario_a")
        scenario_b = _make_scenario("scenario_b")
        scenarios = Scenarios(schema_version="1.0", scenarios=[scenario_a, scenario_b])

        # err_critical has 2 occurrences (higher count), severity=critical
        # err_medium has 1 occurrence, severity=medium
        # Sorting: critical first, then medium; within same severity, desc by count
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

        result = generate_focus_files(
            unique_errors=errors,
            conv_to_scenario=conv_to_scenario,
            scenarios=scenarios,
            output_dir=str(tmp_path),
        )

        assert len(result) == 2

        # Verify sorted order: critical first (error_1), medium second (error_2)
        assert result[0].error_index == 1
        assert result[0].unique_error_id == "err_critical"
        assert result[0].severity == "critical"
        assert set(result[0].scenario_ids) == {"scenario_a"}

        assert result[1].error_index == 2
        assert result[1].unique_error_id == "err_medium"
        assert result[1].severity == "medium"
        assert set(result[1].scenario_ids) == {"scenario_b"}

        # Verify file paths in FocusFileInfo
        focus_dir = os.path.join(str(tmp_path), "focus")
        assert result[0].file_path == os.path.join(focus_dir, "error_1.json")
        assert result[1].file_path == os.path.join(focus_dir, "error_2.json")

        # Verify error_1.json content (critical error -> scenario_a)
        with open(result[0].file_path) as f:
            data_1 = json.load(f)
        assert data_1["schema_version"] == "1.0"
        assert len(data_1["scenarios"]) == 1
        assert data_1["scenarios"][0]["scenario_id"] == "scenario_a"

        # Verify error_2.json content (medium error -> scenario_b)
        with open(result[1].file_path) as f:
            data_2 = json.load(f)
        assert data_2["schema_version"] == "1.0"
        assert len(data_2["scenarios"]) == 1
        assert data_2["scenarios"][0]["scenario_id"] == "scenario_b"

        # Verify all_failures.json contains the union of all scenarios
        all_path = os.path.join(focus_dir, "all_failures.json")
        assert os.path.exists(all_path)
        with open(all_path) as f:
            all_data = json.load(f)
        all_ids = {s["scenario_id"] for s in all_data["scenarios"]}
        assert all_ids == {"scenario_a", "scenario_b"}

    def test_no_errors_returns_empty(self, tmp_path: pytest.TempPathFactory) -> None:
        scenarios = Scenarios(
            schema_version="1.0", scenarios=[_make_scenario("scenario_a")]
        )
        result = generate_focus_files(
            unique_errors=[],
            conv_to_scenario={},
            scenarios=scenarios,
            output_dir=str(tmp_path),
        )

        assert result == []
        focus_dir = os.path.join(str(tmp_path), "focus")
        assert not os.path.exists(focus_dir)

    def test_missing_scenario_gracefully_skipped(
        self, tmp_path: pytest.TempPathFactory, caplog: pytest.LogCaptureFixture
    ) -> None:
        # scenario_ghost does not exist in the Scenarios object
        scenarios = Scenarios(
            schema_version="1.0", scenarios=[_make_scenario("scenario_real")]
        )
        errors = [
            _make_error("err_ghost", [("conv_ghost", 0)], severity="high"),
        ]
        conv_to_scenario = {"conv_ghost": "scenario_ghost"}

        with caplog.at_level("DEBUG", logger="arksim.evaluator.focus"):
            result = generate_focus_files(
                unique_errors=errors,
                conv_to_scenario=conv_to_scenario,
                scenarios=scenarios,
                output_dir=str(tmp_path),
            )

        assert result == []
        focus_dir = os.path.join(str(tmp_path), "focus")
        assert not os.path.exists(focus_dir)
