# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from arksim.evaluator.entities import Occurrence, UniqueError
from arksim.evaluator.focus import build_error_scenario_map


def _make_error(error_id: str, occurrences: list[tuple[str, int]]) -> UniqueError:
    return UniqueError(
        unique_error_id=error_id,
        behavior_failure_category="false information",
        unique_error_description=f"Error {error_id}",
        severity="high",
        occurrences=[
            Occurrence(conversation_id=cid, turn_id=tid) for cid, tid in occurrences
        ],
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
