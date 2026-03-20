# SPDX-License-Identifier: Apache-2.0
"""Tests for Evaluator.evaluate(), save_results(), display_evaluation_summary()."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from arksim.evaluator.entities import EvaluationParams
from arksim.evaluator.evaluator import Evaluator
from arksim.evaluator.utils.schema import QualSchema, ScoreSchema, UniqueErrorsSchema
from arksim.simulation_engine.entities import (
    Conversation,
    Message,
    SimulatedUserPrompt,
    Simulation,
)


def _mock_llm() -> MagicMock:
    """LLM that returns score 4 for quant metrics and empty unique errors."""
    llm = MagicMock()

    def _call_side_effect(
        messages: list, schema: type | None = None, **kw: object
    ) -> object:
        if schema is UniqueErrorsSchema:
            return UniqueErrorsSchema(unique_errors=[])
        if schema is not None and hasattr(schema, "model_fields"):
            if "label" in schema.model_fields:
                return QualSchema(label="no failure", reason="fine")
            return ScoreSchema(score=4, reason="good")
        return "text response"

    llm.call.side_effect = _call_side_effect
    return llm


def _simulation() -> Simulation:
    return Simulation(
        schema_version="v1.1",
        simulator_version="v1",
        simulation_id="sim-1",
        conversations=[
            Conversation(
                conversation_id="conv-1",
                scenario_id="sc-1",
                conversation_history=[
                    Message(turn_id=0, role="simulated_user", content="Hi"),
                    Message(turn_id=0, role="assistant", content="Hello!"),
                ],
                simulated_user_prompt=SimulatedUserPrompt(
                    simulated_user_prompt_template="tmpl",
                    variables={
                        "scenario.goal": "help",
                        "scenario.knowledge": [],
                    },
                ),
            )
        ],
    )


class TestEvaluatorEvaluate:
    def test_evaluate_returns_evaluation(self) -> None:
        llm = _mock_llm()
        params = EvaluationParams(output_dir="/tmp/eval_test", num_workers=1)
        ev = Evaluator(params=params, llm=llm)
        result = ev.evaluate(_simulation())
        assert result is not None
        assert len(result.conversations) == 1
        assert result.simulation_id == "sim-1"
        assert ev.total_conversations == 1
        assert ev.total_turns >= 1

    def test_evaluate_auto_workers(self) -> None:
        llm = _mock_llm()
        params = EvaluationParams(output_dir="/tmp/eval_test", num_workers="auto")
        ev = Evaluator(params=params, llm=llm)
        result = ev.evaluate(_simulation())
        assert len(result.conversations) == 1


class TestEvaluatorSaveResults:
    def test_save_raises_without_evaluate(self) -> None:
        params = EvaluationParams(output_dir="/tmp/eval_test", num_workers=1)
        ev = Evaluator(params=params, llm=None)
        with pytest.raises(ValueError, match="No evaluation results"):
            ev.save_results()

    def test_save_after_evaluate(self, temp_dir: str) -> None:
        llm = _mock_llm()
        params = EvaluationParams(output_dir=temp_dir, num_workers=1)
        ev = Evaluator(params=params, llm=llm)
        ev.evaluate(_simulation())
        ev.save_results()
        assert os.path.exists(os.path.join(temp_dir, "evaluation.json"))


class TestDisplayEvaluationSummary:
    def test_display_without_results(self) -> None:
        params = EvaluationParams(output_dir="/tmp/eval_test", num_workers=1)
        ev = Evaluator(params=params, llm=None)
        ev.total_conversations = 0
        ev.total_turns = 0
        ev.display_evaluation_summary()  # should not crash

    def test_display_with_results(self) -> None:
        llm = _mock_llm()
        params = EvaluationParams(output_dir="/tmp/eval_test", num_workers=1)
        ev = Evaluator(params=params, llm=llm)
        ev.evaluate(_simulation())
        ev.display_evaluation_summary()  # should not crash
