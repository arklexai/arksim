# SPDX-License-Identifier: Apache-2.0
"""Structural e2e tests for the evaluation pipeline using eval_test_simulation.json.

Uses a mock LLM — these tests verify pipeline plumbing (turn counts, IDs,
metric shapes, knowledge forwarding) against a realistic multi-conversation
fixture, not LLM classification correctness.
"""

from __future__ import annotations

import json
import pathlib
from unittest.mock import MagicMock

from arksim.evaluator.entities import EvaluationParams
from arksim.evaluator.evaluator import Evaluator
from arksim.evaluator.utils.prompts import goal_completion_system_prompt
from arksim.evaluator.utils.schema import QualSchema, ScoreSchema, UniqueErrorsSchema
from arksim.simulation_engine.entities import Simulation

_SIM_PATH = (
    pathlib.Path(__file__).parent.parent / "test_data" / "eval_test_simulation.json"
)


def _load_simulation() -> Simulation:
    with open(_SIM_PATH) as f:
        return Simulation.model_validate(json.load(f))


def _mock_llm(score: int = 4, behavior_label: str = "no failure") -> MagicMock:
    llm = MagicMock()

    def _side_effect(
        messages: list, schema: type | None = None, **kw: object
    ) -> object:
        if schema is UniqueErrorsSchema:
            return UniqueErrorsSchema(unique_errors=[])
        if schema is not None and hasattr(schema, "model_fields"):
            if "label" in schema.model_fields:
                return QualSchema(label=behavior_label, reason="mock reason")
            # goal_completion uses 0-1; all other ScoreSchema metrics use 1-5
            is_goal_completion = any(
                m.get("role") == "system"
                and goal_completion_system_prompt.strip() in m.get("content", "")
                for m in messages
            )
            return ScoreSchema(
                score=1 if is_goal_completion else score, reason="mock score"
            )
        return "text response"

    llm.call.side_effect = _side_effect
    return llm


class TestEvalE2EStructure:
    """Pipeline produces correctly shaped Evaluation from eval_test_simulation.json."""

    def test_all_conversations_evaluated(self) -> None:
        sim = _load_simulation()
        result = Evaluator(
            params=EvaluationParams(output_dir="/tmp/eval_e2e", num_workers=1),
            llm=_mock_llm(),
        ).evaluate(sim)
        assert len(result.conversations) == len(sim.conversations)
        assert result.simulation_id == sim.simulation_id

    def test_conversation_ids_round_trip(self) -> None:
        sim = _load_simulation()
        result = Evaluator(
            params=EvaluationParams(output_dir="/tmp/eval_e2e", num_workers=1),
            llm=_mock_llm(),
        ).evaluate(sim)
        expected_ids = {c.conversation_id for c in sim.conversations}
        result_ids = {c.conversation_id for c in result.conversations}
        assert result_ids == expected_ids

    def test_turn_counts_match_conversation_history(self) -> None:
        sim = _load_simulation()
        result = Evaluator(
            params=EvaluationParams(output_dir="/tmp/eval_e2e", num_workers=1),
            llm=_mock_llm(),
        ).evaluate(sim)

        id_to_expected_turns = {}
        for conv in sim.conversations:
            pairs = sum(1 for m in conv.conversation_history if m.role == "assistant")
            id_to_expected_turns[conv.conversation_id] = pairs

        for conv in result.conversations:
            assert (
                len(conv.turn_scores) == id_to_expected_turns[conv.conversation_id]
            ), (
                f"{conv.conversation_id}: expected "
                f"{id_to_expected_turns[conv.conversation_id]} turns, "
                f"got {len(conv.turn_scores)}"
            )

    def test_all_five_builtin_scores_present(self) -> None:
        sim = _load_simulation()
        result = Evaluator(
            params=EvaluationParams(output_dir="/tmp/eval_e2e", num_workers=1),
            llm=_mock_llm(score=4),
        ).evaluate(sim)
        expected = {
            "helpfulness",
            "coherence",
            "verbosity",
            "relevance",
            "faithfulness",
        }
        for conv in result.conversations:
            for turn in conv.turn_scores:
                names = {s.name for s in turn.scores}
                assert names == expected, (
                    f"Turn {turn.turn_id} of {conv.conversation_id} "
                    f"missing metrics: {expected - names}"
                )

    def test_scores_in_valid_range(self) -> None:
        sim = _load_simulation()
        result = Evaluator(
            params=EvaluationParams(output_dir="/tmp/eval_e2e", num_workers=1),
            llm=_mock_llm(score=4),
        ).evaluate(sim)
        for conv in result.conversations:
            assert 0.0 <= conv.goal_completion_score <= 1.0
            assert 0.0 <= conv.overall_agent_score <= 1.0
            assert 0.0 <= conv.turn_success_ratio <= 1.0

    def test_knowledge_forwarded_to_llm(self) -> None:
        sim = _load_simulation()
        conv_with_knowledge = next(
            c
            for c in sim.conversations
            if c.simulated_user_prompt.variables.get("scenario.knowledge")
        )
        snippet = conv_with_knowledge.simulated_user_prompt.variables[
            "scenario.knowledge"
        ][0][:30]

        llm = _mock_llm(score=4)
        single = Simulation(
            schema_version=sim.schema_version,
            simulator_version=sim.simulator_version,
            simulation_id=sim.simulation_id,
            conversations=[conv_with_knowledge],
        )
        Evaluator(
            params=EvaluationParams(output_dir="/tmp/eval_e2e", num_workers=1),
            llm=llm,
        ).evaluate(single)

        all_user_msgs = [
            m.get("content", "")
            for call in llm.call.call_args_list
            for m in call[0][0]
            if m.get("role") == "user"
        ]
        assert any(snippet in msg for msg in all_user_msgs), (
            f"Knowledge snippet '{snippet}' was not forwarded to LLM"
        )
