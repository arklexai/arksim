# SPDX-License-Identifier: Apache-2.0
"""End-to-end evaluation tests using eval_test_simulation.json.

All scenarios live in tests/test_data/eval_test_simulation.json, which is a
standard Simulation format (schema_version v1.1). Each conversation carries a
"test.expected_label" variable that records the correct agent behavior failure
label for that scenario.

Two test classes:

  TestEvalE2EStructure
      Mock LLM, always run. Verifies pipeline plumbing: turn counts, IDs,
      metric shapes, knowledge forwarding, and file output. Mock is correct
      here because these tests are about the pipeline, not LLM classification.

  TestEvalE2ELabelClassification
      Real LLM, @integration, skipped without OPENAI_API_KEY. Verifies that
      the prompt actually guides the LLM to produce the labels in
      "test.expected_label". Mock-LLM classification tests are circular — you
      assert what you told the mock to return — so only a real LLM validates
      prompt correctness.
"""

from __future__ import annotations

import json
import os
import pathlib
from unittest.mock import MagicMock

import pytest

from arksim.evaluator.entities import EvaluationParams
from arksim.evaluator.evaluator import Evaluator
from arksim.evaluator.utils.prompts import goal_completion_system_prompt
from arksim.evaluator.utils.schema import QualSchema, ScoreSchema, UniqueErrorsSchema
from arksim.simulation_engine.entities import Simulation

_SIM_PATH = pathlib.Path(__file__).parent.parent / "test_data" / "eval_test_simulation.json"

_requires_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping real-LLM classification tests",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_simulation() -> Simulation:
    with open(_SIM_PATH) as f:
        return Simulation.model_validate(json.load(f))


def _expected_labels() -> dict[str, str]:
    """Return {conversation_id: expected_label} read from the raw JSON.

    ``test.expected_label`` is stored on assistant messages in
    ``conversation_history``, not in ``simulated_user_prompt.variables``, so
    we read the raw JSON rather than the Pydantic-parsed model (which drops
    unknown fields on Message).

    A conversation is considered 'no failure' only when every assistant turn
    carries that label.  The first non-'no failure' label wins otherwise.
    """
    result: dict[str, str] = {}
    raw = json.loads(_SIM_PATH.read_text())
    for conv in raw["conversations"]:
        labels = [
            msg["test.expected_label"]
            for msg in conv["conversation_history"]
            if msg.get("role") == "assistant" and "test.expected_label" in msg
        ]
        non_pass = [lbl for lbl in labels if lbl != "no failure"]
        result[conv["conversation_id"]] = non_pass[0] if non_pass else "no failure"
    return result


def _mock_llm(score: int = 4, behavior_label: str = "no failure") -> MagicMock:
    llm = MagicMock()

    def _side_effect(messages: list, schema: type | None = None, **kw: object) -> object:
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
            return ScoreSchema(score=1 if is_goal_completion else score, reason="mock score")
        return "text response"

    llm.call.side_effect = _side_effect
    return llm


def _real_llm() -> object:
    from arksim.llms.chat import LLM
    return LLM(model="gpt-4.1", provider="openai")


# ---------------------------------------------------------------------------
# Structural tests — mock LLM, always run
# ---------------------------------------------------------------------------


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
            pairs = sum(
                1 for m in conv.conversation_history if m.role == "assistant"
            )
            id_to_expected_turns[conv.conversation_id] = pairs

        for conv in result.conversations:
            assert len(conv.turn_scores) == id_to_expected_turns[conv.conversation_id], (
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
        expected = {"helpfulness", "coherence", "verbosity", "relevance", "faithfulness"}
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
        # Pick a conversation that has non-empty knowledge
        conv_with_knowledge = next(
            c for c in sim.conversations
            if c.simulated_user_prompt.variables.get("scenario.knowledge")
        )
        snippet = conv_with_knowledge.simulated_user_prompt.variables["scenario.knowledge"][0][:30]

        llm = _mock_llm(score=4)
        # Run only the single conversation to keep the check focused
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

    def test_results_can_be_saved(self, temp_dir: str) -> None:
        sim = _load_simulation()
        ev = Evaluator(
            params=EvaluationParams(output_dir=temp_dir, num_workers=1),
            llm=_mock_llm(),
        )
        ev.evaluate(sim)
        ev.save_results()
        assert os.path.exists(os.path.join(temp_dir, "evaluation.json"))


# ---------------------------------------------------------------------------
# Classification tests — real LLM, skipped without OPENAI_API_KEY
# ---------------------------------------------------------------------------


@_requires_openai
@pytest.mark.integration
class TestEvalE2ELabelClassification:
    """Verifies the prompt guides the real LLM to produce the expected labels.

    Each conversation in eval_test_simulation.json has a "test.expected_label"
    variable. Tests are parametrized directly from the simulation file, so
    adding a new scenario to the JSON automatically adds a new test case.
    """

    @staticmethod
    def _run(conv_id: str) -> tuple[str, list[str]]:
        """Run evaluation on a single conversation, return (expected_label, actual_labels)."""
        sim = _load_simulation()
        conv = next(c for c in sim.conversations if c.conversation_id == conv_id)
        expected_label = _expected_labels()[conv_id]

        single = Simulation(
            schema_version=sim.schema_version,
            simulator_version=sim.simulator_version,
            simulation_id=sim.simulation_id,
            conversations=[conv],
        )
        result = Evaluator(
            params=EvaluationParams(output_dir="/tmp/eval_e2e_real", num_workers=1),
            llm=_real_llm(),
        ).evaluate(single)

        actual_labels = [t.turn_behavior_failure for t in result.conversations[0].turn_scores]
        return expected_label, actual_labels

    # Build parametrize list at collection time from the simulation file
    _sim_data = json.loads(_SIM_PATH.read_text())
    _conv_ids = [c["conversation_id"] for c in _sim_data["conversations"]]

    @pytest.mark.parametrize("conv_id", _conv_ids)
    def test_conversation_matches_expected_label(self, conv_id: str) -> None:
        """For each conversation, at least one turn must match the expected label,
        or all turns must be 'no failure' when the expected label is 'no failure'."""
        sim = _load_simulation()
        conv = next(c for c in sim.conversations if c.conversation_id == conv_id)
        description = conv.simulated_user_prompt.variables.get("test.description", "")
        expected_label, actual_labels = self._run(conv_id)

        if expected_label == "no failure":
            for label in actual_labels:
                assert label != "false information", (
                    f"[{conv_id}] {description}\n"
                    f"No contradiction exists — should not be 'false information'. "
                    f"Got {actual_labels}"
                )
        else:
            assert expected_label in actual_labels, (
                f"[{conv_id}] {description}\n"
                f"Expected '{expected_label}' in turn labels, got {actual_labels}"
            )
