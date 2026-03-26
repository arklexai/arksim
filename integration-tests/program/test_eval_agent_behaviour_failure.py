# SPDX-License-Identifier: Apache-2.0
"""Integration tests verifying the evaluator correctly detects agent behaviour failures.

Each conversation in eval_test_simulation.json has a ``test.expected_behavior_failure``
field on each assistant message. Tests are parametrized directly from the
simulation file, so adding a new scenario to the JSON automatically adds a
new test case.

Mock-LLM classification tests are circular (you assert what you told the mock
to return), so only a real LLM validates prompt correctness.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from arksim.evaluator.entities import EvaluationParams
from arksim.evaluator.evaluator import Evaluator
from arksim.simulation_engine.entities import Simulation

from .conftest import OPENAI_MODEL, requires_openai

pytestmark = [pytest.mark.integration, requires_openai]

_SIM_PATH = (
    pathlib.Path(__file__).parent.parent / "test_data" / "eval_test_simulation.json"
)

# Labels that indicate a real failure — skipped/system outcomes are excluded
_FAILURE_LABELS = {
    "false information",
    "lack of specific information",
    "disobey user request",
    "failure to ask for clarification",
    "repetition",
}


def _load_simulation() -> Simulation:
    with open(_SIM_PATH) as f:
        return Simulation.model_validate(json.load(f))


def _expected_labels() -> dict[str, str]:
    """Return {conversation_id: expected_behavior_failure} read from the raw JSON.

    ``test.expected_behavior_failure`` is stored on assistant messages in
    ``conversation_history``, not in ``simulated_user_prompt.variables``, so
    we read the raw JSON rather than the Pydantic-parsed model (which drops
    unknown fields on Message).

    A conversation is 'no failure' only when every assistant turn carries that
    label. The first non-'no failure' label wins otherwise.
    """
    result: dict[str, str] = {}
    raw = json.loads(_SIM_PATH.read_text())
    for conv in raw["conversations"]:
        labels = [
            msg["test.expected_behavior_failure"]
            for msg in conv["conversation_history"]
            if msg.get("role") == "assistant"
            and "test.expected_behavior_failure" in msg
        ]
        non_pass = [lbl for lbl in labels if lbl != "no failure"]
        result[conv["conversation_id"]] = non_pass[0] if non_pass else "no failure"
    return result


def _real_llm() -> object:
    from arksim.llms.chat import LLM

    return LLM(model=OPENAI_MODEL, provider="openai")


# Build parametrize list at collection time from the simulation file
_sim_data = json.loads(_SIM_PATH.read_text())
_conv_ids = [c["conversation_id"] for c in _sim_data["conversations"]]


@pytest.mark.parametrize("conv_id", _conv_ids)
def test_conversation_matches_expected_label(
    conv_id: str, tmp_path: pathlib.Path
) -> None:
    """For each conversation, at least one turn must match the expected label,
    or all turns must be 'no failure' when the expected label is 'no failure'."""
    sim = _load_simulation()
    conv = next(c for c in sim.conversations if c.conversation_id == conv_id)
    description = conv.simulated_user_prompt.variables.get("test.description", "")
    expected_label = _expected_labels()[conv_id]

    single = Simulation(
        schema_version=sim.schema_version,
        simulator_version=sim.simulator_version,
        simulation_id=sim.simulation_id,
        conversations=[conv],
    )
    result = Evaluator(
        params=EvaluationParams(output_dir=str(tmp_path), num_workers=1),
        llm=_real_llm(),
    ).evaluate(single)

    actual_labels = [
        t.turn_behavior_failure for t in result.conversations[0].turn_scores
    ]

    if expected_label == "no failure":
        for label in actual_labels:
            assert label not in _FAILURE_LABELS, (
                f"[{conv_id}] {description}\n"
                f"Clean conversation should have no failure label. "
                f"Got {actual_labels}"
            )
    else:
        assert expected_label in actual_labels, (
            f"[{conv_id}] {description}\n"
            f"Expected '{expected_label}' in turn labels, got {actual_labels}"
        )
