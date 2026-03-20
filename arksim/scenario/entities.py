# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from arksim.utils.output import load_json_file


class KnowledgeItem(BaseModel):
    """Knowledge used in a scenario."""

    content: str = ""


class Scenario(BaseModel):
    """A single scenario item."""

    scenario_id: str
    user_id: str
    goal: str
    agent_context: str
    knowledge: list[KnowledgeItem] = []
    user_profile: str
    expected_outcomes: list[str] = Field(
        default_factory=list,
        description=(
            "Optional list of expected outcomes for evaluation. "
            "When provided, the evaluator uses these instead of the user goal "
            "to judge agent behavior — enabling correct classification of "
            "intentional scope-based refusals as 'no failure'."
        ),
    )
    origin: dict = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def check_user_attributes_without_profile(cls, data: dict) -> dict:
        """Error when legacy user_attributes exists without user_profile."""
        if isinstance(data, dict):
            has_attrs = "user_attributes" in data
            has_profile = bool(data.get("user_profile"))
            if has_attrs and not has_profile:
                sid = data.get("scenario_id", "<unknown>")
                raise ValueError(
                    f"Scenario '{sid}' has 'user_attributes' but no "
                    "'user_profile'. 'user_attributes' has moved into "
                    "'origin' and is no longer used for profile "
                    "generation. Set 'user_profile' directly."
                )
        return data


class Scenarios(BaseModel):
    """Output of the scenario builder."""

    schema_version: str
    scenarios: list[Scenario]

    @classmethod
    def load(cls, path: str) -> Scenarios:
        """Load scenarios from a JSON file."""
        return cls.model_validate(load_json_file(path))
