# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from arksim.utils.output import load_json_file


class KnowledgeItem(BaseModel):
    """Knowledge used in a scenario."""

    content: str = ""


class ExpectedToolCall(BaseModel):
    """An expected tool call for trajectory matching.

    Simplified model (no id/result/error) since expected calls only
    describe what the agent should call, not the outcomes.
    """

    name: str
    arguments: dict[str, Any] = {}
    arg_match_mode: Literal["exact", "ignore", "subset"] = "ignore"


class Scenario(BaseModel):
    """A single scenario item."""

    scenario_id: str
    user_id: str
    goal: str
    agent_context: str
    knowledge: list[KnowledgeItem] = []
    user_profile: str
    origin: dict = Field(default_factory=dict)
    expected_tool_calls: list[ExpectedToolCall] | None = None
    match_mode: Literal["strict", "unordered", "subset", "superset"] = "unordered"

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
