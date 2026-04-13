# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Annotated, Any, Literal

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
    arg_match_mode: Literal["exact", "ignore", "partial"] = "ignore"


# ── Assertion types ──


class AssertionType:
    """Constants for assertion type discriminators."""

    TOOL_CALLS = "tool_calls"


class ToolCallsAssertion(BaseModel):
    """Assert expected tool calls with trajectory matching."""

    type: Literal["tool_calls"]
    expected: list[ExpectedToolCall]
    match_mode: Literal["strict", "unordered", "contains", "within"] = "unordered"


# Discriminated union on the "type" field. When adding new assertion
# types, use Union: Annotated[TypeA | TypeB, Field(discriminator="type")]
Assertion = Annotated[ToolCallsAssertion, Field(discriminator="type")]


class Scenario(BaseModel):
    """A single scenario item."""

    scenario_id: str
    user_id: str
    goal: str
    agent_context: str
    knowledge: list[KnowledgeItem] = []
    user_profile: str
    origin: dict = Field(default_factory=dict)
    custom_rules: list[str] = Field(default_factory=list)
    assertions: list[Assertion] = []

    def find_assertion(self, assertion_type: str) -> ToolCallsAssertion | None:
        """Return the first assertion matching the given type, or None."""
        for a in self.assertions:
            if a.type == assertion_type:
                return a
        return None

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
