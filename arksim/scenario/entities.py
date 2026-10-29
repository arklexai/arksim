# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Annotated, Any, Literal

import re2
from pydantic import (
    BaseModel,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from arksim.utils.output import load_json_file


class KnowledgeItem(BaseModel):
    """Knowledge used in a scenario.

    ``id`` is a scenario-local chunk identifier (e.g. ``kc_01``) assigned at
    scenario compilation time so ``GoalChecklistItem.knowledge_chunk_ids`` can
    reference the chunks that answer each sub-goal. Uncompiled scenarios leave
    it unset.
    """

    id: str | None = None
    content: str = ""


class BehaviorParams(BaseModel):
    """Numeric knobs that seed the simulated user's state."""

    patience: int = 3


class GoalChecklistItem(BaseModel):
    """One sub-goal of the scenario goal."""

    type: Literal["knowledge_answer"] = "knowledge_answer"
    description: str
    satisfied_when: str
    mandatory: bool = True
    knowledge_chunk_ids: list[str] = []


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
    LLM_JUDGE = "llm_judge"
    TEXT_MATCH = "text_match"


class ToolCallsAssertion(BaseModel):
    """Assert expected tool calls with trajectory matching.

    ``forbidden`` lists calls the agent must not make; a forbidden entry
    matches with the same name and argument rules as an expected one.
    """

    type: Literal["tool_calls"]
    expected: list[ExpectedToolCall] = []
    match_mode: Literal["strict", "unordered", "contains", "within"] = "unordered"
    forbidden: list[ExpectedToolCall] = []

    @model_serializer(mode="wrap")
    def _omit_empty_forbidden(self, handler: SerializerFunctionWrapHandler) -> dict:
        # Scenarios written before ``forbidden`` existed round-trip unchanged.
        data = handler(self)
        if not self.forbidden:
            data.pop("forbidden", None)
        return data


class LLMJudgeAssertion(BaseModel):
    """Expected agent behavior, judged by an LLM over the whole conversation."""

    type: Literal["llm_judge"]
    expected: str = Field(min_length=1)


class TextMatchAssertion(BaseModel):
    """Deterministic check over the agent's replies.

    ``contains`` and ``matches`` (regex) pass when any reply matches;
    ``not_contains`` passes only when no reply does. Case is always ignored.
    """

    type: Literal["text_match"]
    op: Literal["contains", "not_contains", "matches"]
    value: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_pattern(self) -> TextMatchAssertion:
        try:
            self.compiled()
        except re2.error as e:
            raise ValueError(f"Invalid regex {self.value!r}: {e}") from e
        return self

    def compiled(self) -> re2._Regexp:
        """The value compiled as RE2, literal unless ``matches``, ignoring case.

        RE2 is the dialect the platform validates patterns with, and it matches
        in linear time, so a user pattern cannot stall evaluation.
        """
        options = re2.Options()
        options.log_errors = False
        options.case_sensitive = False
        options.literal = self.op != "matches"
        return re2.compile(self.value, options)


Assertion = Annotated[
    ToolCallsAssertion | LLMJudgeAssertion | TextMatchAssertion,
    Field(discriminator="type"),
]


class Scenario(BaseModel):
    """A single scenario item."""

    scenario_id: str
    user_id: str
    goal: str
    agent_context: str
    knowledge: list[KnowledgeItem] = []
    user_profile: str
    origin: dict = Field(default_factory=dict)
    assertions: list[Assertion] = []
    behavior: BehaviorParams | None = None
    goal_checklist: list[GoalChecklistItem] = []

    def find_assertion(
        self, assertion_type: str
    ) -> ToolCallsAssertion | LLMJudgeAssertion | TextMatchAssertion | None:
        """Return the first assertion matching the given type, or None."""
        for a in self.assertions:
            if a.type == assertion_type:
                return a
        return None

    @model_validator(mode="after")
    def check_single_tool_calls_assertion(self) -> Scenario:
        """At most one tool_calls assertion; trajectories are matched as a whole."""
        if sum(a.type == AssertionType.TOOL_CALLS for a in self.assertions) > 1:
            raise ValueError("A scenario can have at most one tool_calls assertion.")
        return self

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
