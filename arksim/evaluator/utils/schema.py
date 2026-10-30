# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pydantic import BaseModel


class ScoreSchema(BaseModel):
    # reason precedes score: structured output generates fields in property
    # order, so the judge reasons before committing to a score.
    reason: str
    score: int


class QualSchema(BaseModel):
    # reason precedes label for the same reason as ScoreSchema.
    reason: str
    label: str


class UniqueErrorSchema(BaseModel):
    agent_behavior_failure_category: str
    unique_error_description: str
    occurrences: list[str]


class UniqueErrorsSchema(BaseModel):
    unique_errors: list[UniqueErrorSchema]
