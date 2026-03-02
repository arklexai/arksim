from pydantic import BaseModel


class ScoreSchema(BaseModel):
    score: int
    reason: str


class QualSchema(BaseModel):
    label: str
    reason: str


class UniqueErrorSchema(BaseModel):
    agent_behavior_failure_category: str
    unique_error_description: str
    occurrences: list[str]


class UniqueErrorsSchema(BaseModel):
    unique_errors: list[UniqueErrorSchema]
