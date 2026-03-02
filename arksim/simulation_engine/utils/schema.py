from pydantic import BaseModel


class MatchedAttributeSchema(BaseModel):
    """Schema for LLM-matched user attributes."""

    thought: str
    attribute: str
