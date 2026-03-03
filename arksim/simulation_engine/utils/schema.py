# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel


class MatchedAttributeSchema(BaseModel):
    """Schema for LLM-matched user attributes."""

    thought: str
    attribute: str
