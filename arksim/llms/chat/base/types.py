# SPDX-License-Identifier: Apache-2.0
from typing import Literal

from pydantic import BaseModel
from typing_extensions import TypedDict


class LLMMessage(TypedDict):
    """Type for formatted LLM messages."""

    role: Literal["user", "assistant", "system"]
    content: str


class LLMConfig(BaseModel):
    model: str
    temperature: float
    provider: str | None = None
