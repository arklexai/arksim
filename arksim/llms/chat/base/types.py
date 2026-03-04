# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict


class LLMMessage(TypedDict):
    """Type for formatted LLM messages."""

    role: Literal["user", "assistant", "system"]
    content: str
