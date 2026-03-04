# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from enum import Enum


class AgentType(Enum):
    """Agent type."""

    CHAT_COMPLETIONS = "chat_completions"
    A2A = "a2a"
