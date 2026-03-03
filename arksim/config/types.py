# SPDX-License-Identifier: Apache-2.0
from enum import Enum


class AgentType(Enum):
    """Agent type."""

    CHAT_COMPLETIONS = "chat_completions"
    A2A = "a2a"
