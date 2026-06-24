# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from enum import Enum


class AgentType(Enum):
    """Agent type."""

    CHAT_COMPLETIONS = "chat_completions"
    A2A = "a2a"
    CUSTOM = "custom"
    VOICE = "voice"


class VoiceFramework(str, Enum):
    """Voice agent framework backing a ``voice`` agent."""

    PIPECAT = "pipecat"
    LIVEKIT = "livekit"
