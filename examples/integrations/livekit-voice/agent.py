# SPDX-License-Identifier: Apache-2.0
"""A LiveKit Agents voice agent for arksim evaluation.

Requires:
    pip install 'arksim[livekit-voice]' 'livekit-agents[openai]>=1.5.9,<2.0'
    export OPENAI_API_KEY=...
"""

from __future__ import annotations

import os
from typing import Any


def build() -> tuple[Any, Any]:
    """Return the agent's own ``(AgentSession, Agent)``."""
    from livekit.agents import Agent, AgentSession, TurnHandlingOptions
    from livekit.plugins import openai

    session = AgentSession(
        stt=openai.STT(model="gpt-4o-mini-transcribe"),
        llm=openai.LLM(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini")),
        tts=openai.TTS(model="gpt-4o-mini-tts", voice="ash"),
        vad=None,
        turn_handling=TurnHandlingOptions(turn_detection="manual"),
    )
    agent = Agent(
        instructions=(
            "You are a friendly phone support agent for an online store. "
            "Keep replies short and conversational."
        )
    )
    return session, agent
