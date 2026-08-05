# SPDX-License-Identifier: Apache-2.0
"""A Pipecat voice agent for arksim evaluation.

``build()`` returns the agent's own ``(LLMContext, [stages])`` including its
real STT, LLM, and TTS services. arksim voices the simulated user with its own
TTS, injects that audio into this pipeline, captures the agent's spoken reply,
and transcribes it with its own STT. Only the agent's *own* speech stack is
exercised; tool calls the agent makes are captured automatically.

Requires:
    pip install 'arksim[voice]' 'pipecat-ai[openai,whisper,silero]'
    export OPENAI_API_KEY=...
"""

from __future__ import annotations

from typing import Any


def build() -> tuple[Any, list[Any]]:
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
    )
    from pipecat.services.openai.llm import OpenAILLMService
    from pipecat.services.openai.tts import OpenAITTSService
    from pipecat.services.whisper.stt import WhisperSTTService

    context = LLMContext(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a friendly phone support agent for an online "
                    "store. Keep replies short and conversational."
                ),
            }
        ]
    )
    aggregators = LLMContextAggregatorPair(context)
    stt = WhisperSTTService(model="base.en")
    llm = OpenAILLMService(model="gpt-4o-mini")
    tts = OpenAITTSService(voice="alloy")

    return context, [
        stt,
        aggregators.user(),
        llm,
        tts,
        aggregators.assistant(),
    ]
