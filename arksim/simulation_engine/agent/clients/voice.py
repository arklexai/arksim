# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid
from typing import Protocol

from arksim.config import AgentConfig, AgentType, VoiceFramework
from arksim.simulation_engine.agent.base import BaseAgent
from arksim.simulation_engine.tool_types import AgentResponse
from arksim.utils.module_loader import load_callable


class VoiceDriver(Protocol):
    """Framework-specific driver that runs one voice turn."""

    async def run_turn(self, user_query: str) -> AgentResponse: ...

    async def get_chat_id(self) -> str: ...

    async def close(self) -> None: ...


def _build_driver_for(agent_config: AgentConfig) -> VoiceDriver:
    """Resolve the user factory and build the framework driver.

    The framework is checked before constructing speech providers so an
    unsupported framework fails fast without requiring the voice extras.
    """
    vc = agent_config.voice_config
    if vc is None:
        raise ValueError("Voice agent requires voice_config")
    if vc.framework == VoiceFramework.LIVEKIT:
        from arksim.integrations.livekit import LiveKitVoiceDriver
        from arksim.speech import create_stt, create_tts

        factory = load_callable(vc.agent_factory)
        return LiveKitVoiceDriver(
            factory, tts=create_tts(vc.tts), stt=create_stt(vc.stt)
        )
    if vc.framework == VoiceFramework.PIPECAT:
        from arksim.integrations.pipecat import PipecatVoiceDriver
        from arksim.speech import create_stt, create_tts

        factory = load_callable(vc.agent_factory)
        return PipecatVoiceDriver(
            factory, tts=create_tts(vc.tts), stt=create_stt(vc.stt)
        )
    raise ValueError(f"Unsupported voice framework: {vc.framework}")


class VoiceAgent(BaseAgent):
    """Native voice agent: drives a framework agent through its ASR/LLM/TTS stack."""

    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        if agent_config.agent_type != AgentType.VOICE.value:
            raise ValueError("Agent config must be of type voice")
        self._chat_id = str(uuid.uuid4())
        self._driver: VoiceDriver | None = None

    def _driver_or_build(self) -> VoiceDriver:
        if self._driver is None:
            self._driver = _build_driver_for(self.agent_config)
        return self._driver

    async def get_chat_id(self) -> str:
        if self._driver is not None:
            return await self._driver.get_chat_id()
        return self._chat_id

    async def execute(self, user_query: str, **kwargs: object) -> AgentResponse:
        return await self._driver_or_build().run_turn(user_query)

    async def close(self) -> None:
        driver = self._driver
        self._driver = None
        if driver is not None:
            await driver.close()
