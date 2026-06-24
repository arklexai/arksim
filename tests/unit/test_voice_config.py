# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
from pydantic import ValidationError

from arksim.config import AgentConfig, VoiceConfig, VoiceFramework


def _voice_dict(**over: object) -> dict:
    base = {
        "agent_name": "support-bot",
        "agent_type": "voice",
        "voice_config": {
            "framework": "pipecat",
            "agent_factory": "./agent.py:build",
        },
    }
    base["voice_config"].update(over)
    return base


def test_voice_config_parses_and_defaults_local_providers() -> None:
    cfg = AgentConfig.model_validate(_voice_dict())
    assert cfg.agent_type == "voice"
    assert cfg.voice_config.framework is VoiceFramework.PIPECAT
    assert cfg.voice_config.agent_factory == "./agent.py:build"
    assert cfg.voice_config.tts.provider == "kokoro"
    assert cfg.voice_config.stt.provider == "faster_whisper"


def test_voice_config_accepts_explicit_providers() -> None:
    cfg = AgentConfig.model_validate(
        _voice_dict(
            tts={"provider": "kokoro", "options": {"voice": "af_heart"}},
            stt={"provider": "faster_whisper", "model": "base.en"},
        )
    )
    assert cfg.voice_config.stt.model == "base.en"
    assert cfg.voice_config.tts.options == {"voice": "af_heart"}


def test_voice_agent_requires_voice_config() -> None:
    with pytest.raises(ValueError, match="requires 'voice_config'"):
        AgentConfig(agent_name="x", agent_type="voice")


def test_unknown_framework_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentConfig.model_validate(_voice_dict(framework="nope"))


def test_voice_config_constructed_directly() -> None:
    vc = VoiceConfig(framework=VoiceFramework.PIPECAT, agent_factory="m:build")
    assert vc.tts.provider == "kokoro"
    assert vc.stt.provider == "faster_whisper"
