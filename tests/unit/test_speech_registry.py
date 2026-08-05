# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

from arksim.config import SpeechProviderConfig
from arksim.speech import (
    AudioBuffer,
    STTProvider,
    TTSProvider,
    create_stt,
    create_tts,
    register_stt,
    register_tts,
)


@register_tts("fake")
class _FakeTTS(TTSProvider):
    async def synthesize(self, text: str) -> AudioBuffer:
        return AudioBuffer(samples=np.zeros(16000, dtype=np.float32), sample_rate=16000)


@register_stt("fake")
class _FakeSTT(STTProvider):
    async def transcribe(self, audio: AudioBuffer) -> str:
        return f"len={len(audio.samples)}"


async def test_create_and_roundtrip() -> None:
    tts = create_tts(SpeechProviderConfig(provider="fake"))
    stt = create_stt(SpeechProviderConfig(provider="fake"))
    audio = await tts.synthesize("hello")
    assert audio.sample_rate == 16000
    assert await stt.transcribe(audio) == "len=16000"


def test_unknown_tts_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown TTS provider 'ghost'"):
        create_tts(SpeechProviderConfig(provider="ghost"))


def test_unknown_stt_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown STT provider 'ghost'"):
        create_stt(SpeechProviderConfig(provider="ghost"))
