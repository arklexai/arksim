# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys

import pytest

from arksim.config import SpeechProviderConfig

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="kokoro/pipecat require 3.11+"
)


async def test_kokoro_to_whisper_roundtrip() -> None:
    pytest.importorskip("kokoro")
    pytest.importorskip("faster_whisper")
    from arksim.speech import create_stt, create_tts

    tts = create_tts(SpeechProviderConfig(provider="kokoro"))
    stt = create_stt(SpeechProviderConfig(provider="faster_whisper", model="tiny.en"))
    audio = await tts.synthesize("the quick brown fox")
    assert audio.sample_rate == 24000
    assert len(audio.samples) > 0
    text = await stt.transcribe(audio)
    assert "fox" in text.lower()
