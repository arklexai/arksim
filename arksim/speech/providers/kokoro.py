# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from arksim.config import SpeechProviderConfig
from arksim.speech.base import TTSProvider
from arksim.speech.registry import register_tts
from arksim.speech.types import AudioBuffer

_KOKORO_SAMPLE_RATE = 24000


@register_tts("kokoro")
class KokoroTTS(TTSProvider):
    """Local Kokoro TTS (Apache-2.0). Requires ``arksim[voice]``."""

    def __init__(self, cfg: SpeechProviderConfig) -> None:
        super().__init__(cfg)
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise ImportError(
                "Kokoro TTS requires: pip install 'arksim[voice]'"
            ) from exc
        self._voice = cfg.options.get("voice", "af_heart")
        self._pipeline = KPipeline(lang_code=cfg.options.get("lang_code", "a"))

    async def synthesize(self, text: str) -> AudioBuffer:
        # KPipeline yields (graphemes, phonemes, audio) chunks; audio is a
        # torch.float32 tensor at 24 kHz.
        chunks = [chunk[-1] for chunk in self._pipeline(text, voice=self._voice)]
        if not chunks:
            raise RuntimeError(f"Kokoro produced no audio for text: {text!r}")
        samples = np.concatenate(
            [np.asarray(chunk, dtype=np.float32) for chunk in chunks]
        )
        return AudioBuffer(samples=samples, sample_rate=_KOKORO_SAMPLE_RATE)
