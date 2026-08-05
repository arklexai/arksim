# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from arksim.config import SpeechProviderConfig
from arksim.speech.base import STTProvider
from arksim.speech.registry import register_stt
from arksim.speech.types import AudioBuffer

_WHISPER_SAMPLE_RATE = 16000


@register_stt("faster_whisper")
class FasterWhisperSTT(STTProvider):
    """Local faster-whisper STT. Requires ``arksim[voice]``."""

    def __init__(self, cfg: SpeechProviderConfig) -> None:
        super().__init__(cfg)
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "faster-whisper STT requires: pip install 'arksim[voice]'"
            ) from exc
        self._model = WhisperModel(
            cfg.model or "base.en",
            device=cfg.options.get("device", "cpu"),
            compute_type=cfg.options.get("compute_type", "int8"),
        )

    async def transcribe(self, audio: AudioBuffer) -> str:
        samples = _to_whisper_mono16k(audio)
        segments, _ = self._model.transcribe(samples, language="en")
        return "".join(segment.text for segment in segments).strip()


def _to_whisper_mono16k(audio: AudioBuffer) -> np.ndarray:
    """Downmix to mono and resample to 16 kHz float32 for Whisper."""
    samples = np.asarray(audio.samples, dtype=np.float32)
    if audio.num_channels > 1:
        samples = samples.reshape(-1, audio.num_channels).mean(axis=1)
    if audio.sample_rate != _WHISPER_SAMPLE_RATE:
        ratio = _WHISPER_SAMPLE_RATE / audio.sample_rate
        idx = np.round(np.arange(0, len(samples) * ratio) / ratio).astype(int)
        idx = idx[idx < len(samples)]
        samples = samples[idx]
    return samples
