# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AudioBuffer:
    """PCM audio interchange type for the voice loop.

    ``samples`` is mono float32 in [-1, 1] unless ``num_channels`` > 1.
    """

    samples: np.ndarray
    sample_rate: int
    num_channels: int = 1


def pcm16_bytes(audio: AudioBuffer) -> bytes:
    """Encode a float32 ``AudioBuffer`` as little-endian signed 16-bit PCM."""
    clipped = np.clip(np.asarray(audio.samples, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def audio_from_pcm16(
    data: bytes, sample_rate: int, num_channels: int = 1
) -> AudioBuffer:
    """Decode little-endian signed 16-bit PCM into a float32 ``AudioBuffer``."""
    samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    return AudioBuffer(
        samples=samples, sample_rate=sample_rate, num_channels=num_channels
    )
