# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

from arksim.speech.types import AudioBuffer, audio_from_pcm16, pcm16_bytes


def test_audio_buffer_normalizes_samples_to_float32() -> None:
    audio = AudioBuffer(samples=np.array([0, 1], dtype=np.int16), sample_rate=16000)

    assert audio.samples.dtype == np.float32


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"samples": np.zeros((2, 2)), "sample_rate": 16000}, "one-dimensional"),
        ({"samples": np.zeros(2), "sample_rate": 0}, "sample_rate"),
        (
            {"samples": np.zeros(2), "sample_rate": 16000, "num_channels": 0},
            "num_channels",
        ),
        (
            {"samples": np.zeros(3), "sample_rate": 16000, "num_channels": 2},
            "divisible",
        ),
    ],
)
def test_audio_buffer_rejects_invalid_shape_or_metadata(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        AudioBuffer(**kwargs)


def test_pcm16_roundtrip_clips_and_preserves_metadata() -> None:
    original = AudioBuffer(
        samples=np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32),
        sample_rate=24000,
        num_channels=2,
    )

    decoded = audio_from_pcm16(
        pcm16_bytes(original),
        sample_rate=original.sample_rate,
        num_channels=original.num_channels,
    )

    assert decoded.sample_rate == 24000
    assert decoded.num_channels == 2
    assert decoded.samples == pytest.approx(
        [-1.0, -0.5, 0.5, 32767 / 32768], abs=1 / 32768
    )
