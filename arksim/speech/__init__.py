# SPDX-License-Identifier: Apache-2.0
"""Pluggable text-to-speech and speech-to-text providers for the voice loop."""

from __future__ import annotations

from arksim.speech.base import STTProvider, TTSProvider
from arksim.speech.registry import (
    create_stt,
    create_tts,
    register_stt,
    register_tts,
)
from arksim.speech.types import AudioBuffer

__all__ = [
    "AudioBuffer",
    "TTSProvider",
    "STTProvider",
    "create_tts",
    "create_stt",
    "register_tts",
    "register_stt",
]
