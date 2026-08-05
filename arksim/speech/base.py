# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from abc import ABC, abstractmethod

from arksim.config import SpeechProviderConfig
from arksim.speech.types import AudioBuffer


class TTSProvider(ABC):
    """Synthesizes the simulated user's utterance into audio."""

    def __init__(self, cfg: SpeechProviderConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    async def synthesize(self, text: str) -> AudioBuffer: ...


class STTProvider(ABC):
    """Transcribes the agent's audio reply back to text."""

    def __init__(self, cfg: SpeechProviderConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    async def transcribe(self, audio: AudioBuffer) -> str: ...
