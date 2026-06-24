# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from arksim.config import SpeechProviderConfig
from arksim.speech.base import STTProvider, TTSProvider

_TTS: dict[str, type[TTSProvider]] = {}
_STT: dict[str, type[STTProvider]] = {}

T = TypeVar("T", bound=TTSProvider)
S = TypeVar("S", bound=STTProvider)


def register_tts(name: str) -> Callable[[type[T]], type[T]]:
    """Register a TTS provider class under ``name``."""

    def deco(cls: type[T]) -> type[T]:
        _TTS[name] = cls
        return cls

    return deco


def register_stt(name: str) -> Callable[[type[S]], type[S]]:
    """Register an STT provider class under ``name``."""

    def deco(cls: type[S]) -> type[S]:
        _STT[name] = cls
        return cls

    return deco


def create_tts(cfg: SpeechProviderConfig) -> TTSProvider:
    """Instantiate the registered TTS provider named ``cfg.provider``."""
    if cfg.provider not in _TTS:
        raise ValueError(
            f"Unknown TTS provider '{cfg.provider}'. Registered: {sorted(_TTS)}"
        )
    return _TTS[cfg.provider](cfg)


def create_stt(cfg: SpeechProviderConfig) -> STTProvider:
    """Instantiate the registered STT provider named ``cfg.provider``."""
    if cfg.provider not in _STT:
        raise ValueError(
            f"Unknown STT provider '{cfg.provider}'. Registered: {sorted(_STT)}"
        )
    return _STT[cfg.provider](cfg)
