# SPDX-License-Identifier: Apache-2.0
"""Key-free, fully-local smoke test of the native ``voice`` agent type.

Runs the real arksim voice loop end to end with no API keys: a genuine Pipecat
pipeline using real faster-whisper ASR + real Kokoro TTS, with a deterministic
"brain" standing in for an LLM. Proves the audio path:

    text -> arksim Kokoro TTS -> AUDIO -> agent ASR -> brain -> agent Kokoro TTS
         -> AUDIO -> arksim faster-whisper STT -> text

Usage:
    pip install 'arksim[voice]'
    python examples/integrations/pipecat-voice/smoke_local.py

The first run downloads the small Whisper and Kokoro models. For a real LLM-
backed agent, see ``agent.py`` / ``config.yaml`` in this directory instead.
"""

from __future__ import annotations

import asyncio

import numpy as np
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

_WHISPER_RATE = 16000
_KOKORO_RATE = 24000


def _to_16k_float(audio_bytes: bytes, sample_rate: int) -> np.ndarray:
    samples = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32) / 32768.0
    if sample_rate != _WHISPER_RATE:
        ratio = _WHISPER_RATE / sample_rate
        idx = np.round(np.arange(0, len(samples) * ratio) / ratio).astype(int)
        samples = samples[idx[idx < len(samples)]]
    return samples


class RealWhisperSTT(FrameProcessor):
    """The agent's own ASR: transcribes injected user audio with faster-whisper."""

    def __init__(self) -> None:
        super().__init__()
        from faster_whisper import WhisperModel

        self._model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            segments, _ = self._model.transcribe(
                _to_16k_float(frame.audio, frame.sample_rate), language="en"
            )
            text = "".join(segment.text for segment in segments).strip()
            await self.push_frame(
                TranscriptionFrame(text, "sim-user", ""), FrameDirection.DOWNSTREAM
            )
        else:
            await self.push_frame(frame, direction)


class CannedBrain(FrameProcessor):
    """Stands in for an LLM: echoes the transcript so ASR fidelity is visible."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            reply = f"I heard you say: {frame.text}"
            await self.push_frame(
                LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM
            )
            await self.push_frame(TextFrame(reply), FrameDirection.DOWNSTREAM)
            await self.push_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
        else:
            await self.push_frame(frame, direction)


class RealKokoroTTS(FrameProcessor):
    """The agent's own TTS: speaks the reply with Kokoro."""

    def __init__(self) -> None:
        super().__init__()
        from kokoro import KPipeline

        self._pipeline = KPipeline(lang_code="a")

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            chunks = [
                chunk[-1] for chunk in self._pipeline(frame.text, voice="af_heart")
            ]
            samples = np.concatenate(
                [np.asarray(chunk, dtype=np.float32) for chunk in chunks]
            )
            pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            await self.push_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)
            await self.push_frame(
                TTSAudioRawFrame(audio=pcm, sample_rate=_KOKORO_RATE, num_channels=1),
                FrameDirection.DOWNSTREAM,
            )
            await self.push_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
        else:
            await self.push_frame(frame, direction)


def build_demo_agent() -> tuple[None, list[FrameProcessor]]:
    """Return a real Pipecat voice pipeline with a deterministic brain."""
    return None, [RealWhisperSTT(), CannedBrain(), RealKokoroTTS()]


async def _main() -> None:
    from arksim.config import AgentConfig
    from arksim.simulation_engine.agent.factory import create_agent

    config = AgentConfig.model_validate(
        {
            "agent_name": "demo-voice",
            "agent_type": "voice",
            "voice_config": {
                "framework": "pipecat",
                "agent_factory": f"{__file__}:build_demo_agent",
                "tts": {"provider": "kokoro"},
                "stt": {"provider": "faster_whisper", "model": "tiny.en"},
            },
        }
    )
    agent = create_agent(config)
    try:
        for utterance in ["Hello, where is my order?", "Thanks, that is helpful."]:
            response = await agent.execute(utterance)
            print(f"\nUSER  (text -> speech): {utterance}")
            print(f"AGENT (speech -> text): {response.content!r}")
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(_main())
