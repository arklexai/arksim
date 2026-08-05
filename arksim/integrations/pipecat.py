# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import Callable
from typing import Any

import numpy as np

from arksim.simulation_engine.tool_types import (
    AgentResponse,
    ToolCall,
    ToolCallSource,
)
from arksim.speech.base import STTProvider, TTSProvider
from arksim.speech.types import AudioBuffer, audio_from_pcm16, pcm16_bytes

try:
    from pipecat.frames.frames import (
        EndFrame,
        Frame,
        FunctionCallResultFrame,
        InputAudioRawFrame,
        OutputAudioRawFrame,
        TTSAudioRawFrame,
        TTSStartedFrame,
        TTSStoppedFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
        VADUserStartedSpeakingFrame,
        VADUserStoppedSpeakingFrame,
    )
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineTask
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
except ImportError as exc:  # pragma: no cover - exercised only without extras
    raise ImportError(
        "Pipecat voice support requires: pip install 'arksim[voice]'"
    ) from exc

# Seconds to wait for one agent turn (STT -> LLM -> TTS) to produce audio.
_TURN_TIMEOUT_S = 120.0
_DEFAULT_SAMPLE_RATE = 24000
# Pipecat's default audio_in_sample_rate; segmented STT writes its buffer as a
# WAV at this rate without resampling, so injected audio must match it.
_PIPELINE_INPUT_RATE = 16000


def _resample(audio: AudioBuffer, target_rate: int) -> AudioBuffer:
    """Downmix and nearest-neighbor resample audio for Pipecat input."""
    if target_rate <= 0:
        raise ValueError("target_rate must be positive")
    if audio.sample_rate == target_rate and audio.num_channels == 1:
        return audio
    samples = np.asarray(audio.samples, dtype=np.float32)
    if audio.num_channels > 1:
        samples = samples.reshape(-1, audio.num_channels).mean(axis=1)
    if audio.sample_rate == target_rate:
        return AudioBuffer(samples=samples, sample_rate=target_rate)
    ratio = target_rate / audio.sample_rate
    idx = np.round(np.arange(0, len(samples) * ratio) / ratio).astype(int)
    return AudioBuffer(
        samples=samples[idx[idx < len(samples)]],
        sample_rate=target_rate,
        num_channels=1,
    )


class _CaptureProcessor(FrameProcessor):
    """Collects the agent's TTS audio and tool calls for a single turn."""

    def __init__(self) -> None:
        super().__init__()
        self._audio = bytearray()
        self._sample_rate: int | None = None
        self._num_channels = 1
        self.tool_calls: list[ToolCall] = []
        self._collecting = False
        self.turn_done = asyncio.Event()

    def reset(self) -> None:
        self._audio = bytearray()
        self._sample_rate = None
        self._num_channels = 1
        self.tool_calls = []
        self._collecting = False
        self.turn_done.clear()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSStartedFrame):
            self._collecting = True
        elif (
            isinstance(frame, TTSAudioRawFrame | OutputAudioRawFrame)
            and self._collecting
        ):
            self._audio.extend(frame.audio)
            self._sample_rate = frame.sample_rate
            self._num_channels = frame.num_channels
        elif isinstance(frame, FunctionCallResultFrame):
            self.tool_calls.append(
                ToolCall(
                    id=frame.tool_call_id or frame.function_name,
                    name=frame.function_name,
                    arguments=frame.arguments or {},
                    result=None if frame.result is None else str(frame.result),
                    source=ToolCallSource.PIPECAT,
                )
            )
        elif isinstance(frame, TTSStoppedFrame):
            self._collecting = False
            self.turn_done.set()
        await self.push_frame(frame, direction)

    def captured_audio(self) -> AudioBuffer | None:
        if not self._audio:
            return None
        return audio_from_pcm16(
            bytes(self._audio),
            self._sample_rate or _DEFAULT_SAMPLE_RATE,
            self._num_channels,
        )


class PipecatVoiceDriver:
    """Drives a Pipecat voice agent through its ASR/LLM/TTS stack.

    arksim synthesizes the simulated user's utterance (``tts``), injects it as
    audio, captures the agent's spoken reply, and transcribes it (``stt``).
    """

    def __init__(
        self,
        factory: Callable[[], Any],
        *,
        tts: TTSProvider,
        stt: STTProvider,
    ) -> None:
        self._factory = factory
        self._tts = tts
        self._stt = stt
        self._chat_id = str(uuid.uuid4())
        self._task: PipelineTask | None = None
        self._runner_run: asyncio.Task[Any] | None = None
        self._capture: _CaptureProcessor | None = None

    async def _ensure_started(self) -> None:
        if self._task is not None:
            return
        built = self._factory()
        stages = built[1] if isinstance(built, tuple) else built
        self._capture = _CaptureProcessor()
        self._task = PipelineTask(Pipeline([*stages, self._capture]))
        runner = PipelineRunner()
        self._runner_run = asyncio.create_task(runner.run(self._task))
        # Yield so the runner emits StartFrame before the first turn is queued.
        await asyncio.sleep(0.1)

    async def run_turn(self, user_query: str) -> AgentResponse:
        await self._ensure_started()
        assert self._task is not None
        assert self._capture is not None
        self._capture.reset()
        audio = self._perturb(await self._tts.synthesize(user_query))
        # The agent's segmented STT writes its buffered audio as a WAV at its
        # own input sample rate (no resampling), so match that rate; and it
        # segments on VAD speaking frames. Bracket the audio with both VAD and
        # plain speaking frames so the STT transcribes and the user-context
        # aggregator advances the turn.
        audio = _resample(audio, _PIPELINE_INPUT_RATE)
        await self._task.queue_frames(
            [
                VADUserStartedSpeakingFrame(),
                UserStartedSpeakingFrame(),
                InputAudioRawFrame(
                    audio=pcm16_bytes(audio),
                    sample_rate=_PIPELINE_INPUT_RATE,
                    num_channels=1,
                ),
                UserStoppedSpeakingFrame(),
                VADUserStoppedSpeakingFrame(),
            ]
        )
        try:
            await asyncio.wait_for(self._capture.turn_done.wait(), _TURN_TIMEOUT_S)
        except (TimeoutError, asyncio.TimeoutError) as exc:
            await self.close()
            raise TimeoutError(
                f"Pipecat agent produced no audio within {_TURN_TIMEOUT_S}s "
                f"for: {user_query!r}"
            ) from exc
        captured = self._capture.captured_audio()
        if captured is None:
            await self.close()
            raise RuntimeError(f"Pipecat agent produced no audio for: {user_query!r}")
        reply = await self._stt.transcribe(captured)
        return AgentResponse(content=reply, tool_calls=list(self._capture.tool_calls))

    def _perturb(self, audio: AudioBuffer) -> AudioBuffer:
        # v2 seam: accent / background-noise / volume perturbation hooks here.
        return audio

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def close(self) -> None:
        if self._task is not None:
            await self._task.queue_frames([EndFrame()])
            self._task = None
        if self._runner_run is not None:
            try:
                await asyncio.wait_for(self._runner_run, timeout=10)
            except (TimeoutError, asyncio.TimeoutError):
                self._runner_run.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._runner_run
            self._runner_run = None


def _build_echo_stages_for_test() -> tuple[None, list[FrameProcessor]]:
    """A fake voice pipeline for unit tests: no real models or network.

    On each injected user audio frame it emits one tool-call result then a
    short TTS audio burst bracketed by TTS start/stop, mimicking real Pipecat
    frame ordering.
    """

    class _Echo(FrameProcessor):
        async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
            await super().process_frame(frame, direction)
            if isinstance(frame, InputAudioRawFrame):
                await self.push_frame(
                    FunctionCallResultFrame(
                        function_name="lookup_order",
                        tool_call_id="call_1",
                        arguments={"order_id": "A123"},
                        result="shipped",
                    ),
                    FrameDirection.DOWNSTREAM,
                )
                await self.push_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)
                await self.push_frame(
                    TTSAudioRawFrame(
                        audio=frame.audio,
                        sample_rate=frame.sample_rate,
                        num_channels=frame.num_channels,
                    ),
                    FrameDirection.DOWNSTREAM,
                )
                await self.push_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
            else:
                await self.push_frame(frame, direction)

    return None, [_Echo()]
