# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable, Iterator

import numpy as np

from arksim.simulation_engine.tool_types import (
    AgentResponse,
    ToolCall,
    ToolCallSource,
)
from arksim.speech.base import STTProvider, TTSProvider
from arksim.speech.types import AudioBuffer, audio_from_pcm16, pcm16_bytes

try:
    from livekit import rtc
    from livekit.agents import Agent, AgentSession
    from livekit.agents.voice.events import FunctionToolsExecutedEvent
    from livekit.agents.voice.io import (
        AudioInput,
        AudioOutput,
        AudioOutputCapabilities,
    )
except ImportError as exc:  # pragma: no cover - exercised only without extras
    raise ImportError(
        "LiveKit voice support requires: pip install 'arksim[livekit-voice]'"
    ) from exc

_TURN_TIMEOUT_S = 120.0
_TRANSCRIPT_TIMEOUT_S = 10.0
_STT_FLUSH_DURATION_S = 1.0
_FRAME_DURATION_MS = 20


def _parse_arguments(raw: object) -> dict[str, object]:
    """Normalize LiveKit's JSON-encoded function arguments."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {"_value": raw}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"_value": raw}
    return parsed if isinstance(parsed, dict) else {"_value": parsed}


def _rtc_frames(audio: AudioBuffer) -> Iterator[rtc.AudioFrame]:
    """Split an audio buffer into LiveKit-compatible 20 ms PCM frames."""
    if audio.num_channels < 1:
        raise ValueError("Audio must have at least one channel")
    pcm = pcm16_bytes(audio)
    bytes_per_sample = 2 * audio.num_channels
    if len(pcm) % bytes_per_sample:
        raise ValueError("Audio sample count must be divisible by num_channels")

    samples_per_channel = len(pcm) // bytes_per_sample
    frame_samples = max(1, audio.sample_rate * _FRAME_DURATION_MS // 1000)
    for start in range(0, samples_per_channel, frame_samples):
        count = min(frame_samples, samples_per_channel - start)
        byte_start = start * bytes_per_sample
        byte_end = byte_start + count * bytes_per_sample
        yield rtc.AudioFrame(
            data=pcm[byte_start:byte_end],
            sample_rate=audio.sample_rate,
            num_channels=audio.num_channels,
            samples_per_channel=count,
        )


class _QueueAudioInput(AudioInput):
    """In-memory LiveKit audio source fed by arksim's simulated user."""

    def __init__(self) -> None:
        super().__init__(label="arksim")
        self._frames: asyncio.Queue[rtc.AudioFrame | None] = asyncio.Queue()
        self._attached = True
        self._delivered_frame_pending = False

    def push(self, audio: AudioBuffer) -> None:
        if not self._attached:
            raise RuntimeError("LiveKit audio input is detached")
        for frame in _rtc_frames(audio):
            self._frames.put_nowait(frame)

    async def __anext__(self) -> rtc.AudioFrame:
        # LiveKit asks for the next frame only after pushing the previous one
        # into AgentSession. Mark it delivered at that point so wait_drained()
        # cannot race commit_user_turn() ahead of the audio forwarding task.
        if self._delivered_frame_pending:
            self._frames.task_done()
            self._delivered_frame_pending = False
        frame = await self._frames.get()
        if frame is None:
            self._frames.task_done()
            raise StopAsyncIteration
        self._delivered_frame_pending = True
        return frame

    def on_attached(self) -> None:
        self._attached = True

    def on_detached(self) -> None:
        self._attached = False

    def close(self) -> None:
        self._frames.put_nowait(None)

    async def wait_drained(self) -> None:
        await self._frames.join()


class _CaptureAudioOutput(AudioOutput):
    """In-memory LiveKit audio sink that captures complete speech segments."""

    def __init__(self) -> None:
        super().__init__(
            label="arksim",
            capabilities=AudioOutputCapabilities(pause=False),
        )
        self.turn_done = asyncio.Event()
        self._audio = bytearray()
        self._sample_rate: int | None = None
        self._num_channels: int | None = None
        self._segment_duration = 0.0

    def reset(self) -> None:
        self.turn_done.clear()
        self._audio.clear()
        self._sample_rate = None
        self._num_channels = None
        self._segment_duration = 0.0

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        if self._sample_rate is None:
            self._sample_rate = frame.sample_rate
            self._num_channels = frame.num_channels
        elif (
            frame.sample_rate != self._sample_rate
            or frame.num_channels != self._num_channels
        ):
            raise RuntimeError("LiveKit changed audio format during one turn")
        self._audio.extend(frame.data)
        self._segment_duration += frame.duration

    def flush(self) -> None:
        super().flush()
        if not self._segment_duration:
            return
        duration = self._segment_duration
        self._segment_duration = 0.0
        self.on_playback_finished(
            playback_position=duration,
            interrupted=False,
        )
        self.turn_done.set()

    def clear_buffer(self) -> None:
        duration = self._segment_duration
        self._segment_duration = 0.0
        self._audio.clear()
        if duration:
            self.on_playback_finished(
                playback_position=duration,
                interrupted=True,
            )
        self.turn_done.set()

    def captured_audio(self) -> AudioBuffer | None:
        if not self._audio:
            return None
        return audio_from_pcm16(
            bytes(self._audio),
            self._sample_rate or 24000,
            self._num_channels or 1,
        )


class LiveKitVoiceDriver:
    """Drive a LiveKit AgentSession through its STT/LLM/TTS audio path.

    The factory must return ``(AgentSession, Agent)``. The session owns the
    agent's real STT, LLM, and TTS providers; arksim replaces only its media
    input and output with in-memory PCM endpoints.
    """

    def __init__(
        self,
        factory: Callable[[], tuple[AgentSession, Agent]],
        *,
        tts: TTSProvider,
        stt: STTProvider,
    ) -> None:
        self._factory = factory
        self._tts = tts
        self._stt = stt
        self._chat_id = str(uuid.uuid4())
        self._session: AgentSession | None = None
        self._audio_input: _QueueAudioInput | None = None
        self._audio_output: _CaptureAudioOutput | None = None
        self._tool_calls: list[ToolCall] = []

    async def _ensure_started(self) -> None:
        if self._session is not None:
            return
        built = self._factory()
        if not isinstance(built, tuple) or len(built) != 2:
            raise TypeError("LiveKit agent_factory must return (AgentSession, Agent)")
        session, agent = built
        if not isinstance(session, AgentSession) or not isinstance(agent, Agent):
            raise TypeError("LiveKit agent_factory must return (AgentSession, Agent)")
        if session.turn_detection != "manual":
            raise ValueError(
                "LiveKit AgentSession must use manual turn detection so arksim "
                "controls audio turn boundaries"
            )

        audio_input = _QueueAudioInput()
        audio_output = _CaptureAudioOutput()
        self._session = session
        self._audio_input = audio_input
        self._audio_output = audio_output
        try:
            session.input.audio = audio_input
            session.output.audio = audio_output
            session.on("function_tools_executed", self._on_tools_executed)
            await session.start(agent=agent, record=False)
            await session.wait_for_idle()
            audio_output.reset()
        except BaseException:
            await self.close()
            raise

    def _on_tools_executed(self, event: FunctionToolsExecutedEvent) -> None:
        for fn_call, fn_output in event.zipped():
            result: str | None = None
            error: str | None = None
            if fn_output is not None:
                value = None if fn_output.output is None else str(fn_output.output)
                if fn_output.is_error:
                    error = value
                else:
                    result = value
            self._tool_calls.append(
                ToolCall(
                    id=fn_call.call_id or fn_call.name,
                    name=fn_call.name,
                    arguments=_parse_arguments(fn_call.arguments),
                    result=result,
                    error=error,
                    source=ToolCallSource.LIVEKIT,
                )
            )

    async def run_turn(self, user_query: str) -> AgentResponse:
        await self._ensure_started()
        assert self._session is not None
        assert self._audio_input is not None
        assert self._audio_output is not None

        self._tool_calls = []
        self._audio_output.reset()
        audio = self._perturb(await self._tts.synthesize(user_query))
        self._audio_input.push(audio)
        # AgentSession only adds flush silence when no audio input is attached.
        # arksim attaches an in-memory input, so explicitly send silence through
        # it to let streaming STT and VAD emit a final transcript before commit.
        self._audio_input.push(
            AudioBuffer(
                samples=np.zeros(
                    int(audio.sample_rate * _STT_FLUSH_DURATION_S * audio.num_channels),
                    dtype=np.float32,
                ),
                sample_rate=audio.sample_rate,
                num_channels=audio.num_channels,
            )
        )
        await self._audio_input.wait_drained()

        transcript_future = self._session.commit_user_turn(
            transcript_timeout=_TRANSCRIPT_TIMEOUT_S,
            stt_flush_duration=_STT_FLUSH_DURATION_S,
        )
        try:
            await asyncio.wait_for(
                transcript_future,
                timeout=_TRANSCRIPT_TIMEOUT_S + _STT_FLUSH_DURATION_S + 1,
            )
            await asyncio.wait_for(
                self._audio_output.turn_done.wait(), timeout=_TURN_TIMEOUT_S
            )
            await asyncio.wait_for(
                self._session.wait_for_idle(), timeout=_TURN_TIMEOUT_S
            )
        except (TimeoutError, asyncio.TimeoutError) as exc:
            await self.close()
            raise TimeoutError(
                f"LiveKit agent did not complete its audio turn for: {user_query!r}"
            ) from exc

        captured = self._audio_output.captured_audio()
        if captured is None:
            await self.close()
            raise RuntimeError(f"LiveKit agent produced no audio for: {user_query!r}")
        reply = await self._stt.transcribe(captured)
        return AgentResponse(content=reply, tool_calls=list(self._tool_calls))

    def _perturb(self, audio: AudioBuffer) -> AudioBuffer:
        # v2 seam: accent / background-noise / volume perturbation hooks here.
        return audio

    async def get_chat_id(self) -> str:
        return self._chat_id

    async def close(self) -> None:
        audio_input = self._audio_input
        session = self._session
        self._session = None
        self._audio_input = None
        self._audio_output = None
        if audio_input is not None:
            audio_input.close()
        if session is not None:
            await session.aclose()
