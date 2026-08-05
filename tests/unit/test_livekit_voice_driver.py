# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio

import pytest


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"order_id": "A123"}, {"order_id": "A123"}),
        ('{"order_id": "A123"}', {"order_id": "A123"}),
        ("not-json", {"_value": "not-json"}),
        ("[1, 2]", {"_value": [1, 2]}),
        (7, {"_value": 7}),
    ],
)
def test_parse_arguments_normalizes_livekit_payloads(
    raw: object, expected: dict[str, object]
) -> None:
    pytest.importorskip("livekit.agents")

    from arksim.integrations.livekit import _parse_arguments

    assert _parse_arguments(raw) == expected


async def test_driver_validates_factory_contract_before_starting() -> None:
    pytest.importorskip("livekit.agents")

    from arksim.integrations.livekit import LiveKitVoiceDriver

    driver = LiveKitVoiceDriver(lambda: "not-a-tuple", tts=object(), stt=object())

    with pytest.raises(TypeError, match=r"return \(AgentSession, Agent\)"):
        await driver.run_turn("hello")


async def test_driver_requires_manual_turn_detection() -> None:
    pytest.importorskip("livekit.agents")
    from livekit.agents import Agent, AgentSession

    from arksim.integrations.livekit import LiveKitVoiceDriver

    session = AgentSession(vad=None)
    agent = Agent(instructions="Be concise.")
    driver = LiveKitVoiceDriver(lambda: (session, agent), tts=object(), stt=object())

    with pytest.raises(ValueError, match="manual turn detection"):
        await driver.run_turn("hello")


async def test_driver_runs_audio_turn_and_captures_tool_call() -> None:
    pytest.importorskip("livekit.agents")
    import numpy as np
    from livekit.agents import Agent, AgentSession, TurnHandlingOptions
    from livekit.agents.llm import FunctionCall, FunctionCallOutput
    from livekit.agents.voice.events import FunctionToolsExecutedEvent

    from arksim.integrations.livekit import LiveKitVoiceDriver
    from arksim.simulation_engine.tool_types import ToolCallSource
    from arksim.speech.types import AudioBuffer

    class FakeSession(AgentSession):
        def __init__(self) -> None:
            super().__init__(
                vad=None,
                turn_handling=TurnHandlingOptions(turn_detection="manual"),
            )
            self.closed = False
            self.frames: list[object] = []
            self.committed_frames: list[object] = []
            self.consumer: asyncio.Task[None] | None = None

        async def start(
            self,
            agent: Agent,
            *,
            record: bool = False,
        ) -> None:
            async def consume_audio() -> None:
                assert self.input.audio is not None
                async for frame in self.input.audio:
                    self.frames.append(frame)

            self.consumer = asyncio.create_task(consume_audio())

        async def wait_for_idle(self) -> object:
            return object()

        def commit_user_turn(
            self,
            *,
            transcript_timeout: float = 2.0,
            stt_flush_duration: float = 2.0,
            skip_reply: bool = False,
        ) -> asyncio.Future[str]:
            self.committed_frames = list(self.frames)
            future: asyncio.Future[str] = asyncio.Future()

            async def respond() -> None:
                assert self.output.audio is not None
                assert self.frames
                frame = self.frames[0]
                self.frames.clear()
                self.emit(
                    "function_tools_executed",
                    FunctionToolsExecutedEvent(
                        function_calls=[
                            FunctionCall(
                                call_id="call-1",
                                name="lookup_order",
                                arguments='{"order_id": "A123"}',
                            )
                        ],
                        function_call_outputs=[
                            FunctionCallOutput(
                                call_id="call-1",
                                name="lookup_order",
                                output="shipped",
                                is_error=False,
                            )
                        ],
                    ),
                )
                await self.output.audio.capture_frame(frame)
                self.output.audio.flush()
                future.set_result("hello agent")

            asyncio.create_task(respond())
            return future

        async def aclose(self) -> None:
            self.closed = True
            if self.consumer is not None:
                await self.consumer

    class FakeTTS:
        async def synthesize(self, text: str) -> AudioBuffer:
            return AudioBuffer(np.zeros(960, dtype=np.float32), 24000)

    class FakeSTT:
        async def transcribe(self, audio: AudioBuffer) -> str:
            assert audio.sample_rate == 24000
            assert len(audio.samples) == 480
            return "agent reply"

    session = FakeSession()
    agent = Agent(instructions="You are a concise support agent.")
    driver = LiveKitVoiceDriver(lambda: (session, agent), tts=FakeTTS(), stt=FakeSTT())
    try:
        response = await driver.run_turn("where is my order?")
        assert response.content == "agent reply"
        assert sum(
            frame.duration for frame in session.committed_frames
        ) == pytest.approx(1.04)
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "call-1"
        assert response.tool_calls[0].arguments == {"order_id": "A123"}
        assert response.tool_calls[0].result == "shipped"
        assert response.tool_calls[0].source is ToolCallSource.LIVEKIT

        second = await driver.run_turn("thanks")
        assert second.content == "agent reply"
    finally:
        await driver.close()
    assert session.closed


def test_rtc_frames_preserve_pcm_and_chunk_duration() -> None:
    pytest.importorskip("livekit.agents")
    import numpy as np

    from arksim.integrations.livekit import _rtc_frames
    from arksim.speech.types import AudioBuffer, pcm16_bytes

    audio = AudioBuffer(np.linspace(-0.5, 0.5, 1200, dtype=np.float32), 24000)
    frames = list(_rtc_frames(audio))

    assert [frame.samples_per_channel for frame in frames] == [480, 480, 240]
    assert b"".join(bytes(frame.data) for frame in frames) == pcm16_bytes(audio)


async def test_driver_closes_session_after_turn_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("livekit.agents")
    import numpy as np
    from livekit.agents import Agent, AgentSession, TurnHandlingOptions

    import arksim.integrations.livekit as livekit_integration
    from arksim.speech.types import AudioBuffer

    class SilentSession(AgentSession):
        def __init__(self) -> None:
            super().__init__(
                vad=None,
                turn_handling=TurnHandlingOptions(turn_detection="manual"),
            )
            self.closed = False
            self.consumer: asyncio.Task[None] | None = None

        async def start(self, agent: Agent, *, record: bool = False) -> None:
            async def consume_audio() -> None:
                assert self.input.audio is not None
                async for _ in self.input.audio:
                    pass

            self.consumer = asyncio.create_task(consume_audio())

        async def wait_for_idle(self) -> object:
            return object()

        def commit_user_turn(
            self,
            *,
            transcript_timeout: float = 2.0,
            stt_flush_duration: float = 2.0,
            skip_reply: bool = False,
        ) -> asyncio.Future[str]:
            future: asyncio.Future[str] = asyncio.Future()
            future.set_result("hello")
            return future

        async def aclose(self) -> None:
            self.closed = True
            if self.consumer is not None:
                await self.consumer

    class FakeTTS:
        async def synthesize(self, text: str) -> AudioBuffer:
            return AudioBuffer(np.zeros(160), 16000)

    class FakeSTT:
        async def transcribe(self, audio: AudioBuffer) -> str:
            return "unused"

    monkeypatch.setattr(livekit_integration, "_TURN_TIMEOUT_S", 0.01)
    session = SilentSession()
    agent = Agent(instructions="Be concise.")
    driver = livekit_integration.LiveKitVoiceDriver(
        lambda: (session, agent), tts=FakeTTS(), stt=FakeSTT()
    )

    with pytest.raises(TimeoutError, match="did not complete"):
        await driver.run_turn("hello")

    assert session.closed
    assert driver._session is None
