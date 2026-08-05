# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="pipecat requires 3.11+"
)


async def test_driver_runs_one_turn_with_tool_call() -> None:
    pytest.importorskip("pipecat")
    import numpy as np

    from arksim.integrations.pipecat import (
        PipecatVoiceDriver,
        _build_echo_stages_for_test,
    )
    from arksim.simulation_engine.tool_types import ToolCallSource
    from arksim.speech.types import AudioBuffer

    class FakeTTS:
        async def synthesize(self, text: str) -> AudioBuffer:
            return AudioBuffer(np.zeros(2400, dtype=np.float32), 24000)

    class FakeSTT:
        async def transcribe(self, audio: AudioBuffer) -> str:
            return "agent reply"

    driver = PipecatVoiceDriver(
        _build_echo_stages_for_test, tts=FakeTTS(), stt=FakeSTT()
    )
    try:
        resp = await driver.run_turn("hello agent")
        assert resp.content == "agent reply"
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "lookup_order"
        assert resp.tool_calls[0].source is ToolCallSource.PIPECAT
        # second turn proves the worker is reused
        resp2 = await driver.run_turn("again")
        assert resp2.content == "agent reply"
    finally:
        await driver.close()


async def test_driver_drives_vad_gated_stt() -> None:
    """The driver must emit VAD speaking frames so a segmented STT transcribes.

    Guards against regressions where the VAD bracket is dropped: a real pipecat
    SegmentedSTTService only runs on VADUserStoppedSpeakingFrame, so without it
    the turn would time out. No models needed.
    """
    pytest.importorskip("pipecat")
    import numpy as np
    from pipecat.frames.frames import (
        Frame,
        InputAudioRawFrame,
        TTSAudioRawFrame,
        TTSStartedFrame,
        TTSStoppedFrame,
        VADUserStoppedSpeakingFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    from arksim.integrations.pipecat import PipecatVoiceDriver
    from arksim.speech.types import AudioBuffer

    class VadGatedStt(FrameProcessor):
        def __init__(self) -> None:
            super().__init__()
            self._got_audio = False

        async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
            await super().process_frame(frame, direction)
            if isinstance(frame, InputAudioRawFrame):
                self._got_audio = True
                await self.push_frame(frame, direction)
            elif isinstance(frame, VADUserStoppedSpeakingFrame) and self._got_audio:
                self._got_audio = False
                await self.push_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)
                await self.push_frame(
                    TTSAudioRawFrame(
                        audio=b"\x00\x00" * 800, sample_rate=16000, num_channels=1
                    ),
                    FrameDirection.DOWNSTREAM,
                )
                await self.push_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
                await self.push_frame(frame, direction)
            else:
                await self.push_frame(frame, direction)

    class FakeTTS:
        async def synthesize(self, text: str) -> AudioBuffer:
            return AudioBuffer(np.zeros(2400, dtype=np.float32), 24000)

    class FakeSTT:
        async def transcribe(self, audio: AudioBuffer) -> str:
            return "vad reply"

    driver = PipecatVoiceDriver(
        lambda: (None, [VadGatedStt()]), tts=FakeTTS(), stt=FakeSTT()
    )
    try:
        resp = await driver.run_turn("hi")
        assert resp.content == "vad reply"
    finally:
        await driver.close()


async def test_driver_closes_pipeline_after_turn_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pipecat")
    import numpy as np
    from pipecat.frames.frames import Frame
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    import arksim.integrations.pipecat as pipecat_integration
    from arksim.speech.types import AudioBuffer

    class FakeTTS:
        async def synthesize(self, text: str) -> AudioBuffer:
            return AudioBuffer(np.zeros(160), 16000)

    class FakeSTT:
        async def transcribe(self, audio: AudioBuffer) -> str:
            return "unused"

    class SilentStage(FrameProcessor):
        async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)

    monkeypatch.setattr(pipecat_integration, "_TURN_TIMEOUT_S", 0.01)
    driver = pipecat_integration.PipecatVoiceDriver(
        lambda: [SilentStage()], tts=FakeTTS(), stt=FakeSTT()
    )

    with pytest.raises(TimeoutError, match="produced no audio"):
        await driver.run_turn("hello")

    assert driver._task is None
    assert driver._runner_run is None


def test_resample_downsamples_to_target_rate() -> None:
    pytest.importorskip("pipecat")
    import numpy as np

    from arksim.integrations.pipecat import _resample
    from arksim.speech.types import AudioBuffer

    out = _resample(AudioBuffer(np.zeros(2400, dtype=np.float32), 24000), 16000)
    assert out.sample_rate == 16000
    assert abs(len(out.samples) - 1600) <= 1


def test_resample_downmixes_stereo_even_at_target_rate() -> None:
    pytest.importorskip("pipecat")
    import numpy as np

    from arksim.integrations.pipecat import _resample
    from arksim.speech.types import AudioBuffer

    stereo = AudioBuffer(
        np.array([1.0, -1.0, 0.5, 0.5], dtype=np.float32),
        sample_rate=16000,
        num_channels=2,
    )

    out = _resample(stereo, 16000)

    assert out.num_channels == 1
    assert out.samples == pytest.approx([0.0, 0.5])


def test_resample_rejects_invalid_target_rate() -> None:
    pytest.importorskip("pipecat")
    import numpy as np

    from arksim.integrations.pipecat import _resample
    from arksim.speech.types import AudioBuffer

    with pytest.raises(ValueError, match="target_rate"):
        _resample(AudioBuffer(np.zeros(10), 16000), 0)
