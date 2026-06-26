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


def test_resample_downsamples_to_target_rate() -> None:
    pytest.importorskip("pipecat")
    import numpy as np

    from arksim.integrations.pipecat import _resample
    from arksim.speech.types import AudioBuffer

    out = _resample(AudioBuffer(np.zeros(2400, dtype=np.float32), 24000), 16000)
    assert out.sample_rate == 16000
    assert abs(len(out.samples) - 1600) <= 1
