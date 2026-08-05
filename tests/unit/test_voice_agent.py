# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.clients.voice import VoiceAgent
from arksim.simulation_engine.tool_types import AgentResponse


def _cfg(framework: str = "pipecat") -> AgentConfig:
    return AgentConfig.model_validate(
        {
            "agent_name": "v",
            "agent_type": "voice",
            "voice_config": {"framework": framework, "agent_factory": "os.path:join"},
        }
    )


async def test_voice_agent_delegates_to_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    close_calls = 0

    class StubDriver:
        async def run_turn(self, q: str) -> AgentResponse:
            return AgentResponse(content=f"heard: {q}")

        async def get_chat_id(self) -> str:
            return "chat-1"

        async def close(self) -> None:
            nonlocal close_calls
            close_calls += 1

    import arksim.simulation_engine.agent.clients.voice as mod

    monkeypatch.setattr(mod, "_build_driver_for", lambda cfg: StubDriver())

    agent = VoiceAgent(_cfg())
    resp = await agent.execute("hello")
    assert isinstance(resp, AgentResponse)
    assert resp.content == "heard: hello"
    assert await agent.get_chat_id() == "chat-1"
    await agent.close()
    await agent.close()
    assert close_calls == 1


def test_factory_builds_voice_agent() -> None:
    from arksim.simulation_engine.agent.factory import create_agent

    agent = create_agent(_cfg())
    assert isinstance(agent, VoiceAgent)


@pytest.mark.parametrize(
    ("framework", "module_name", "driver_name"),
    [
        ("pipecat", "arksim.integrations.pipecat", "PipecatVoiceDriver"),
        ("livekit", "arksim.integrations.livekit", "LiveKitVoiceDriver"),
    ],
)
def test_build_driver_selects_framework_without_eager_optional_imports(
    monkeypatch: pytest.MonkeyPatch,
    framework: str,
    module_name: str,
    driver_name: str,
) -> None:
    import arksim.simulation_engine.agent.clients.voice as voice_module
    import arksim.speech as speech_module

    created: dict[str, object] = {}

    class StubDriver:
        def __init__(self, factory: object, *, tts: object, stt: object) -> None:
            created.update(factory=factory, tts=tts, stt=stt)

    integration_module = ModuleType(module_name)
    setattr(integration_module, driver_name, StubDriver)
    monkeypatch.setitem(sys.modules, module_name, integration_module)

    factory = object()
    monkeypatch.setattr(voice_module, "load_callable", lambda pointer: factory)
    monkeypatch.setattr(speech_module, "create_tts", lambda config: "tts")
    monkeypatch.setattr(speech_module, "create_stt", lambda config: "stt")

    driver = voice_module._build_driver_for(_cfg(framework))

    assert isinstance(driver, StubDriver)
    assert created == {"factory": factory, "tts": "tts", "stt": "stt"}


def test_build_driver_rejects_missing_voice_config() -> None:
    import arksim.simulation_engine.agent.clients.voice as voice_module

    config = SimpleNamespace(voice_config=None)
    with pytest.raises(ValueError, match="requires voice_config"):
        voice_module._build_driver_for(config)
