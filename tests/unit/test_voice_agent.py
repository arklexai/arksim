# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

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
    class StubDriver:
        async def run_turn(self, q: str) -> AgentResponse:
            return AgentResponse(content=f"heard: {q}")

        async def get_chat_id(self) -> str:
            return "chat-1"

        async def close(self) -> None:
            return None

    import arksim.simulation_engine.agent.clients.voice as mod

    monkeypatch.setattr(mod, "_build_driver_for", lambda cfg: StubDriver())

    agent = VoiceAgent(_cfg())
    resp = await agent.execute("hello")
    assert isinstance(resp, AgentResponse)
    assert resp.content == "heard: hello"
    assert await agent.get_chat_id() == "chat-1"
    await agent.close()


def test_factory_builds_voice_agent() -> None:
    from arksim.simulation_engine.agent.factory import create_agent

    agent = create_agent(_cfg())
    assert isinstance(agent, VoiceAgent)
