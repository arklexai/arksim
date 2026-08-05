# LiveKit voice agent

Evaluate a [LiveKit Agents](https://docs.livekit.io/agents/) voice agent with
arksim's native `voice` agent type.

## How it works

arksim drives the agent through its real speech stack:

```text
simulated user TEXT -> [arksim TTS] -> AUDIO -> LiveKit STT -> LLM -> TTS
  -> AUDIO -> [arksim STT] -> TEXT -> evaluator
```

- `agent.py:build()` returns the agent's own `(AgentSession, Agent)`, including
  its real STT, LLM, TTS, and tools.
- arksim installs in-memory audio input/output on the session. No microphone,
  speaker, or LiveKit room is needed for an evaluation run.
- Tool calls are captured from LiveKit's `function_tools_executed` event and
  stored with source `livekit`.
- The `AgentSession` must use manual turn detection. arksim commits each audio
  turn after the simulated user's synthesized speech has been injected.

This exercises the LiveKit Agents speech and orchestration pipeline. It does
not test WebRTC room transport, packet loss, or client-side audio processing.

## Setup

```bash
pip install 'arksim[livekit-voice]' 'livekit-agents[openai]>=1.5.9,<2.0'
export OPENAI_API_KEY=...
```

`arksim[livekit-voice]` installs LiveKit Agents plus the arksim-side Kokoro TTS
and faster-whisper STT.
The LiveKit OpenAI plugin supplies the agent's own STT, LLM, and TTS services.
This example currently targets Python 3.10–3.12 because Kokoro requires Python
<3.13. Use a custom ArkSim-side speech provider if your environment differs.

## Run

From this directory:

```bash
arksim simulate-evaluate ./config.yaml
```

The evaluation path uses LiveKit's `AgentSession` orchestration with in-memory
audio I/O. `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` are not
needed unless you separately run the agent through a hosted LiveKit room.

## Files

- `agent.py` - LiveKit `AgentSession` and `Agent` behind `build()`.
- `config.yaml` - voice framework, factory pointer, and arksim speech providers.
- `scenarios.json` - phone-support scenarios.
