# Pipecat voice agent

Evaluate a [Pipecat](https://github.com/pipecat-ai/pipecat) voice agent with
arksim's native `voice` agent type.

## How it works

arksim drives your agent through its **real speech stack**:

```
simulated user TEXT -> [arksim TTS] -> AUDIO -> your agent (ASR -> LLM -> TTS) -> AUDIO -> [arksim STT] -> TEXT -> evaluator
```

- `agent.py:build()` returns your agent's own `(LLMContext, [stages])`, including
  its real STT, LLM, and TTS services. Your agent code does not import arksim.
- arksim synthesizes each simulated-user turn to audio (`voice_config.tts`,
  default local Kokoro), injects it, captures the agent's spoken reply, and
  transcribes it back to text (`voice_config.stt`, default local faster-whisper)
  for the existing evaluator.
- Tool calls your agent makes are captured automatically (source `pipecat`).

Only the agent's own speech stack is under test. Accent, background-noise, and
volume perturbation are a planned follow-on; this example uses a clean voice.

## Setup

```bash
pip install 'arksim[voice]' 'pipecat-ai[openai,whisper,silero]'
export OPENAI_API_KEY=...
```

`arksim[voice]` installs the arksim-side TTS/STT (Kokoro + faster-whisper). The
pipecat extras install the agent's own STT/LLM/TTS services used in `agent.py`.

## Run

```bash
arksim simulate-evaluate --config ./config.yaml
```

## Try it without an API key

`smoke_local.py` runs the full audio loop with real local ASR + TTS and a
deterministic brain (no LLM key, no `OPENAI_API_KEY`):

```bash
pip install 'arksim[voice]'
python smoke_local.py
```

## Files

- `agent.py` - your Pipecat voice pipeline behind a zero-arg `build()`.
- `config.yaml` - `agent_type: voice`, framework, factory pointer, and the
  arksim-side `tts`/`stt` providers.
- `scenarios.json` - phone-support scenarios.
