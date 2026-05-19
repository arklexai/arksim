# LiveKit Agents Integration

A LiveKit Agents text-mode session with two mock tools (`lookup_order`, `book_table`), wired to arksim through `ArksimLiveKitHandler` so every tool call is captured in `simulation.json`.

## Setup

1. Install arksim with the LiveKit extra:

   ```bash
   pip install 'arksim[livekit]'
   ```

2. Set your credentials:

   ```bash
   export OPENAI_API_KEY="<your-key>"
   export LIVEKIT_API_KEY="<your-livekit-key>"
   export LIVEKIT_API_SECRET="<your-livekit-secret>"
   ```

   The LiveKit credentials are required by `livekit.agents.inference.LLM`, which proxies LLM calls through LiveKit Cloud. Free credentials are available at <https://cloud.livekit.io>.

## Run

From this example directory:

```bash
arksim simulate-evaluate config.yaml
```

## How it works

LiveKit Agents is voice-first, but `AgentSession.run(user_input=..., input_modality="text")` exposes a text-in / text-out path that needs no audio room. The agent starts the session without a `room` argument so no RTC plumbing is set up. `ArksimLiveKitHandler` subscribes to the `function_tools_executed` event on `AgentSession`; LiveKit emits one event per parallel tool-call batch, and the handler turns each call into a `ToolCall` record on the active turn. Before each agent turn the simulator binds `conversation_id`, `turn_id`, and the trace receiver into contextvars, so the captured calls land in the `tool_calls` field of every turn in `results/simulation/simulation.json`.

## Expected output

After running the example, each turn that invoked a tool contains entries like this in `results/simulation/simulation.json`:

```json
"tool_calls": [
  {
    "name": "lookup_order",
    "arguments": {"order_id": "ORD-1001"},
    "result": "Order ORD-1001: shipped, arrives Tuesday.",
    "source": "livekit"
  },
  {
    "name": "book_table",
    "arguments": {"party_size": 4, "time": "7pm"},
    "result": "Booked table for 4 at 7pm.",
    "source": "livekit"
  }
]
```

## Files

| File              | Description                                       |
| ----------------- | ------------------------------------------------- |
| `custom_agent.py` | LiveKit `AgentSession` + `ArksimLiveKitHandler`   |
| `tools.py`        | `lookup_order` and `book_table` mock tools        |
| `config.yaml`     | Simulator configuration                           |
| `scenarios.json`  | Two scenarios, one per tool                       |
