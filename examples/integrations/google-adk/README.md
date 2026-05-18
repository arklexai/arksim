# Google ADK Integration

A [Google ADK](https://github.com/google/adk-python) `LlmAgent` with two mock tools (`lookup_order`, `book_table`), wired to arksim through `ArksimADKPlugin` so every tool call is captured in `simulation.json`.

## Setup

1. Install arksim with the Google ADK extra:

   ```bash
   pip install 'arksim[google-adk]'
   ```

2. Set your API key:

   ```bash
   export GOOGLE_API_KEY="<your-key>"
   ```

## Run

From this example directory:

```bash
arksim simulate-evaluate config.yaml
```

## How it works

`ArksimADKPlugin` extends ADK's `BasePlugin` and overrides `after_tool_callback` to emit a `ToolCall` after every tool invocation. The agent registers it on the runner via `InMemoryRunner(plugins=[plugin])` so one instance covers the `LlmAgent` and any sub-agents it spawns. Before each turn the simulator binds `conversation_id`, `turn_id`, and the trace receiver into contextvars, so when ADK invokes the plugin callback it submits records that land in the `tool_calls` field of every turn in `results/simulation/simulation.json`.

> Note: ADK's `InMemoryRunner` historically did not invoke plugin callbacks in some releases (see [adk-python issue #4464](https://github.com/google/adk-python/issues/4464)). If you find empty `tool_calls` arrays, swap `InMemoryRunner` for the production `Runner` with explicit `session_service`, `artifact_service`, and `memory_service` arguments.

## Expected output

After running the example, each turn that invoked a tool contains entries like this in `results/simulation/simulation.json`:

```json
"tool_calls": [
  {
    "name": "lookup_order",
    "arguments": {"order_id": "ORD-1001"},
    "result": "{'result': 'Order ORD-1001: shipped, arrives Tuesday.'}",
    "source": "google_adk"
  },
  {
    "name": "book_table",
    "arguments": {"party_size": 4, "time": "7pm"},
    "result": "{'result': 'Booked table for 4 at 7pm.'}",
    "source": "google_adk"
  }
]
```

## Files

| File              | Description                                |
| ----------------- | ------------------------------------------ |
| `custom_agent.py` | ADK `LlmAgent` + `ArksimADKPlugin`         |
| `tools.py`        | `lookup_order` and `book_table` mock tools |
| `config.yaml`     | Simulator configuration                    |
| `scenarios.json`  | Two scenarios, one per tool                |
