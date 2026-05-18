# CrewAI Integration

A single-agent [CrewAI](https://github.com/crewAIInc/crewAI) `Crew` with two mock tools (`lookup_order`, `book_table`), wired to arksim through `ArksimCrewEventListener` so every tool call is captured in `simulation.json`.

## Setup

1. Install arksim with the CrewAI extra:

   ```bash
   pip install 'arksim[crewai]'
   ```

2. Set your API key:

   ```bash
   export OPENAI_API_KEY="<your-key>"
   ```

## Run

From this example directory:

```bash
arksim simulate-evaluate config.yaml
```

## How it works

`ArksimCrewEventListener` extends CrewAI's `BaseEventListener`, which registers its handlers eagerly against the global `crewai_event_bus` from its constructor. Instantiating one listener per agent is enough; no `Crew(event_listeners=[...])` wiring is required. Before each turn the simulator binds `conversation_id`, `turn_id`, and the trace receiver into contextvars, so when the listener receives `ToolUsageFinishedEvent` or `ToolUsageErrorEvent` from inside `kickoff_async` it emits `ToolCall` records that land in the `tool_calls` field of every turn in `results/simulation/simulation.json`.

## Expected output

After running the example, each turn that invoked a tool contains entries like this in `results/simulation/simulation.json`:

```json
"tool_calls": [
  {
    "name": "lookup_order",
    "arguments": {"order_id": "ORD-1001"},
    "result": "Order ORD-1001: shipped, arrives Tuesday.",
    "source": "crewai"
  },
  {
    "name": "book_table",
    "arguments": {"party_size": 4, "time": "7pm"},
    "result": "Booked table for 4 at 7pm.",
    "source": "crewai"
  }
]
```

## Files

| File              | Description                                |
| ----------------- | ------------------------------------------ |
| `custom_agent.py` | CrewAI `Crew` + `ArksimCrewEventListener`  |
| `tools.py`        | `lookup_order` and `book_table` mock tools |
| `config.yaml`     | Simulator configuration                    |
| `scenarios.json`  | Two scenarios, one per tool                |
