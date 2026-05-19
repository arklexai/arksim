# Smolagents Integration

A [Smolagents](https://github.com/huggingface/smolagents) (Hugging Face) `CodeAgent` with two mock tools (`lookup_order`, `book_table`), wired to arksim through `ArksimSmolagentsCallback` so every tool call is captured in `simulation.json`.

## Setup

1. Install arksim with the Smolagents extra:

   ```bash
   pip install 'arksim[smolagents]'
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

`ArksimSmolagentsCallback` is a smolagents step callback. The agent registers one instance via `CodeAgent(step_callbacks=[self._callback])`, and smolagents invokes the callback after each `ActionStep`, where the tool calls live. Before each agent turn the simulator binds `conversation_id`, `turn_id`, and the trace receiver into contextvars, so when smolagents fires the callback the recorded tool calls land in the `tool_calls` field of every turn in `results/simulation/simulation.json`. Other step types (`PlanningStep`, `TaskStep`, `FinalAnswerStep`, `SystemPromptStep`) are ignored.

## Expected output

After running the example, each turn that invoked a tool contains entries like this in `results/simulation/simulation.json`:

```json
"tool_calls": [
  {
    "name": "lookup_order",
    "arguments": {"order_id": "ORD-1001"},
    "result": "Order ORD-1001: shipped, arrives Tuesday.",
    "source": "smolagents"
  },
  {
    "name": "book_table",
    "arguments": {"party_size": 4, "time": "7pm"},
    "result": "Booked table for 4 at 7pm.",
    "source": "smolagents"
  }
]
```

## Files

| File              | Description                                         |
| ----------------- | --------------------------------------------------- |
| `custom_agent.py` | Smolagents `CodeAgent` + `ArksimSmolagentsCallback` |
| `tools.py`        | `lookup_order` and `book_table` mock tools          |
| `config.yaml`     | Simulator configuration                             |
| `scenarios.json`  | Two scenarios, one per tool                         |
