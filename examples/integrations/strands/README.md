# Strands Agents Integration

A Strands `Agent` with two mock tools (`lookup_order`, `book_table`), wired to arksim through `ArksimStrandsHookProvider` so every tool call is captured in `simulation.json`.

## Setup

1. Install arksim with the Strands extra plus the Strands OpenAI provider:

   ```bash
   pip install 'arksim[strands]'
   pip install 'strands-agents[openai]'
   ```

   Strands defaults to AWS Bedrock. This example uses `OpenAIModel` so the wiring matches the rest of the integration examples (single `OPENAI_API_KEY` env var). Note: the quotes around `'strands-agents[openai]'` are required on zsh; the brackets are shell glob characters.

2. Set your API key:

   ```bash
   export OPENAI_API_KEY="<your-key>"
   ```

If `gpt-5.1` isn't available in your account, set `OPENAI_MODEL=<your-model>` and re-run, or edit `config.yaml` directly.

## Run

From this example directory:

```bash
arksim simulate-evaluate config.yaml
```

## How it works

`ArksimStrandsHookProvider` implements the Strands `HookProvider` protocol and registers a callback on `AfterToolCallEvent`. The agent passes one provider instance via `hooks=[ArksimStrandsHookProvider()]` at construction. Before each agent turn the simulator binds `conversation_id`, `turn_id`, and the trace receiver into contextvars, so when Strands fires `AfterToolCallEvent` (with `tool_use["name"]`, `tool_use["toolUseId"]`, `tool_use["input"]`, and either `result` or `exception`) the provider emits a `ToolCall` record on the active turn. Successes carry the `result`, failures forward the exception type and message into `error`.

## Expected output

After running the example, each turn that invoked a tool contains entries like this in `results/simulation/simulation.json`:

```json
"tool_calls": [
  {
    "name": "lookup_order",
    "arguments": {"order_id": "ORD-1001"},
    "result": "Order ORD-1001: shipped, arrives Tuesday.",
    "source": "strands"
  },
  {
    "name": "book_table",
    "arguments": {"party_size": 4, "time": "7pm"},
    "result": "Booked table for 4 at 7pm.",
    "source": "strands"
  }
]
```

## Files

| File              | Description                                          |
| ----------------- | ---------------------------------------------------- |
| `custom_agent.py` | Strands `Agent` + `ArksimStrandsHookProvider`        |
| `tools.py`        | `lookup_order` and `book_table` mock tools           |
| `config.yaml`     | Simulator configuration                              |
| `scenarios.json`  | Two scenarios, one per tool                          |
