# OpenAI Agents SDK Integration

An [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) `Agent` with two mock tools (`lookup_order`, `book_table`), wired to arksim through `ArksimTracingProcessor` so every tool call is captured in `simulation.json`.

## Setup

1. Install arksim with the OTel extra (required for the trace receiver this example uses):

   ```bash
   pip install 'arksim[otel]'
   ```

2. Install the OpenAI Agents SDK:

   ```bash
   pip install openai-agents
   ```

3. Set your API key:

   ```bash
   export OPENAI_API_KEY="<your-key>"
   ```

## Run

From this example directory:

```bash
arksim simulate-evaluate config.yaml
```

## How it works

`ArksimTracingProcessor` implements the OpenAI Agents SDK's `TracingProcessor` interface. The agent registers one instance at module load via `add_trace_processor(ArksimTracingProcessor())`; the simulator caches modules by file path so the registration runs exactly once. Each tool is wrapped with `@function_tool` so the SDK emits a `FunctionSpanData` entry when it fires. Before each agent turn the simulator binds `conversation_id`, `turn_id`, and the trace receiver into contextvars, so when the SDK calls `on_span_end` the processor reads that context, converts the span into a `ToolCall`, and injects it into the receiver's buffer. The captured calls land in the `tool_calls` field of every turn in `results/simulation/simulation.json`.

## Expected output

After running the example, each turn that invoked a tool contains entries like this in `results/simulation/simulation.json`:

```json
"tool_calls": [
  {
    "name": "lookup_order",
    "arguments": {"order_id": "ORD-1001"},
    "result": "Order ORD-1001: shipped, arrives Tuesday.",
    "source": "openai_agents"
  },
  {
    "name": "book_table",
    "arguments": {"party_size": 4, "time": "7pm"},
    "result": "Booked table for 4 at 7pm.",
    "source": "openai_agents"
  }
]
```

## Files

| File              | Description                                              |
| ----------------- | -------------------------------------------------------- |
| `custom_agent.py` | OpenAI Agents SDK `Agent` + `ArksimTracingProcessor`     |
| `tools.py`        | `lookup_order` and `book_table` mock tools               |
| `config.yaml`     | Simulator configuration with `trace_receiver` enabled    |
| `scenarios.json`  | Two scenarios, one per tool                              |
