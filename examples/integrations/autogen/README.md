# AutoGen Integration

A [Microsoft AutoGen](https://github.com/microsoft/autogen) `AssistantAgent` with two mock tools (`lookup_order`, `book_table`) whose calls are exported as OpenTelemetry spans and captured by arksim's built-in OTLP receiver, landing in `simulation.json`.

## Setup

1. Install arksim with the OTel extra (required for the trace receiver this example uses):

   ```bash
   pip install 'arksim[otel]'
   ```

2. Install AutoGen and the OTel exporter:

   ```bash
   pip install autogen-agentchat autogen-ext[openai]
   pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
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

AutoGen does not emit `gen_ai.tool.*` spans natively when `AssistantAgent.on_messages` runs directly (no runtime), so `tools.py` wraps each tool body in an OpenTelemetry span that follows the OTel GenAI semantic conventions. The agent constructs a `TracerProvider` that exports through `OTLPSpanExporter` to `127.0.0.1:4318/v1/traces`, which is where arksim's built-in OTLP receiver listens when `trace_receiver.enabled: true` is set in `config.yaml`. A small `_ArksimRoutingProcessor` reads `arksim.turn_id` from arksim's contextvar and stamps it on every span; `arksim.conversation_id` is pinned to the agent's `chat_id` via a `Resource` attribute. No extra arksim adapter is needed: the receiver decodes the OTLP payload, picks the routing attributes, and lands each tool span as a `ToolCall` on the correct turn in `results/simulation/simulation.json`.

## Expected output

After running the example, each turn that invoked a tool contains entries like this in `results/simulation/simulation.json`:

```json
"tool_calls": [
  {
    "name": "lookup_order",
    "arguments": {"order_id": "ORD-1001"},
    "result": "Order ORD-1001: shipped, arrives Tuesday.",
    "source": "otel_trace"
  },
  {
    "name": "book_table",
    "arguments": {"party_size": 4, "time": "7pm"},
    "result": "Booked table for 4 at 7pm.",
    "source": "otel_trace"
  }
]
```

## Files

| File              | Description                                              |
| ----------------- | -------------------------------------------------------- |
| `custom_agent.py` | AutoGen `AssistantAgent` + OTLP tracer provider          |
| `tools.py`        | `lookup_order` and `book_table` with OTel span wrappers  |
| `config.yaml`     | Simulator configuration with `trace_receiver` enabled    |
| `scenarios.json`  | Two scenarios, one per tool                              |
