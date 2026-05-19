# Pydantic AI Integration

A [Pydantic AI](https://github.com/pydantic/pydantic-ai) `Agent` with two mock tools (`lookup_order`, `book_table`) whose calls are exported as OpenTelemetry spans and captured by arksim's built-in OTLP receiver, landing in `simulation.json`.

## Setup

1. Install Pydantic AI and the OTel exporter:

   ```bash
   pip install pydantic-ai
   pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
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

Pydantic AI emits `gen_ai.tool.*` spans natively for every tool invocation when the `Agent` is constructed with `instrument=True`. The agent builds a `TracerProvider` that exports through `OTLPSpanExporter` to `127.0.0.1:4318/v1/traces` and passes it to `InstrumentationSettings(tracer_provider=...)` so Pydantic AI routes all its tool and model spans through that provider. A small `_ArksimRoutingProcessor` reads `arksim.turn_id` from arksim's contextvar and stamps it on every span; `arksim.conversation_id` is pinned to the agent's `chat_id` via a `Resource` attribute. Arksim's OTLP receiver listens on `127.0.0.1:4318` whenever `trace_receiver.enabled: true` is set in `config.yaml`, decodes the spans, picks the routing attributes, and lands each tool span as a `ToolCall` on the correct turn in `results/simulation/simulation.json`. The same path works for any OTel exporter you wire up (Logfire, stdlib OTLP, etc.); the example uses the stdlib exporter to avoid an extra dependency.

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

| File              | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| `custom_agent.py` | Pydantic AI `Agent` + OTLP tracer provider                   |
| `tools.py`        | `lookup_order` and `book_table` plain Python callables       |
| `config.yaml`     | Simulator configuration with `trace_receiver` enabled        |
| `scenarios.json`  | Two scenarios, one per tool                                  |
