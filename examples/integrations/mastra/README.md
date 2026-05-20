# Mastra Integration

A [Mastra](https://github.com/mastra-ai/mastra) agent running as a Node.js service with two mock tools (`lookup_order`, `book_table`) whose calls are exported as OpenTelemetry spans and captured by arksim's built-in OTLP receiver, landing in `simulation.json`. Arksim drives the agent via an OpenAI-compatible chat completions endpoint.

## Setup

1. Install arksim with the OTel extra (required for the trace receiver this example uses):

   ```bash
   pip install 'arksim[otel]'
   ```

2. Install Node dependencies:

   ```bash
   npm install
   ```

   This installs `@mastra/core`, `@ai-sdk/openai`, `hono`, the OpenTelemetry SDK and OTLP exporter, and `zod`.

3. Set your API key:

   ```bash
   export OPENAI_API_KEY="<your-key>"
   ```

If `gpt-5.1` isn't available in your account, set `OPENAI_MODEL=<your-model>` before starting the server to override the agent's model. The simulator's own model (used for the simulated user and evaluation) is set separately by the top-level `model:` field in `config.yaml`.

## Run

Start the Mastra agent server in one terminal:

```bash
npm start
```

In a separate terminal, run the simulation and evaluation:

```bash
arksim simulate-evaluate config.yaml
```

## How it works

Mastra is TypeScript-only, so arksim drives it through its `chat_completions` agent connector: each turn becomes an HTTP POST to `http://localhost:8888/v1/chat/completions` carrying the conversation history plus a `metadata` block with `chat_id` and `turn_id` (`enable_metadata: true` in `config.yaml` switches on that forward). On the Node side, the server starts an OpenTelemetry SDK pointed at arksim's OTLP receiver (`127.0.0.1:4318/v1/traces`) and threads `chat_id` and `turn_id` through an `AsyncLocalStorage`. A custom `ArksimRoutingProcessor` reads that store and stamps `arksim.conversation_id` and `arksim.turn_id` on every span it sees. Both tools are registered with `createTool({ ... })`; their bodies are wrapped in `tracer.startActiveSpan("execute_tool ...")` with the OTel GenAI semantic conventions (`gen_ai.tool.name`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`). The receiver decodes the OTLP payload, routes by the two arksim attributes, and lands each tool span as a `ToolCall` on the correct turn in `results/simulation/simulation.json`.

**Tracing note:** Mastra is migrating away from OpenTelemetry toward a proprietary AI Tracing system (see [Mastra GitHub issue #8577](https://github.com/mastra-ai/mastra/issues/8577)). The OTel path used in this example works today and is independent of Mastra's internal tracing, but the upstream direction may change. If you adopt Mastra's new tracing surface in the future, point its exporter at the same OTLP endpoint or write a small bridge.

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

| File              | Description                                                       |
| ----------------- | ----------------------------------------------------------------- |
| `agent_server.ts` | Mastra agent + OTLP exporter + tool wrappers                      |
| `config.yaml`     | Simulator config (`chat_completions` + `trace_receiver` enabled)  |
| `scenarios.json`  | Two scenarios, one per tool                                       |
| `package.json`    | Node.js dependencies (Mastra, OTel SDK, Zod)                      |
