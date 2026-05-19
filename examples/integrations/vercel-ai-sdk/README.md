# Vercel AI SDK Integration

A [Vercel AI SDK](https://github.com/vercel/ai) agent running as a Node.js service with two mock tools (`lookup_order`, `book_table`) whose calls are exported as OpenTelemetry spans and captured by arksim's built-in OTLP receiver, landing in `simulation.json`. arksim drives the agent via an OpenAI-compatible chat completions endpoint.

## Setup

1. Install Node dependencies:

   ```bash
   npm install
   ```

   This installs `ai`, `@ai-sdk/openai`, `hono`, the OpenTelemetry SDK and OTLP exporter, and `zod`.

2. Set your API key:

   ```bash
   export OPENAI_API_KEY="<your-key>"
   ```

## Run

Start the Vercel AI SDK agent server in one terminal:

```bash
npm start
```

In a separate terminal, run the simulation and evaluation:

```bash
arksim simulate-evaluate config.yaml
```

## How it works

The Vercel AI SDK is TypeScript-only, so arksim drives it through its `chat_completions` agent connector: each turn becomes an HTTP POST to `http://localhost:8888/v1/chat/completions` carrying the conversation history plus a `metadata` block with `chat_id` and `turn_id` (`enable_metadata: true` in `config.yaml` switches on that forward). arksim's chat-completions response parser only reads the assistant text from `choices[0].message.content` and ignores any `tool_calls` in the body, so the server cannot surface tool calls that way. Instead, the server starts an OpenTelemetry SDK pointed at arksim's OTLP receiver (`127.0.0.1:4318/v1/traces`) and threads `chat_id` and `turn_id` through an `AsyncLocalStorage`. A custom `ArksimRoutingProcessor` reads that store and stamps `arksim.conversation_id` and `arksim.turn_id` on every span. Both tools are registered with `tool({ ... })` from the `ai` package and passed to `generateText` with `stopWhen: stepCountIs(5)` to allow multi-step tool use; their bodies are wrapped in `tracer.startActiveSpan("execute_tool ...")` with the OTel GenAI semantic conventions (`gen_ai.tool.name`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`). The receiver decodes the OTLP payload, routes by the two arksim attributes, and lands each tool span as a `ToolCall` on the correct turn in `results/simulation/simulation.json`.

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
| `agent_server.ts` | Vercel AI SDK agent + OTLP exporter + tool wrappers               |
| `config.yaml`     | Simulator config (`chat_completions` + `trace_receiver` enabled)  |
| `scenarios.json`  | Two scenarios, one per tool                                       |
| `package.json`    | Node.js dependencies (ai, OTel SDK, Zod)                          |
