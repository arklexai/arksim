# Trace Receiver

The trace receiver is a lightweight OTLP/HTTP server that captures tool call spans from agents that handle tools internally. Instead of requiring agents to return tool calls in `AgentResponse`, agents push OpenTelemetry traces to the receiver, and arksim attaches the captured tool calls to `Message.tool_calls` for evaluation.

## Configuration

Add a `trace_receiver` block to your simulation config:

```yaml
trace_receiver:
  enabled: true
  host: 127.0.0.1   # bind address; use 0.0.0.0 for container deployments
  port: 4318         # IANA-assigned OTLP/HTTP standard port
  wait_timeout: 5    # seconds to wait for traces after each agent response
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Enable the trace receiver |
| `host` | string | `127.0.0.1` | Bind address for the HTTP server |
| `port` | int | `4318` | Port number (1-65535) |
| `wait_timeout` | float | `5.0` | Max seconds to wait for traces per turn |

## Agent Instrumentation

Agents must tag their tool call spans with two routing attributes so arksim can match traces to the correct conversation and turn:

| Attribute | Where | Description |
|-----------|-------|-------------|
| `arksim.conversation_id` | Resource or span | Conversation ID from `metadata["chat_id"]` |
| `arksim.turn_id` | Span | Turn index from `metadata["turn_id"]` |

### Supported attribute conventions

The receiver parses tool call data from two OTLP attribute families:

**OTel GenAI semconv** (recommended):
- `gen_ai.tool.name` - tool name
- `gen_ai.tool.call.arguments` - JSON string of arguments
- `gen_ai.tool.call.result` - JSON string of result
- `gen_ai.tool.call.id` - tool call ID

**OpenInference (Arize)**:
- `tool.name` - tool name
- `tool_call.function.arguments` or `tool.parameters` - JSON string of arguments
- `output.value` - tool result
- `tool_call.id` or `tool.id` - tool call ID

### Minimal example

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

provider = TracerProvider(
    resource=Resource.create({
        "service.name": "my-agent",
        "arksim.conversation_id": metadata["chat_id"],
    })
)
provider.add_span_processor(
    SimpleSpanProcessor(OTLPSpanExporter(endpoint="http://127.0.0.1:4318/v1/traces"))
)
tracer = provider.get_tracer("my-agent")

# After each tool execution:
with tracer.start_as_current_span("execute_tool search_flights") as span:
    span.set_attribute("arksim.turn_id", metadata["turn_id"])
    span.set_attribute("gen_ai.tool.name", "search_flights")
    span.set_attribute("gen_ai.tool.call.arguments", '{"origin": "NYC"}')
    span.set_attribute("gen_ai.tool.call.result", '{"flights": [...]}')

provider.force_flush()
```

**Important:** If the agent runs in the same process as arksim, the OTel exporter uses synchronous HTTP (`requests` library). Wrap the trace push in `asyncio.to_thread()` to avoid blocking the event loop. See `examples/customer-service/traced_agent.py` for a complete example.

## Capture Paths

arksim supports two ways to capture tool calls, and they can be used together:

1. **AgentResponse** - Agent returns `AgentResponse(content=..., tool_calls=[...])` from `execute()`. Direct, no infrastructure needed.
2. **Trace receiver** - Agent pushes OTel spans. Requires `trace_receiver.enabled: true` in config.

When both are active, arksim merges tool calls from both sources and deduplicates by tool call ID first, then by `(name, arguments)` signature. Tool calls from `AgentResponse` take precedence over traced duplicates.

## Protobuf vs JSON

The receiver accepts both OTLP protobuf (`application/x-protobuf`) and JSON (`application/json`) payloads. The standard Python `OTLPSpanExporter` sends protobuf by default.

Protobuf support requires the `opentelemetry-proto` package:

```bash
pip install arksim[otel]
```

Without this package, protobuf payloads are rejected with HTTP 415. JSON payloads work without any extra dependencies.
