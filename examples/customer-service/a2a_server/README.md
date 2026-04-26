# Customer Service - A2A Server

A2A variant of the customer-service example. Same SQLite-backed tools and scenarios as the Python connector (`custom_agent.py`) and traced (`traced_agent.py`) variants, but connected via the A2A protocol.

Tool calls are surfaced via the arksim A2A tool capture extension (`A2AToolCaptureExtension`). The server declares the extension in its AgentCard, and only emits tool call metadata when the client requests it via the `A2A-Extensions` header.

## Setup

All commands below must be run from `examples/customer-service/`.

1. Install dependencies:
   ```bash
   pip install arksim openai-agents a2a-sdk uvicorn
   ```

2. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="<your-key>"
   ```

3. Start the A2A server (foreground; `Ctrl+C` to stop):
   ```bash
   python -m a2a_server.server
   ```

4. In a separate terminal (also from `examples/customer-service/`), run arksim:
   ```bash
   arksim simulate-evaluate config_a2a.yaml
   ```

5. View results in `./results/evaluation/final_report.html`.

6. Stop the server with `Ctrl+C`.

## Comparison

| Variant | File | Capture method |
|---------|------|----------------|
| Python connector (explicit) | `custom_agent.py` | Returns `AgentResponse` with `tool_calls` |
| Python connector (traced) | `traced_agent.py` | `ArksimTracingProcessor` captures automatically |
| **A2A** | `a2a_server/` | Tool calls in `Task.artifacts[*].metadata` via extension |

All three use the same tools (`tools.py`) and scenarios (`scenarios.json`).
