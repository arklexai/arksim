# Customer Service - A2A Server

A2A variant of the customer-service example. Same SQLite-backed tools and scenarios as the Python connector (`openai_agents/custom_agent.py`) and traced (`openai_agents/traced_agent.py`) variants, but connected via the A2A protocol.

Tool calls are surfaced via the arksim A2A tool capture extension (`A2AToolCaptureExtension`). The server declares the extension in its AgentCard, and only emits tool call metadata when the client requests it via the `A2A-Extensions` header.

> **The client must send that header.** a2a-sdk resolves requested extensions from `A2A-Extensions`, but arksim's A2A client only sets `Message.extensions`. Without the header the server suppresses tool call metadata and `tool_call_behavior_failure` scores against an empty trajectory with no error. `config_a2a.yaml` sets it under `api_config.headers`; any other consumer must do the same.

`agent.py` is backend-selectable via the `AGENT_BACKEND` env var (`openai` or `adk`, default `openai`) - both backends expose the agent through the same A2A server (`server.py`, unchanged either way). Each backend's SDK-specific imports are lazy, so an environment with only one of `openai-agents` / `google-adk` installed works fine for that backend.

## Setup

All commands below must be run from `examples/customer-service/`.

1. Install dependencies for the backend you want to run:
   ```bash
   pip install arksim a2a-sdk uvicorn openai-agents   # OpenAI backend
   pip install arksim a2a-sdk uvicorn -r ../google_adk/requirements.txt  # ADK backend
   ```

2. Set the API key for that backend:
   ```bash
   export OPENAI_API_KEY="<your-key>"   # OpenAI backend
   export GOOGLE_API_KEY="<your-key>"   # ADK backend
   ```

3. Start the A2A server (foreground; `Ctrl+C` to stop), picking the backend:
   ```bash
   AGENT_BACKEND=openai python -m a2a_server.server   # or AGENT_BACKEND=adk
   ```

4. In a separate terminal (also from `examples/customer-service/`), run arksim:
   ```bash
   arksim simulate-evaluate config_a2a.yaml
   ```

5. View results in `./results/evaluation/final_report.html`.

6. Stop the server with `Ctrl+C`.

## Mounting inside another app

`server.py` exposes `build_agent_card()`, `build_routes()` and `build_app()` so the A2A surface can be served by a parent application rather than its own process:

```python
from a2a_server.server import build_routes

for route in build_routes("https://host/my-prefix", "/my-prefix"):
    parent_app.router.routes.append(route)
```

`build_routes()` returns exact Routes, not a mountable sub-app - a Starlette `Mount` at `/x` only matches `/x/...`, so a bare POST to the JSON-RPC url would miss it. The first argument is what the AgentCard advertises, so it must be the full externally reachable URL including the prefix.

## Comparison

| Variant | File | Capture method |
|---------|------|----------------|
| Python connector (explicit) | `openai_agents/custom_agent.py` | Returns `AgentResponse` with `tool_calls` |
| Python connector (traced) | `openai_agents/traced_agent.py` | `ArksimTracingProcessor` captures automatically |
| Python connector (ADK) | `google_adk/custom_agent.py` | Returns `AgentResponse` via `run_and_capture_tool_calls()` |
| **A2A** | `a2a_server/` | Tool calls in `Task.artifacts[*].metadata` via extension, either backend |

All variants use the same underlying tools (`store_tools.py`) and scenarios (`scenarios.json`).
