# Customer Service - Google ADK Variant

Google ADK variant of the customer-service example, built the way many production agents are (ADK `FunctionTool`s behind a conversational agent, exposed over MCP). Reuses the same `store_tools.py` core as the OpenAI Agents SDK variant (`../openai_agents/`) - only the framework-specific tool wrapping and agent wiring differ.

## Files

| File | Role |
|------|------|
| `adk_tools.py` | Wraps `store_tools.py`'s functions as ADK `FunctionTool`s. `ADK_TOOLS` is the agent's own set; `MCP_TOOLS` adds three fixture tools exposed only over MCP. Also builds `TOOL_ANNOTATIONS`, a plain lookup table `mcp_server.py` reads to set MCP `readOnlyHint`/`destructiveHint` (not an ADK concept - `FunctionTool` has no native annotations attribute). |
| `agent.py` | `build_agent()` constructs the `LlmAgent`; `run_and_capture_tool_calls()` pairs ADK's per-event function calls with their responses by id (ADK emits these as separate stream events, unlike the OpenAI Agents SDK's single `RunResult`) into arksim `ToolCall` objects. |
| `custom_agent.py` | arksim `BaseAgent` wrapper (`GoogleADKAgent`) for the direct/CLI path - manages the ADK session and returns `AgentResponse` with captured tool calls. |
| `mcp_server.py` | Exposes `MCP_TOOLS` over MCP (streamable HTTP, port 9997) using the low-level `mcp.server.Server` directly, not the decorator-per-tool `MCPServer` - demonstrates wrapping an existing `FunctionTool` set the way a team with existing tools would, rather than redeclaring them in a new framework's decorator style. Also holds the auth and fault-injection gate. |
| `mcp_shapes.py` | Turns tool return values into `CallToolResult` shapes (structured content, multi-block, prose, image, both error shapes). No ADK or arksim imports, so the protocol shapes are testable on their own. |
| `smoke_client.py` | Generic MCP client (no ADK/arksim dependency) that asserts the protocol behaviours a connector must handle and exits non-zero on failure - usable as a CI check, not only a demo. |
| `try_it.py` | Scratch script for manually asking the agent one question during development - not part of the official example surface. |

Requires `mcp>=2.0.0`. The handler registration API (`Server.add_request_handler`), the client entry point (`mcp.Client`), and snake_case result fields (`is_error`, `structured_content`) all changed in 2.0.

One interop detail worth knowing when wrapping ADK tools: ADK emits an empty input schema for a zero-argument tool, and MCP rejects that because `inputSchema.type` is required. `mcp_server.list_tools` normalizes it.

## Setup

All commands below must be run from `examples/customer-service/`.

```bash
pip install -r google_adk/requirements.txt
export GOOGLE_API_KEY="<your-key>"
```

## Running the example

**Direct (arksim CLI):**
```bash
arksim simulate-evaluate config_adk.yaml
```

**A2A server** (see `../a2a_server/README.md` for the full A2A flow):
```bash
AGENT_BACKEND=adk python -m a2a_server.server
```

**MCP server + smoke test** (no LLM call needed - pure protocol check over the expanded 225-product and 89-article fixture):
```bash
python -m google_adk.mcp_server       # terminal 1
python google_adk/smoke_client.py     # terminal 2
```

To run the same checks without binding a port, use the in-memory transport instead:
```bash
python -m unittest test_mcp_server
```

Set `MCP_HOST`/`MCP_PORT`/`MCP_ALLOWED_HOSTS` on the server and `MCP_URL` on the client when running through a container, public development tunnel, or deployed endpoint.

`MCP_AUTH_TOKEN`, `MCP_AUTH_HEADER`, `MCP_FAULT`, and `MCP_FAULT_TOOLS` turn on credential enforcement and failure injection; see the parent README for the full table.

**Quick manual test:**
```bash
python google_adk/try_it.py "What is the status of order ORD-1001?"
```

## Safety: tool allowlist annotations

`TOOL_ANNOTATIONS` in `adk_tools.py` marks each tool's MCP hints:
- 13 tools carry `readOnlyHint: True`.
- `cancel_order` carries `readOnlyHint: False, destructiveHint: True`.
- `render_room_preview` carries `readOnlyHint: False, destructiveHint: False` - generative image output, so it is neither destructive nor a lookup tool, and must not be auto-allowlisted.
- `send_verification_code` deliberately carries **no entry at all** - this exercises the fail-safe default on the consuming side (the platform's MCP connector must treat an absent hint as unsafe, not safe), per the MCP spec's own guidance that `destructiveHint` defaults to `true` when omitted.
