# Claude Agent SDK Integration

A [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk) client with two mock tools (`lookup_order`, `book_table`), wired to arksim through `ArksimClaudeHooks` so every tool call is captured in `simulation.json`. The Claude Agent SDK uses Anthropic models, so this example runs against `claude-sonnet-4-6` rather than an OpenAI model.

## Setup

1. Install arksim with the Claude Agent extra:

   ```bash
   pip install 'arksim[claude-agent]'
   ```

2. Set your API key:

   ```bash
   export ANTHROPIC_API_KEY="<your-key>"
   ```

## Run

From this example directory:

```bash
arksim simulate-evaluate config.yaml
```

## How it works

`ArksimClaudeHooks` exposes a `PostToolUse` hook through its `hooks_dict()` method, which the agent passes straight into `ClaudeAgentOptions(hooks=...)`. The two mock tools are decorated with `@tool` and registered via `create_sdk_mcp_server` as an in-process MCP server named `arksim_tools`; Claude sees them as `mcp__arksim_tools__lookup_order` and `mcp__arksim_tools__book_table`, which `allowed_tools` whitelists. Before each turn the simulator binds `conversation_id`, `turn_id`, and the trace receiver into contextvars, so when the hook fires after each tool invocation it emits `ToolCall` records that land in the `tool_calls` field of every turn in `results/simulation/simulation.json`.

## Expected output

After running the example, each turn that invoked a tool contains entries like this in `results/simulation/simulation.json`:

```json
"tool_calls": [
  {
    "name": "mcp__arksim_tools__lookup_order",
    "arguments": {"order_id": "ORD-1001"},
    "result": "[{'type': 'text', 'text': 'Order ORD-1001: shipped, arrives Tuesday.'}]",
    "source": "claude_agent_sdk"
  },
  {
    "name": "mcp__arksim_tools__book_table",
    "arguments": {"party_size": 4, "time": "7pm"},
    "result": "[{'type': 'text', 'text': 'Booked table for 4 at 7pm.'}]",
    "source": "claude_agent_sdk"
  }
]
```

## Files

| File              | Description                                |
| ----------------- | ------------------------------------------ |
| `custom_agent.py` | `ClaudeSDKClient` + `ArksimClaudeHooks`    |
| `tools.py`        | `lookup_order` and `book_table` mock tools |
| `config.yaml`     | Simulator configuration                    |
| `scenarios.json`  | Two scenarios, one per tool                |
