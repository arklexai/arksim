# Agent Evaluation Example - Tool Calling

This example demonstrates ArkSim's tool-calling support. The agent is configured with tools (`get_weather`, `search_places`) that the LLM can invoke via standard OpenAI `tool_calls` responses. ArkSim intercepts these calls, returns a static mock result, and feeds it back to the model so the conversation continues naturally.

## How it works

1. The agent's request body includes a `tools` array describing available functions.
2. When the LLM decides to call a tool, it returns a `tool_calls` response instead of plain text.
3. ArkSim responds with the configured `tool_call_result` (a static JSON string) for each call.
4. The LLM receives the tool result and produces a final text response.
5. This loop repeats up to `max_tool_call_rounds` times per turn.

## Prerequisites

1. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
   ```

## Running

```bash
arksim simulate-evaluate examples/tool-calling/config.yaml
```

Or simulate only:

```bash
arksim simulate examples/tool-calling/config.yaml
```

## Configuration

Key tool-calling settings in `config.yaml`:

| Setting | Description |
| --- | --- |
| `tools` (in body) | OpenAI-format tool definitions sent with each request |
| `tool_call_result` | Static JSON returned for every tool call during simulation |
| `max_tool_call_rounds` | Max tool-call round-trips per agent turn (default: 10) |

For more information, see the [ArkSim documentation](https://docs.arklex.ai/overview).

## Files

| File | Description |
| --- | --- |
| `config.yaml` | Simulator + evaluator configuration with tool definitions |
| `scenarios.json` | Scenarios that trigger tool usage (weather, places) |
