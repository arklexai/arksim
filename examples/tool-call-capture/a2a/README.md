# Tool Call Capture - A2A

This example shows how to surface tool call data through the A2A protocol so arksim can evaluate tool usage with the `tool_call_behavior_failure` metric.

The agent declares the arksim tool capture extension in its AgentCard and surfaces tool calls in artifact metadata on the Task. arksim's A2A client extracts them automatically - no custom agent wrapper needed.

## How it works

```
User message -> A2A server -> Agent runs get_weather tool
  -> Response: Task with Artifact
       parts:    [TextPart(answer)]
       extensions: ["https://arksim.arklex.ai/a2a/tool-call-capture/v1"]
       metadata: {".../tool_calls": [...]}
  -> arksim reads Task.artifacts -> evaluator scores tool_call_behavior_failure
```

This follows A2A spec section 3.7 (task outputs go in artifacts, not messages) and uses the [AgentExtension](https://a2a-protocol.org/latest/topics/extensions/) mechanism to declare the convention.

## Setup

All commands below must be run from `examples/tool-call-capture/a2a/` (the server uses a relative package import, and `config.yaml` references `scenarios.json` by relative path).

1. Install dependencies. The example uses a few arksim helpers (`A2AToolCaptureExtension`, `ToolCall`, and `extract_tool_calls`), so `arksim` is listed in `requirements.txt` alongside `a2a-sdk` and `openai-agents`:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="<your-key>"
   ```

3. Start the agent server (foreground; `Ctrl+C` to stop):
   ```bash
   python -m agent_server.server
   ```

4. In a separate terminal (also from this directory), run arksim against the server:
   ```bash
   arksim simulate-evaluate config.yaml
   ```

5. When you're done, stop the server with `Ctrl+C` in the terminal from step 3.

## Scenarios

| Scenario | Assertion | What it tests |
|----------|-----------|---------------|
| `weather_query` | `tool_calls contains [get_weather]` | Agent calls the tool when weather is requested |
| `no_tool_needed` | none | Agent handles small talk without spurious tool calls |
