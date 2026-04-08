# Tool Call Capture - A2A

This example shows how to surface tool call data through the A2A protocol so arksim can evaluate tool usage with the `tool_call_behavior_failure` metric.

The agent includes tool calls as a `DataPart` alongside the text answer in every A2A response. arksim's A2A client extracts the `DataPart` automatically - no custom agent wrapper needed.

## How it works

```
User message -> A2A server -> Agent runs get_weather tool
  -> Response: TextPart(answer) + DataPart({"tool_calls": [...]})
  -> arksim extracts DataPart -> evaluator scores tool_call_behavior_failure
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install arksim
   ```

2. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="<your-key>"
   ```

3. Start the agent server:
   ```bash
   cd examples/tool-call-capture/a2a
   python -m agent_server.server
   ```

4. In a separate terminal, run arksim:
   ```bash
   cd examples/tool-call-capture/a2a
   arksim simulate-evaluate config.yaml
   ```

## Scenarios

| Scenario | Assertion | What it tests |
|----------|-----------|---------------|
| `weather_query` | `tool_calls contains [get_weather]` | Agent calls the tool when weather is requested |
| `no_tool_needed` | none | Agent handles small talk without spurious tool calls |
