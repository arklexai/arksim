# Dify Integration

This example connects a [Dify](https://dify.ai) Agent app to arksim. The Python wrapper drives Dify's Chat API over HTTP and extracts tool invocations from the response so they land in `simulation.json` as `ToolCall` instances.

## Setup

1. Install arksim:

   ```bash
   pip install arksim
   ```

2. Create an **Agent** app in Dify (Cloud or self-hosted). Agent apps emit `agent_thoughts` in their blocking-mode responses, which is the payload this wrapper parses. A Chatbot app returns plain text only and will produce empty `tool_calls`.

3. In Dify Studio, create the Agent app and attach two tools matching the scenarios. Dify's docs are the source of truth for the current UI:

   - Create the Agent app following [Build an Agent application](https://docs.dify.ai/en/use-dify/build/agent). Use any model you have credits for; the wrapper only reads `agent_thoughts` from the response.
   - Add the two tools following [Build a tool plugin](https://docs.dify.ai/develop-plugin/dev-guides-and-walkthroughs/tool-plugin). Each tool returns a deterministic string so the simulation is reproducible:
     - `lookup_order(order_id: string) -> string` returning a status string (e.g. `"Order ORD-1001: shipped, arrives Tuesday."`).
     - `book_table(party_size: integer, time: string) -> string` returning a confirmation string (e.g. `"Booked table for 4 at 7pm."`).
   - In the Agent app's **Tools** panel, attach both tools and authorize them so the agent can call them.

   The wrapper does not care how the tools are implemented (custom tool, workflow tool, or built-in); it only reads what Dify reports back in `agent_thoughts`.

4. Publish the app and copy the API key from **API Access** in the dashboard:

   ```bash
   export DIFY_API_KEY="<your-app-api-key>"
   ```

5. Set your OpenAI key (used by arksim's simulated user and evaluator, not by your Dify app):

   ```bash
   export OPENAI_API_KEY="<your-key>"
   ```

6. For self-hosted Dify, also set:

   ```bash
   export DIFY_BASE_URL="http://your-dify-host/v1"
   ```

## Run

```bash
arksim simulate-evaluate config.yaml
```

Results are written to `./results/simulation/simulation.json`.

## How it works

Dify has no Python SDK callback surface for tool calls, so arksim drives Dify through the REST Chat API and reads tool metadata back from the response. For Agent apps in blocking mode, Dify includes an `agent_thoughts` list where each entry records the tool name, the arguments the agent passed, and the observation returned by the tool. The wrapper in `custom_agent.py` parses that list into `ToolCall(name, arguments, result, source="dify")` and returns an `AgentResponse` carrying both the assistant text and the captured tool calls. The actual tool execution still happens inside Dify; the Python side only transports requests and surfaces what the Dify response reveals. If you point the wrapper at a Chatbot app, the `agent_thoughts` field is absent and `tool_calls` will be empty; the conversation still runs but trajectory-based metrics lose tool-call signal.

## Expected output

For a turn that invoked `lookup_order`, `simulation.json` contains:

```json
"tool_calls": [
  {
    "name": "lookup_order",
    "arguments": {"order_id": "ORD-1001"},
    "result": "Order ORD-1001: shipped, arrives Tuesday.",
    "source": "dify"
  }
]
```

## Files

| File              | Description                                                 |
| ----------------- | ----------------------------------------------------------- |
| `custom_agent.py` | arksim agent that drives Dify and extracts `agent_thoughts` |
| `config.yaml`     | arksim simulation and evaluation settings                   |
| `scenarios.json`  | Two scenarios, one per tool                                 |
