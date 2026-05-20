# LlamaIndex Integration

A [LlamaIndex](https://github.com/run-llama/llama_index) `FunctionAgent` with two mock tools (`lookup_order`, `book_table`), wired to arksim through `ArksimLlamaIndexObserver` so every tool call is captured in `simulation.json`.

## Setup

1. Install arksim with the LlamaIndex extra plus the OpenAI LLM:

   ```bash
   pip install 'arksim[llamaindex]'
   pip install llama-index-core llama-index-llms-openai
   ```

2. Set your API key:

   ```bash
   export OPENAI_API_KEY="<your-key>"
   ```

If `gpt-5.1` isn't available in your account, set `OPENAI_MODEL=<your-model>` and re-run, or edit `config.yaml` directly.

## Run

From this example directory:

```bash
arksim simulate-evaluate config.yaml
```

## How it works

Unlike the other integrations that register a callback on the agent, LlamaIndex emits tool-call events through the workflow stream rather than through the `llama_index_instrumentation` dispatcher. The agent calls `self._workflow.run(user_msg=...)` to get a `WorkflowHandler`, hands it to `ArksimLlamaIndexObserver.consume_stream(handler)` to forward every `ToolCall` and `ToolCallResult` event into arksim, then `await`s the handler to collect the final `AgentOutput`. Before each agent turn the simulator binds `conversation_id`, `turn_id`, and the trace receiver into contextvars, so the observer's submissions land in the `tool_calls` field of every turn in `results/simulation/simulation.json`.

## Expected output

After running the example, each turn that invoked a tool contains entries like this in `results/simulation/simulation.json`:

```json
"tool_calls": [
  {
    "name": "lookup_order",
    "arguments": {"order_id": "ORD-1001"},
    "result": "Order ORD-1001: shipped, arrives Tuesday.",
    "source": "llamaindex"
  },
  {
    "name": "book_table",
    "arguments": {"party_size": 4, "time": "7pm"},
    "result": "Booked table for 4 at 7pm.",
    "source": "llamaindex"
  }
]
```

## Files

| File              | Description                                             |
| ----------------- | ------------------------------------------------------- |
| `custom_agent.py` | LlamaIndex `FunctionAgent` + `ArksimLlamaIndexObserver` |
| `tools.py`        | `lookup_order` and `book_table` mock tools              |
| `config.yaml`     | Simulator configuration                                 |
| `scenarios.json`  | Two scenarios, one per tool                             |
