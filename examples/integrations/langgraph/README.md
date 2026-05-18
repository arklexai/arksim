# LangGraph Integration

A hand-built [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph` with two mock tools (`lookup_order`, `book_table`), wired to arksim through `ArksimLangChainHandler` so every tool call is captured in `simulation.json`.

## Setup

1. Install arksim with the LangChain extra plus LangGraph:

   ```bash
   pip install 'arksim[langchain]'
   pip install langgraph langchain-openai
   ```

2. Set your API key:

   ```bash
   export OPENAI_API_KEY="<your-key>"
   ```

## Run

From this example directory:

```bash
arksim simulate-evaluate config.yaml
```

## How it works

LangGraph reuses LangChain's callback bus, so `ArksimLangChainHandler` (a LangChain `AsyncCallbackHandler`) captures tool calls from inside the graph. The agent builds a `StateGraph` with a `chatbot` LLM node and a `ToolNode` for `lookup_order` and `book_table`, with a `tools_condition` conditional edge that routes between them. Before each turn the simulator binds `conversation_id`, `turn_id`, and the trace receiver into contextvars, and the agent passes the handler via `callbacks=[self._handler]` on `ainvoke` so tool invocations land in the `tool_calls` field of every turn in `results/simulation/simulation.json`.

## Expected output

After running the example, each turn that invoked a tool contains entries like this in `results/simulation/simulation.json`:

```json
"tool_calls": [
  {
    "name": "lookup_order",
    "arguments": {"order_id": "ORD-1001"},
    "result": "Order ORD-1001: shipped, arrives Tuesday.",
    "source": "langchain"
  },
  {
    "name": "book_table",
    "arguments": {"party_size": 4, "time": "7pm"},
    "result": "Booked table for 4 at 7pm.",
    "source": "langchain"
  }
]
```

## Files

| File              | Description                                       |
| ----------------- | ------------------------------------------------- |
| `custom_agent.py` | LangGraph `StateGraph` + `ArksimLangChainHandler` |
| `tools.py`        | `lookup_order` and `book_table` mock tools        |
| `config.yaml`     | Simulator configuration                           |
| `scenarios.json`  | Two scenarios, one per tool                       |
