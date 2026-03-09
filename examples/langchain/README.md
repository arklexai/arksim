# LangChain/LangGraph Integration

This example demonstrates how to connect a [LangChain](https://github.com/langchain-ai/langchain)/[LangGraph](https://github.com/langchain-ai/langgraph) agent to ArkSim using the custom agent connector.

## Prerequisites

1. Install LangGraph and the OpenAI integration:

   ```bash
   pip install langgraph langchain-openai
   ```

2. Set your API key:

   ```bash
   export OPENAI_API_KEY="<your-key>"
   ```

## Usage

From this example directory, run:

```bash
arksim simulate config.yaml
```

## Files

| File               | Description                         |
| ------------------ | ----------------------------------- |
| `custom_agent.py`  | LangChain/LangGraph integration     |
| `config.yaml`      | Simulator configuration             |
| `scenarios.json`   | Simulation scenarios                |
