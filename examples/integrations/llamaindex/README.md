# LlamaIndex Integration

This example demonstrates how to connect a [LlamaIndex](https://github.com/run-llama/llama_index) agent to ArkSim using the custom agent connector.

## Prerequisites

1. Install LlamaIndex and the OpenAI integration:

   ```bash
   pip install llama-index llama-index-llms-openai
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
| `custom_agent.py`  | LlamaIndex integration              |
| `config.yaml`      | Simulator configuration             |
| `scenarios.json`   | Simulation scenarios                |
