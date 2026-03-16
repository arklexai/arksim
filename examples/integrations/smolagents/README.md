# Smolagents Integration

This example demonstrates how to connect a [Smolagents](https://github.com/huggingface/smolagents) (Hugging Face) agent to ArkSim using the custom agent connector.

## Prerequisites

1. Install Smolagents:

   ```bash
   pip install smolagents
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
| `custom_agent.py`  | Smolagents integration              |
| `config.yaml`      | Simulator configuration             |
| `scenarios.json`   | Simulation scenarios                |
