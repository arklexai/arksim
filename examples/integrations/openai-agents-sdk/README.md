# OpenAI Agents SDK Integration

This example demonstrates how to connect an [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) agent to ArkSim using the custom agent connector.

## Prerequisites

1. Install the OpenAI Agents SDK:

   ```bash
   pip install openai-agents
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
| `custom_agent.py`  | OpenAI Agents SDK integration       |
| `config.yaml`      | Simulator configuration             |
| `scenarios.json`   | Simulation scenarios                |
