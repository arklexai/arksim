# AutoGen Integration

This example demonstrates how to connect a [Microsoft AutoGen](https://github.com/microsoft/autogen) agent to ArkSim using the custom agent connector.

## Prerequisites

1. Install AutoGen:

   ```bash
   pip install autogen-agentchat autogen-ext[openai]
   ```

2. Set your API key:

   ```bash
   export OPENAI_API_KEY="<your-key>"
   ```

## Usage

From this example directory, run:

```bash
arksim simulate-evaluate config.yaml
```

## Files

| File               | Description                         |
| ------------------ | ----------------------------------- |
| `custom_agent.py`  | AutoGen integration                 |
| `config.yaml`      | Simulator configuration             |
| `scenarios.json`   | Simulation scenarios                |
