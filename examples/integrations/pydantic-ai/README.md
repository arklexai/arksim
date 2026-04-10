# Pydantic AI Integration

This example demonstrates how to connect a [Pydantic AI](https://github.com/pydantic/pydantic-ai) agent to ArkSim using the custom agent connector.

## Prerequisites

1. Install Pydantic AI:

   ```bash
   pip install pydantic-ai
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
| `custom_agent.py`  | Pydantic AI integration             |
| `config.yaml`      | Simulator configuration             |
| `scenarios.json`   | Simulation scenarios                |
