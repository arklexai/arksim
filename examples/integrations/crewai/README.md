# CrewAI Integration

This example demonstrates how to connect a [CrewAI](https://github.com/crewAIInc/crewAI) agent to ArkSim using the custom agent connector.

## Prerequisites

1. Install CrewAI:

   ```bash
   pip install crewai
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
| `custom_agent.py`  | CrewAI integration                  |
| `config.yaml`      | Simulator configuration             |
| `scenarios.json`   | Simulation scenarios                |
