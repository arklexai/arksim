# Google ADK Integration

This example demonstrates how to connect a [Google ADK](https://github.com/google/adk-python) agent to ArkSim using the custom agent connector.

## Prerequisites

1. Install Google ADK:

   ```bash
   pip install google-adk
   ```

2. Set your API key:

   ```bash
   export GOOGLE_API_KEY="<your-key>"
   ```

## Usage

From this example directory, run:

```bash
arksim simulate config.yaml
```

## Files

| File               | Description                         |
| ------------------ | ----------------------------------- |
| `custom_agent.py`  | Google ADK integration              |
| `config.yaml`      | Simulator configuration             |
| `scenarios.json`   | Simulation scenarios                |
