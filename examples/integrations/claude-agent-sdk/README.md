# Claude Agent SDK Integration

This example demonstrates how to connect a [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk) agent to ArkSim using the custom agent connector.

## Prerequisites

1. Install the Claude Agent SDK:

   ```bash
   pip install claude-agent-sdk
   ```

2. Set your API key:

   ```bash
   export ANTHROPIC_API_KEY="<your-key>"
   ```

## Usage

From this example directory, run:

```bash
arksim simulate config.yaml
```

## Files

| File               | Description                         |
| ------------------ | ----------------------------------- |
| `custom_agent.py`  | Claude Agent SDK integration        |
| `config.yaml`      | Simulator configuration             |
| `scenarios.json`   | Simulation scenarios                |
