# Rasa Integration

This example demonstrates how to connect a [Rasa](https://rasa.com/) agent to ArkSim using a custom agent that communicates with Rasa's REST webhook channel.

## Prerequisites

1. Install Rasa (requires Python 3.10):

   ```bash
   pip3 install rasa
   ```

2. Train the included Rasa project:

   ```bash
   cd rasa_project
   rasa train
   cd ..
   ```

## Usage

Start the Rasa server:

```bash
cd rasa_project
rasa run --enable-api --cors "*"
```

In a separate terminal, run the simulation:

```bash
arksim simulate config.yaml
```

By default the agent connects to `http://localhost:5005/webhooks/rest/webhook`. Override this with the `RASA_ENDPOINT` environment variable if your server runs elsewhere.

## Files

| File                          | Description                                        |
| ----------------------------- | -------------------------------------------------- |
| `custom_agent.py`             | Rasa agent wrapper (async HTTP via `httpx`)         |
| `config.yaml`                 | Simulator configuration (custom agent)              |
| `scenarios.json`              | Simulation scenarios                                |
| `rasa_project/config.yml`     | Rasa pipeline and policy configuration              |
| `rasa_project/domain.yml`     | Intents, responses, and session settings            |
| `rasa_project/credentials.yml`| Channel credentials (REST channel enabled)          |
| `rasa_project/endpoints.yml`  | Action server endpoint configuration                |
| `rasa_project/data/nlu.yml`   | NLU training examples                              |
| `rasa_project/data/rules.yml` | Conversation rules and fallback handling            |
