# Rasa Pro Integration

This example demonstrates how to evaluate a [Rasa Pro](https://rasa.com/) assistant with ArkSim. The Rasa bot uses [CALM](https://rasa.com/docs/pro/), Rasa's LLM-powered dialogue engine, which replaces traditional NLU pipelines with flow-based conversation management.

The integration communicates with Rasa via its REST webhook channel, making setup straightforward: point ArkSim at a running Rasa server and run your scenarios.

## What This Example Tests

ArkSim runs three scenarios against the CALM assistant to evaluate different capabilities:

| Scenario | What It Tests |
|----------|---------------|
| **Product inquiry** | Simple Q&A flow routing (pricing, features) |
| **Order status check** | Multi-step flow with slot filling and a custom action |
| **Topic switching** | Context management across flow transitions |

Each scenario runs 3 times to evaluate consistency, which matters because CALM's LLM-based routing is non-deterministic.

## Prerequisites

1. Install Rasa Pro (requires Python 3.10+):

   ```bash
   uv pip install rasa-pro
   ```

   Or with pip:

   ```bash
   pip install rasa-pro
   ```

2. Set your environment variables:

   ```bash
   export RASA_LICENSE="<your-license-key>"
   export OPENAI_API_KEY="<your-key>"
   ```

   Get a free Rasa Pro Developer Edition license at [rasa.com](https://rasa.com/rasa-pro-developer-edition-license-key-request).

3. Train the CALM assistant:

   ```bash
   cd rasa_project
   rasa train
   cd ..
   ```

## Usage

Start the Rasa server (from `rasa_project/`):

```bash
cd rasa_project
rasa run --enable-api --cors "*"
```

In a separate terminal, run the simulation and evaluation:

```bash
arksim simulate-evaluate config.yaml
```

By default the agent connects to `http://localhost:5005/webhooks/rest/webhook`. Override with the `RASA_ENDPOINT` environment variable if your server runs elsewhere.

## Files

| File | Description |
|------|-------------|
| `custom_agent.py` | ArkSim agent wrapper (async HTTP via `httpx`) |
| `config.yaml` | ArkSim simulator configuration |
| `scenarios.json` | Three evaluation scenarios |
| `rasa_project/config.yml` | CALM pipeline configuration |
| `rasa_project/domain.yml` | Slots, responses, and session settings |
| `rasa_project/endpoints.yml` | LLM model groups and action endpoint |
| `rasa_project/data/flows.yml` | CALM flow definitions |
| `rasa_project/actions/actions.py` | Custom action for order status lookup |
| `rasa_project/credentials.yml` | Channel credentials (REST channel enabled) |
