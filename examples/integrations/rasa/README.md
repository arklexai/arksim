# Rasa Integration

This example demonstrates how to connect a [Rasa](https://rasa.com/) agent to ArkSim using a custom agent that communicates with Rasa's REST webhook channel.

The Rasa bot uses a hybrid approach common in production deployments: Rasa's NLU pipeline handles in-domain intents (pricing, features, support hours, refund policy) with structured responses, while a custom action forwards out-of-domain messages to an LLM (OpenAI) for open-ended conversation. The two scenarios exercise both paths so you can evaluate each independently.

## Prerequisites

1. Install Rasa and the OpenAI SDK (requires Python 3.10):

   ```bash
   pip3 install rasa openai
   ```

2. Set your API key:

   ```bash
   export OPENAI_API_KEY="<your-key>"
   ```

3. Train the included Rasa project:

   ```bash
   cd rasa_project
   rasa train
   cd ..
   ```

## Usage

Start the Rasa action server and main server (from `rasa_project/`):

```bash
cd rasa_project
rasa run actions &
rasa run --enable-api --cors "*"
```

In a separate terminal, run the simulation:

```bash
arksim simulate config.yaml
```

By default the agent connects to `http://localhost:5005/webhooks/rest/webhook`. Override this with the `RASA_ENDPOINT` environment variable if your server runs elsewhere.

## Files

| File                              | Description                                    |
| --------------------------------- | ---------------------------------------------- |
| `custom_agent.py`                 | Rasa agent wrapper (async HTTP via `httpx`)    |
| `config.yaml`                     | Simulator configuration (custom agent)         |
| `scenarios.json`                  | Simulation scenarios                           |
| `rasa_project/actions/actions.py` | Custom action that forwards messages to OpenAI |
| `rasa_project/config.yml`         | Rasa pipeline and policy configuration         |
| `rasa_project/domain.yml`         | Intents, responses, and session settings       |
| `rasa_project/credentials.yml`    | Channel credentials (REST channel enabled)     |
| `rasa_project/endpoints.yml`      | Action server endpoint configuration           |
| `rasa_project/data/nlu.yml`       | NLU training examples                          |
| `rasa_project/data/rules.yml`     | Conversation rules and LLM fallback            |
