# Rasa Pro Integration

This example shows how to simulate and evaluate a [Rasa Pro](https://rasa.com/) assistant built with [CALM](https://rasa.com/docs/pro/), Rasa's LLM-powered dialogue engine. CALM replaces traditional NLU pipelines with flow-based conversation management, where an LLM routes user messages to the right flow based on natural language descriptions.

The integration connects to Rasa via its REST webhook channel. Point ArkSim at a running Rasa server, define your scenarios, and get a full evaluation report.

## What This Example Covers

The included CALM assistant handles customer support queries through five flows: greetings, goodbyes, pricing questions, feature questions, and a multi-step order status lookup with slot filling and a custom action.

ArkSim runs four scenarios against it:

| Scenario | What It Exercises |
|----------|-------------------|
| **Product inquiry** | Simple Q&A flow routing (pricing, features) |
| **Order status check** | Multi-step flow with slot filling and a custom action |
| **Topic switching** | Context management when switching between flows mid-conversation |
| **Unknown order** | Error handling when an order ID is not found |

Each scenario runs 3 times to test consistency across runs, since CALM's LLM-based routing is non-deterministic.

After the run, ArkSim generates an evaluation report (`evaluation/final_report.html`) scoring each conversation on goal completion, helpfulness, coherence, relevance, and more. It also surfaces specific failure patterns like repeated responses, missing information, or incorrect behavior, so you know exactly where to improve.

## Prerequisites

Install ArkSim in its own environment (`pip install arksim` or `pip install -e .` from the repo root). The steps below run Rasa Pro in a separate virtualenv managed by Poetry so the two dependency graphs stay isolated.

1. Install Rasa Pro (Python 3.10-3.13).

   The demo ships a `pyproject.toml` in `rasa_project/` that pulls `rasa-pro` from Rasa's private registry with a pinned version:

   ```bash
   cd rasa_project
   poetry install
   ```

   Rasa Pro is distributed from a private package index, so Poetry needs to authenticate against it on first use. See Rasa's [install guide](https://rasa.com/docs/rasa-pro/installation/python/environment-set-up/) for the canonical auth step.

   Alternative: standalone install with `uv pip install rasa-pro` or `pip install rasa-pro`, following the same Rasa docs for index-URL / credentials.

2. Set your environment variables:

   ```bash
   export RASA_LICENSE="<your-license-key>"
   export OPENAI_API_KEY="<your-key>"
   ```

   `RASA_LICENSE` is Rasa Pro's runtime license check. `OPENAI_API_KEY` is used by both CALM (via `endpoints.yml`) and ArkSim's simulated user + evaluator.

   Get a free Rasa Pro Developer Edition license at [rasa.com](https://rasa.com/rasa-pro-developer-edition-license-key-request).

3. Train the CALM assistant:

   ```bash
   cd rasa_project
   poetry run rasa train
   cd ..
   ```

## Usage

Start the Rasa server (from `rasa_project/`). Custom actions run in-process via `actions_module`, so no separate action server is needed.

```bash
cd rasa_project
poetry run rasa run --enable-api --cors "*"
```

`--cors "*"` opens the REST channel to any origin; intended for local development only. Do not expose this server on a public network.

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
| `scenarios.json` | Evaluation scenarios |
| `rasa_project/config.yml` | CALM pipeline configuration |
| `rasa_project/domain.yml` | Slots, responses, and session settings |
| `rasa_project/endpoints.yml` | LLM model groups and action endpoint |
| `rasa_project/data/flows.yml` | CALM flow definitions |
| `rasa_project/actions/actions.py` | Custom action for order status lookup |
| `rasa_project/credentials.yml` | Channel credentials (REST channel enabled) |
