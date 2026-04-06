# Dify Integration

This example demonstrates how to connect a [Dify](https://dify.ai) chatbot to ArkSim using the custom agent connector. It tests a customer service bot backed by a Dify knowledge base.

> **Note:** This integration requires a **Chatbot** app in Dify (not a Workflow or Text Generator app).

## Prerequisites

1. Install ArkSim and the Dify Python SDK:

   ```bash
   pip install arksim dify-client
   ```

2. Create a **Chatbot** app in Dify (Cloud or self-hosted) and add a knowledge base with your documents. The included scenarios assume a home improvement store knowledge base, but you can adapt them to match your own content.

3. Publish the app, then copy the API key from **API Access** in the Dify dashboard:

   ```bash
   export DIFY_API_KEY="<your-app-api-key>"
   ```

4. Set your OpenAI key (used by ArkSim's simulated user and evaluator, not your Dify app):

   ```bash
   export OPENAI_API_KEY="<your-key>"
   ```

5. If you are using a self-hosted Dify instance, also set:

   ```bash
   export DIFY_BASE_URL="http://your-dify-host/v1"
   ```

## Usage

From this example directory, run:

```bash
arksim simulate-evaluate config.yaml
```

Results are written to `./results/simulation/simulation.json`. The evaluation report is printed to stdout with per-scenario metric scores and failure analysis.

## Files

| File              | Description                                              |
| ----------------- | -------------------------------------------------------- |
| `custom_agent.py` | ArkSim agent that connects to Dify via the Chat API      |
| `config.yaml`     | ArkSim simulation and evaluation settings                |
| `scenarios.json`  | Test scenarios for a home improvement customer service bot |
