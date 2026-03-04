# Agent Evaluation Example - Personal AI Assistant

This directory gives an example of running ArkSim with a personal AI assistant agent for the **OpenClaw use case**. You can follow the example to evaluate your own OpenClaw agent.

## Overview

OpenClaw is an open-source personal AI assistant that helps users manage daily tasks, smart home devices, messaging, calendar, and files. This example demonstrates how to simulate and evaluate conversational interactions with a personal assistant agent.

## OpenClaw Instance Setup

Use this for production evaluation against your actual OpenClaw deployment.

### Prerequisites

1. Install and configure OpenClaw: https://openclaw.ai
2. Enable the HTTP Chat Completions endpoint in `~/.openclaw/openclaw.json`:
   ```json
   {
     "gateway": {
       "http": {
         "endpoints": {
           "chatCompletions": {
             "enabled": true
           }
         }
       }
     }
   }
   ```

### Steps

1. Start the OpenClaw gateway:

   ```bash
   openclaw gateway --port 18789 --verbose
   ```

2. Set environment variables:

   ```bash
   export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
   export OPENCLAW_TOKEN="<YOUR_OPENCLAW_GATEWAY_TOKEN>"
   ```

   Find your token in `~/.openclaw/openclaw.json` under `gateway.auth.token`.

3. From this example directory, run:
   ```bash
   arksim simulate-evaluate config.yaml
   ```

### Verifying OpenClaw Connection

Test the endpoint before running the simulator:

```bash
curl -sS "http://127.0.0.1:18789/v1/chat/completions" \
  -H "Authorization: Bearer $OPENCLAW_TOKEN" \
  -H "Content-Type: application/json" \
  -H "x-openclaw-agent-id: main" \
  -d '{"model":"openclaw","messages":[{"role":"user","content":"Hello"}]}'
```

## Configuration

For more information on configuration, see the [ArkSim documentation](https://docs.arklex.ai/overview).

## Files

| File                | Description                                |
| ------------------- | ------------------------------------------ |
| `config.yaml`       | Simulator + evaluator configuration        |
| `scenarios.json`    | Simulation scenarios                       |
