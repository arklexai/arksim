# Mastra Integration

This example demonstrates how to connect a [Mastra](https://github.com/mastra-ai/mastra) agent to ArkSim via an HTTP server exposing an OpenAI-compatible chat completions endpoint.

## Prerequisites

1. Install dependencies:

   ```bash
   npm install
   ```

   This installs `@mastra/core`, `@ai-sdk/openai`, `hono`, and `@hono/node-server`.

2. Set your API key:

   ```bash
   export OPENAI_API_KEY="<your-key>"
   ```

## Usage

Start the agent server:

```bash
npm start
```

In a separate terminal, run the simulation and evaluation:

```bash
arksim simulate-evaluate config.yaml
```

## Files

| File               | Description                                |
| ------------------ | ------------------------------------------ |
| `agent_server.ts`  | Mastra agent as an HTTP server             |
| `config.yaml`      | Simulator configuration (chat_completions) |
| `scenarios.json`   | Simulation scenarios                       |
| `package.json`     | Node.js dependencies                       |
