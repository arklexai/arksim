# Vercel AI SDK Integration

This example demonstrates how to connect a [Vercel AI SDK](https://github.com/vercel/ai) agent to ArkSim via an HTTP server exposing an OpenAI-compatible chat completions endpoint.

## Prerequisites

1. Install dependencies:

   ```bash
   npm install
   ```

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
| `agent_server.ts`  | Vercel AI SDK agent as an HTTP server      |
| `config.yaml`      | Simulator configuration (chat_completions) |
| `scenarios.json`   | Simulation scenarios                       |
| `package.json`     | Node.js dependencies                       |
