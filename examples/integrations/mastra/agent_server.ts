// SPDX-License-Identifier: Apache-2.0
// Mastra integration for ArkSim.
//
// Install: npm install @mastra/core @ai-sdk/openai hono @hono/node-server
// Auth:    export OPENAI_API_KEY="<your-key>"

import { Mastra } from "@mastra/core";
import { Agent } from "@mastra/core/agent";
import { openai } from "@ai-sdk/openai";
import { Hono } from "hono";
import { serve } from "@hono/node-server";

const agent = new Agent({
  id: "assistant",
  name: "assistant",
  instructions: "You are a helpful assistant.",
  model: openai("gpt-4o"),
});

const mastra = new Mastra({ agents: { assistant: agent } });

// In-memory session storage for multi-turn conversations
const sessions: Record<string, Array<{ role: string; content: string }>> = {};

const app = new Hono();

// OpenAI-compatible chat completions endpoint
app.post("/v1/chat/completions", async (c) => {
  const body = await c.req.json();
  const messages: Array<{ role: string; content: string }> = body.messages ?? [];

  const sessionKey = body.session_id ?? "default";
  if (!sessions[sessionKey]) {
    sessions[sessionKey] = [];
  }
  sessions[sessionKey].push(...messages);

  const assistant = mastra.getAgent("assistant");
  const result = await assistant.generate(sessions[sessionKey]);

  const content =
    typeof result.text === "string" ? result.text : JSON.stringify(result.text);

  sessions[sessionKey].push({ role: "assistant", content });

  return c.json({
    id: `chatcmpl-${Date.now()}`,
    object: "chat.completion",
    choices: [
      {
        index: 0,
        message: { role: "assistant", content },
        finish_reason: "stop",
      },
    ],
  });
});

app.get("/health", (c) => c.json({ status: "ok" }));

const port = parseInt(process.env.PORT ?? "8888", 10);
console.log(`Mastra agent server listening on port ${port}`);
serve({ fetch: app.fetch, port });
