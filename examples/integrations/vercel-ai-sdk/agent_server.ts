// SPDX-License-Identifier: Apache-2.0
// Vercel AI SDK integration for ArkSim.
//
// Install: npm install ai @ai-sdk/openai hono @hono/node-server
// Auth:    export OPENAI_API_KEY="<your-key>"

import { generateText } from "ai";
import { openai } from "@ai-sdk/openai";
import { Hono } from "hono";
import { serve } from "@hono/node-server";

type ChatMessage = { role: "system" | "user" | "assistant"; content: string };

// In-memory session storage for multi-turn conversations
const sessions: Record<string, ChatMessage[]> = {};

const app = new Hono();

// OpenAI-compatible chat completions endpoint
app.post("/v1/chat/completions", async (c) => {
  const body = await c.req.json();
  const messages: ChatMessage[] = body.messages ?? [];

  const sessionKey = body.session_id ?? "default";
  if (!sessions[sessionKey]) {
    sessions[sessionKey] = [
      { role: "system", content: "You are a helpful assistant." },
    ];
  }

  for (const msg of messages) {
    if (msg.role !== "system") {
      sessions[sessionKey].push(msg);
    }
  }

  const result = await generateText({
    model: openai("gpt-4o"),
    messages: sessions[sessionKey],
  });

  const content = result.text;
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
console.log(`Vercel AI SDK agent server listening on port ${port}`);
serve({ fetch: app.fetch, port });
