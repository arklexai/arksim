// SPDX-License-Identifier: Apache-2.0
// Mastra integration for arksim.
//
// Install: npm install
// Auth:    export OPENAI_API_KEY="<your-key>"
//
// Exposes a Mastra Agent with two mock tools (lookup_order, book_table)
// over an OpenAI-compatible chat completions endpoint. Each tool wraps
// its body in an OpenTelemetry span following the OTel GenAI semantic
// conventions; spans are exported via OTLP/HTTP to arksim's trace
// receiver on 127.0.0.1:4318. The Python wrapper passes
// metadata.chat_id and metadata.turn_id in every request; this process
// threads them through an AsyncLocalStorage and a custom span
// processor stamps them on every span as arksim.conversation_id and
// arksim.turn_id so the receiver can route tool calls to the right turn.
//
// Note: Mastra is migrating away from OpenTelemetry toward a proprietary
// AI Tracing system (see Mastra GitHub issue #8577). This example uses
// the stdlib OTel path (@opentelemetry/sdk-node) directly, which works
// today and is independent of Mastra's tracing internals.

import { AsyncLocalStorage } from "node:async_hooks";
import { NodeSDK } from "@opentelemetry/sdk-node";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import {
  BatchSpanProcessor,
  ReadableSpan,
  Span,
  SpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { Context } from "@opentelemetry/api";
import { Resource } from "@opentelemetry/resources";
import { trace } from "@opentelemetry/api";
import { Mastra } from "@mastra/core";
import { Agent } from "@mastra/core/agent";
import { createTool } from "@mastra/core/tools";
import { openai } from "@ai-sdk/openai";
import { Hono } from "hono";
import { serve } from "@hono/node-server";
import { z } from "zod";

type Routing = { chatId: string; turnId?: number };
const routingStore = new AsyncLocalStorage<Routing>();

// Stamp arksim.conversation_id and arksim.turn_id on every span from
// the routing AsyncLocalStorage. Mirrors the Python contextvar-driven
// processor used by the autogen and pydantic-ai examples.
class ArksimRoutingProcessor implements SpanProcessor {
  onStart(span: Span, _parentContext: Context): void {
    const routing = routingStore.getStore();
    if (!routing) return;
    span.setAttribute("arksim.conversation_id", routing.chatId);
    if (routing.turnId !== undefined) {
      span.setAttribute("arksim.turn_id", routing.turnId);
    }
  }
  onEnd(_span: ReadableSpan): void {}
  async shutdown(): Promise<void> {}
  async forceFlush(): Promise<void> {}
}

// Start the OTel SDK before any spans are produced.
const otelSdk = new NodeSDK({
  resource: new Resource({
    "service.name": "arksim-mastra-example",
  }),
  spanProcessors: [
    new ArksimRoutingProcessor(),
    new BatchSpanProcessor(
      new OTLPTraceExporter({
        url: "http://127.0.0.1:4318/v1/traces",
      }),
    ),
  ],
});
otelSdk.start();

const tracer = trace.getTracer("arksim.examples.mastra");

const lookup_order = createTool({
  id: "lookup_order",
  description: "Look up an order by ID and return its status.",
  inputSchema: z.object({
    order_id: z.string().describe("The order identifier to look up."),
  }),
  outputSchema: z.string(),
  execute: async ({ context }) => {
    const args = { order_id: context.order_id };
    return tracer.startActiveSpan("execute_tool lookup_order", (span) => {
      try {
        span.setAttribute("gen_ai.tool.name", "lookup_order");
        span.setAttribute("gen_ai.tool.call.arguments", JSON.stringify(args));
        const result = `Order ${context.order_id}: shipped, arrives Tuesday.`;
        span.setAttribute("gen_ai.tool.call.result", result);
        return result;
      } finally {
        span.end();
      }
    });
  },
});

const book_table = createTool({
  id: "book_table",
  description: "Book a restaurant table for the given party size and time.",
  inputSchema: z.object({
    party_size: z.number().int().describe("Number of people to seat."),
    time: z.string().describe("Time to book for, for example '7pm'."),
  }),
  outputSchema: z.string(),
  execute: async ({ context }) => {
    const args = { party_size: context.party_size, time: context.time };
    return tracer.startActiveSpan("execute_tool book_table", (span) => {
      try {
        span.setAttribute("gen_ai.tool.name", "book_table");
        span.setAttribute("gen_ai.tool.call.arguments", JSON.stringify(args));
        const result = `Booked table for ${context.party_size} at ${context.time}.`;
        span.setAttribute("gen_ai.tool.call.result", result);
        return result;
      } finally {
        span.end();
      }
    });
  },
});

const agent = new Agent({
  id: "assistant",
  name: "assistant",
  instructions:
    "You are a helpful assistant with access to two tools: " +
    "lookup_order(order_id) and book_table(party_size, time). " +
    "Call them when relevant to answer the user.",
  model: openai(process.env.OPENAI_MODEL ?? "gpt-4o"),
  tools: { lookup_order, book_table },
});

const mastra = new Mastra({ agents: { assistant: agent } });

// In-memory session storage for multi-turn conversations.
const sessions: Record<string, Array<{ role: string; content: string }>> = {};

const app = new Hono();

// OpenAI-compatible chat completions endpoint.
app.post("/v1/chat/completions", async (c) => {
  const body = await c.req.json();
  const messages: Array<{ role: string; content: string }> = body.messages ?? [];

  // The Python wrapper passes metadata.chat_id and metadata.turn_id.
  const md = body.metadata ?? {};
  const chatId: string = md.chat_id ?? body.session_id ?? "default";
  const turnId: number | undefined =
    typeof md.turn_id === "number" ? md.turn_id : undefined;

  if (!sessions[chatId]) {
    sessions[chatId] = [];
  }
  sessions[chatId].push(...messages);

  // Run the whole turn under the routing store so every emitted span
  // (model, tool, etc.) carries arksim.conversation_id and arksim.turn_id.
  return routingStore.run({ chatId, turnId }, async () => {
    const assistant = mastra.getAgent("assistant");
    // Use generateLegacy for AI SDK v4 model providers.
    const result = await assistant.generateLegacy(sessions[chatId]);

    const content =
      typeof result.text === "string"
        ? result.text
        : JSON.stringify(result.text);

    sessions[chatId].push({ role: "assistant", content });

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
});

app.get("/health", (c) => c.json({ status: "ok" }));

const port = parseInt(process.env.PORT ?? "8888", 10);
console.log(`Mastra agent server listening on port ${port}`);
serve({ fetch: app.fetch, port });

const shutdown = async () => {
  await otelSdk.shutdown();
  process.exit(0);
};
process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);
