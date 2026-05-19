// SPDX-License-Identifier: Apache-2.0
// Vercel AI SDK integration for arksim.
//
// Install: npm install
// Auth:    export OPENAI_API_KEY="<your-key>"
//
// Exposes a Vercel AI SDK agent with two mock tools (lookup_order,
// book_table) over an OpenAI-compatible chat completions endpoint.
// arksim's `chat_completions` connector drops any tool_calls in the
// response body, so this server takes the same path the Mastra example
// uses: it wraps each tool body in an OpenTelemetry span following the
// OTel GenAI semantic conventions and exports spans via OTLP/HTTP to
// arksim's built-in trace receiver (127.0.0.1:4318). The Python wrapper
// passes metadata.chat_id and metadata.turn_id in every request; this
// process threads them through an AsyncLocalStorage and a custom span
// processor stamps them on every span as arksim.conversation_id and
// arksim.turn_id so the receiver routes tool calls to the right turn.

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
import { generateText, tool, stepCountIs } from "ai";
import { openai } from "@ai-sdk/openai";
import { Hono } from "hono";
import { serve } from "@hono/node-server";
import { z } from "zod";

type Routing = { chatId: string; turnId?: number };
const routingStore = new AsyncLocalStorage<Routing>();

// Stamp arksim.conversation_id and arksim.turn_id on every span from
// the routing AsyncLocalStorage.
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

const otelSdk = new NodeSDK({
  resource: new Resource({
    "service.name": "arksim-vercel-ai-sdk-example",
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

const tracer = trace.getTracer("arksim.examples.vercel-ai-sdk");

const lookup_order = tool({
  description: "Look up an order by ID and return its status.",
  inputSchema: z.object({
    order_id: z.string().describe("The order identifier to look up."),
  }),
  execute: async ({ order_id }) => {
    const args = { order_id };
    return tracer.startActiveSpan("execute_tool lookup_order", (span) => {
      try {
        span.setAttribute("gen_ai.tool.name", "lookup_order");
        span.setAttribute("gen_ai.tool.call.arguments", JSON.stringify(args));
        const result = `Order ${order_id}: shipped, arrives Tuesday.`;
        span.setAttribute("gen_ai.tool.call.result", result);
        return result;
      } finally {
        span.end();
      }
    });
  },
});

const book_table = tool({
  description: "Book a restaurant table for the given party size and time.",
  inputSchema: z.object({
    party_size: z.number().int().describe("Number of people to seat."),
    time: z.string().describe("Time to book for, for example '7pm'."),
  }),
  execute: async ({ party_size, time }) => {
    const args = { party_size, time };
    return tracer.startActiveSpan("execute_tool book_table", (span) => {
      try {
        span.setAttribute("gen_ai.tool.name", "book_table");
        span.setAttribute("gen_ai.tool.call.arguments", JSON.stringify(args));
        const result = `Booked table for ${party_size} at ${time}.`;
        span.setAttribute("gen_ai.tool.call.result", result);
        return result;
      } finally {
        span.end();
      }
    });
  },
});

type ChatMessage = { role: "system" | "user" | "assistant"; content: string };

// In-memory session storage for multi-turn conversations.
const sessions: Record<string, ChatMessage[]> = {};

const app = new Hono();

app.post("/v1/chat/completions", async (c) => {
  const body = await c.req.json();
  const messages: ChatMessage[] = body.messages ?? [];

  const md = body.metadata ?? {};
  const chatId: string = md.chat_id ?? body.session_id ?? "default";
  const turnId: number | undefined =
    typeof md.turn_id === "number" ? md.turn_id : undefined;

  if (!sessions[chatId]) {
    sessions[chatId] = [
      {
        role: "system",
        content:
          "You are a helpful assistant with access to two tools: " +
          "lookup_order(order_id) and book_table(party_size, time). " +
          "Call them when relevant to answer the user.",
      },
    ];
  }
  for (const msg of messages) {
    if (msg.role !== "system") {
      sessions[chatId].push(msg);
    }
  }

  return routingStore.run({ chatId, turnId }, async () => {
    const result = await generateText({
      model: openai("gpt-4o"),
      messages: sessions[chatId],
      tools: { lookup_order, book_table },
      stopWhen: stepCountIs(5),
    });

    const content = result.text;
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
console.log(`Vercel AI SDK agent server listening on port ${port}`);
serve({ fetch: app.fetch, port });

const shutdown = async () => {
  await otelSdk.shutdown();
  process.exit(0);
};
process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);
