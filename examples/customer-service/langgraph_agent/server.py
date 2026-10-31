# SPDX-License-Identifier: Apache-2.0
"""Customer-service agent on LangChain/LangGraph behind a chat completions
endpoint, exporting its spans over OTLP.

Each request adopts the caller's ``traceparent``, so the agent's tool spans
join the trace the caller opened for that turn. Point the exporter at the
arksim trace receiver and a capture-enabled simulation records the agent's
tool calls.

TRACE_INSTRUMENTATION picks how LangChain runs become spans:
  openinference (default)  openinference-instrumentation-langchain
  langsmith                LangSmith's own OTel export, nothing sent to LangSmith

The exporter reads the standard OTel variables, for example:
  OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:4318/v1/traces
  OTEL_SERVICE_NAME=customer-service-langgraph

Run from customer-service/:
  pip install -r langgraph_agent/requirements.txt
  OPENAI_API_KEY=... AGENT_API_KEY=... uvicorn langgraph_agent.server:app --port 8890
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import store_tools  # noqa: E402

AGENT_API_KEY = os.environ.get("AGENT_API_KEY", "")
TRACE_INSTRUMENTATION = os.environ.get("TRACE_INSTRUMENTATION", "openinference")

TOOLS = [
    store_tools.lookup_customer,
    store_tools.get_order,
    store_tools.get_delivery_status,
    store_tools.search_products,
    store_tools.list_product_categories,
    store_tools.cancel_order,
    store_tools.send_verification_code,
    store_tools.verify_customer,
    store_tools.get_return_policy,
    store_tools.get_product_guide,
    store_tools.search_store_knowledge,
    store_tools.get_knowledge_article,
    store_tools.get_document,
]

# Set globally before LangChain runs anything: LangSmith's OTel mode exports
# through an existing global provider instead of creating its own.
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)

if TRACE_INSTRUMENTATION == "langsmith":
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_TRACING_MODE", "otel")
elif TRACE_INSTRUMENTATION == "openinference":
    from openinference.instrumentation.langchain import LangChainInstrumentor

    LangChainInstrumentor().instrument(tracer_provider=provider)
else:
    raise RuntimeError(f"Unknown TRACE_INSTRUMENTATION: {TRACE_INSTRUMENTATION}")

# Imported after the instrumentation choice so LangSmith sees its env vars.
from langchain.agents import create_agent  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

store_tools.init_db()
agent = create_agent(
    ChatOpenAI(model="gpt-6-luna"),
    tools=TOOLS,
    system_prompt=store_tools.AGENT_INSTRUCTIONS,
)

app = FastAPI(title="Chat Completion Wrapper - Customer Service (LangGraph)")
FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)


class ChatCompletionRequestMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatCompletionRequestMessage]


@app.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest, authorization: str | None = Header(None)
) -> dict:
    # Same checks as agent_server/chat_completions/server.py: this agent can
    # run cancel_order, so it is never served unauthenticated.
    token = (authorization or "").split(" ")
    if not authorization or len(token) != 2:
        raise HTTPException(status_code=401, detail="Invalid Authorization header")
    if token[1] != AGENT_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API token")

    # The caller resends the whole transcript each turn, so the agent keeps no
    # state between requests. The system prompt is the agent's own.
    messages = [
        {"role": m.role, "content": m.content}
        for m in request.messages
        if m.role != "system"
    ]
    if not messages or messages[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")

    result = await agent.ainvoke({"messages": messages})
    return {
        "choices": [
            {"message": {"role": "assistant", "content": result["messages"][-1].text}}
        ]
    }
