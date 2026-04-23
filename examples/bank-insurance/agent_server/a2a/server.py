# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import uvicorn
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    APIKeySecurityScheme,
    SecurityScheme,
)
from starlette.applications import Starlette
from starlette.authentication import (
    AuthCredentials,
    AuthenticationBackend,
    AuthenticationError,
    SimpleUser,
)
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.requests import HTTPConnection

from .agent_executor import (
    BankInsuranceAgentExecutor,
)

API_KEY = os.environ.get("A2A_API_KEY")
PUBLIC_PATHS = {
    "/.well-known/agent.json",
    "/.well-known/agent-card.json",
}


class APIKeyAuthBackend(AuthenticationBackend):
    async def authenticate(
        self, conn: HTTPConnection
    ) -> tuple[AuthCredentials, SimpleUser] | None:
        if conn.url.path in PUBLIC_PATHS:
            return AuthCredentials([]), SimpleUser("public-agent-card")

        api_key = conn.headers.get("api-key")
        if not API_KEY or api_key != API_KEY:
            raise AuthenticationError("Invalid or missing API key")

        return AuthCredentials(["authenticated"]), SimpleUser("api-key-user")


if __name__ == "__main__":
    # --8<-- [start:AgentSkill]
    skill = AgentSkill(
        id="bank-insurance_customer_service",
        name="Bank-Insurance customer service assistant",
        description=(
            "A conversational assistant that answers about Insurance and banking "
            "customer questions using a retrieval-augmented knowledge base."
        ),
        tags=["customer service", "banking", "insurance", "rag"],
        examples=[
            "What insurance products do you offer?",
            "How do I file an auto claim?",
            "Can you explain Insurance coverage options?",
        ],
    )

    server_url = os.getenv("A2A_SERVER_URL", "http://localhost:9999/")

    # This will be the public-facing agent card
    public_agent_card = AgentCard(
        name="Bank-Insurance Customer Service Agent",
        description=(
            "A RAG-powered Insurance and banking customer service agent that "
            "provides concise, accurate answers based on internal documentation."
        ),
        supported_interfaces=[AgentInterface(url=server_url)],
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        security_schemes={
            "api-key": SecurityScheme(
                api_key_security_scheme=APIKeySecurityScheme(
                    name="api-key",
                    location="header",
                ),
            ),
        },
    )

    task_store = InMemoryTaskStore()
    request_handler = DefaultRequestHandler(
        agent_executor=BankInsuranceAgentExecutor(),
        task_store=task_store,
        agent_card=public_agent_card,
    )

    app = Starlette(
        routes=[
            *create_agent_card_routes(agent_card=public_agent_card),
            *create_jsonrpc_routes(
                request_handler=request_handler,
                rpc_url="/",
            ),
        ],
    )
    app.add_middleware(AuthenticationMiddleware, backend=APIKeyAuthBackend())

    uvicorn.run(app, host="0.0.0.0", port=9999)
