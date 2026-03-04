# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    APIKeySecurityScheme,
    In,
    SecurityScheme,
)
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

    # This will be the public-facing agent card
    public_agent_card = AgentCard(
        name="Bank-Insurance Customer Service Agent",
        description=(
            "A RAG-powered Insurance and banking customer service agent that "
            "provides concise, accurate answers based on internal documentation."
        ),
        url=os.getenv("A2A_SERVER_URL", "http://localhost:9999/"),
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        security=[
            {
                "api-key": [],
            }
        ],
        security_schemes={
            "api-key": SecurityScheme(
                root=APIKeySecurityScheme(
                    name="api-key",
                    in_=In.header,
                    type="apiKey",
                )
            ),
        },
    )

    request_handler = DefaultRequestHandler(
        agent_executor=BankInsuranceAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=public_agent_card,
        http_handler=request_handler,
    )

    app = server.build()
    app.add_middleware(AuthenticationMiddleware, backend=APIKeyAuthBackend())

    uvicorn.run(app, host="0.0.0.0", port=9999)
