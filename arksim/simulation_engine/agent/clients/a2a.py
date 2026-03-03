# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import uuid

import httpx
from a2a.client import (
    A2ACardResolver,
    ClientConfig,
    ClientFactory,
    create_text_message_object,
)
from a2a.types import TransportProtocol
from a2a.utils.message import get_message_text

from arksim.config import A2AConfig, AgentConfig, AgentType
from arksim.simulation_engine.agent.base import BaseAgent

logger = logging.getLogger(__name__)


class A2AAgent(BaseAgent):
    """A2A (Agent-to-Agent) agent implementation."""

    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        if agent_config.agent_type != AgentType.A2A.value:
            raise ValueError("Agent config must be of type a2a")
        self.config: A2AConfig = agent_config.api_config
        self.chat_id = str(uuid.uuid4())

        # These will be initialized lazily on first execute
        self._client = None
        self._httpx_client: httpx.AsyncClient | None = None
        self._initialized = False

    async def get_chat_id(self) -> str:
        """Get the chat ID."""
        return self.chat_id

    async def _ensure_initialized(self) -> None:
        """Lazily initialize the A2A client on first use."""
        if self._initialized:
            return

        try:
            # Get headers from config (with env var substitution)
            headers = self.config.get_headers()

            # Create httpx client with default timeout and custom headers
            self._httpx_client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0), headers=headers if headers else None
            )

            # Initialize A2ACardResolver
            resolver = A2ACardResolver(
                httpx_client=self._httpx_client,
                base_url=self.config.endpoint,
            )

            # Fetch agent card
            agent_card = await resolver.get_agent_card()

            # Determine streaming capability from agent card
            streaming = (
                agent_card.capabilities.streaming if agent_card.capabilities else False
            )

            # Create client config with supported transports
            config = ClientConfig(
                httpx_client=self._httpx_client,
                supported_transports=[
                    TransportProtocol.jsonrpc,
                    TransportProtocol.http_json,
                ],
                streaming=streaming,
            )

            # Create the A2A client
            self._client = ClientFactory(config).create(agent_card)
            self._initialized = True

        except Exception as e:
            logger.error(f"Error: Could not initialize A2A client: {e}")
            raise

    async def execute(self, user_query: str, **kwargs: object) -> str:
        """Execute user query using A2A protocol."""
        await self._ensure_initialized()

        try:
            # Create message payload
            send_message_payload = create_text_message_object(
                content=user_query,
            )
            send_message_payload.context_id = self.chat_id

            # Send message and collect response
            response_text = ""
            async for message in self._client.send_message(send_message_payload):
                response_text = get_message_text(message)

            return response_text

        except Exception as e:
            logger.error(f"Error: Error calling A2A agent: {e}")
            raise

    async def close(self) -> None:
        """Close the httpx client and cleanup resources."""
        if self._httpx_client:
            await self._httpx_client.aclose()
            self._httpx_client = None
            self._client = None
            self._initialized = False
