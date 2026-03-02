import logging
import os
from uuid import uuid4

import httpx
from a2a.client import (
    A2ACardResolver,
    ClientConfig,
    ClientFactory,
    create_text_message_object,
)
from a2a.client.auth.credentials import CredentialService
from a2a.client.auth.interceptor import AuthInterceptor
from a2a.client.middleware import ClientCallContext
from a2a.types import TransportProtocol
from a2a.utils.message import get_message_text

API_KEY = os.environ.get("A2A_API_KEY", "")


class EnvOrConfigCredentialService(CredentialService):
    """A credential service that returns an API key from the environment or configuration."""

    async def get_credentials(
        self,
        security_scheme_name: str,
        context: ClientCallContext | None,
    ) -> str | None:
        """Returns the static API key for any security scheme."""
        return os.getenv("A2A_CLIENT_CREDENTIAL", API_KEY)


async def main() -> None:
    # Configure logging to show INFO level messages
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)  # Get a logger instance

    # --8<-- [start:A2ACardResolver]

    base_url = "http://localhost:9999"

    # Configure httpx client with longer timeout for LLM calls
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(60.0),  # 60 second timeout for LLM responses
    ) as httpx_client:
        # Initialize A2ACardResolver with auth header for card resolution
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
        )

        try:
            # Fetch agent card (may need auth header for protected endpoints)
            agent_card = await resolver.get_agent_card()

            # Create A2A client with the agent card and auth interceptor
            config = ClientConfig(
                httpx_client=httpx_client,
                supported_transports=[
                    TransportProtocol.jsonrpc,
                    TransportProtocol.http_json,
                ],
                streaming=agent_card.capabilities.streaming,
            )

            # Use the built-in AuthInterceptor with a credential service
            credential_service = EnvOrConfigCredentialService()
            auth_interceptor = AuthInterceptor(credential_service)
            client = ClientFactory(config).create(
                agent_card, interceptors=[auth_interceptor]
            )
        except Exception as e:
            logger.error(f"Error initializing client: {e}")
            logger.exception(e)
            return
        logger.info("A2AClient initialized.")

        chat_id = uuid4().hex

        send_message_payload = create_text_message_object(
            content="What are the products you have?",
        )
        send_message_payload.context_id = chat_id

        response_text = ""
        async for message in client.send_message(send_message_payload):
            # Message contains parts with the response text
            response_text = get_message_text(message)

        logger.info(response_text)
        # --8<-- [end:send_message]

        # ask for the price of the ADAM
        send_message_payload = create_text_message_object(
            content="I want a health insurance im 27 years old and i live in Toronto",
        )
        send_message_payload.context_id = chat_id

        response_text = ""
        async for message in client.send_message(send_message_payload):
            response_text = get_message_text(message)

        logger.info(response_text)

        # ask for the price of the Scorpion
        send_message_payload = create_text_message_object(
            content="What is the deductible for the health insurance?",
        )
        send_message_payload.context_id = chat_id

        response_text = ""
        async for message in client.send_message(send_message_payload):
            response_text = get_message_text(message)

        logger.info(response_text)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
