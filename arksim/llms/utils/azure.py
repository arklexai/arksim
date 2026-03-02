import os
from collections.abc import Callable

AZURE_COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"


def get_azure_token_provider(client_id: str | None = None) -> Callable[[], str]:
    """
    Get Azure AD token provider for Managed Identity authentication.

    Args:
        client_id: Optional client ID for User-Assigned Managed Identity.
                   If None, uses System-Assigned Managed Identity.

    Returns:
        A callable that returns a bearer token string.
    """
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    credential = (
        DefaultAzureCredential(managed_identity_client_id=client_id)
        if client_id
        else DefaultAzureCredential()
    )

    return get_bearer_token_provider(credential, AZURE_COGNITIVE_SERVICES_SCOPE)


def check_azure_env_vars() -> None:
    """
    Check if the Azure environment variables are set.
    """
    client_id = os.getenv("AZURE_CLIENT_ID")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    if not client_id or not api_version or not deployment_name:
        raise ValueError(
            "AZURE_CLIENT_ID, AZURE_OPENAI_API_VERSION, and AZURE_OPENAI_DEPLOYMENT_NAME are required for initializing Azure OpenAI Agent!"
        )
