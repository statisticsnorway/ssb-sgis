"""Utilities for fetching secrets from Google Secret Manager using Dapla credentials."""
from dapla_auth_client import AuthClient
from google.cloud import secretmanager


def get_secret_version(
    project_id: str, shortname: str, version_id: str = "latest"
) -> str:
    """Access the payload for a given secret version.

    The user's Google credentials are used to authorize that the user has
    permission to access the secret.

    Args:
        project_id: ID of the Google Cloud project where the secret is stored.
        shortname: Name (not full path) of the secret in Secret Manager.
        version_id: The version of the secret to access. Defaults to 'latest'.

    Returns:
        The payload of the secret version as a UTF-8 decoded string.

    Raises:
        ImportError: If dapla-auth-client or google-cloud-secret-manager is not installed.
    """
    try:
        from dapla_auth_client import AuthClient
        from google.cloud import secretmanager
    except ImportError as e:
        raise ImportError(
            "dapla-auth-client and google-cloud-secret-manager are required. "
            "Install them with: pip install dapla-auth-client google-cloud-secret-manager"
        ) from e

    client = secretmanager.SecretManagerServiceClient(
        credentials=AuthClient.fetch_google_credentials()
    )
    secret_name = f"projects/{project_id}/secrets/{shortname}/versions/{version_id}"
    response = client.access_secret_version(name=secret_name)
    return response.payload.data.decode("UTF-8")


def get_credentials(
    project_id: str,
    username_secret_id: str,
    password_secret_id: str,
    version_id: str = "latest",
) -> tuple[str, str]:
    """Fetch username and password from Google Secret Manager.

    Args:
        project_id: GCP project ID where the secrets are stored.
        username_secret_id: Secret ID for the username.
        password_secret_id: Secret ID for the password.
        version_id: The version of the secrets to access. Defaults to 'latest'.

    Returns:
        A tuple of (username, password).

    Raises:
        ImportError: If dapla-auth-client or google-cloud-secret-manager is not installed.

    Example:
        >>> username, password = get_credentials(
        ...     project_id="my-gcp-project",
        ...     username_secret_id="my-service-username",
        ...     password_secret_id="my-service-password",
        ... )
    """
    username = get_secret_version(project_id, username_secret_id, version_id)
    password = get_secret_version(project_id, password_secret_id, version_id)
    return username, password
