"""Script for creating secrets in Google Secret Manager using Dapla credentials.

Run this script once from a Dapla Jupyter notebook or terminal to store
credentials (e.g. Norge i bilder username and password) in Secret Manager.

Example usage:
    from sgis.maps.create_secrets import create_secret, create_credentials

    create_credentials(
        project_id="my-team-gcp-project",
        username_secret_id="nib-username",
        username="my_actual_username",
        password_secret_id="nib-password",
        password="my_actual_password",
    )
"""

from __future__ import annotations


def _get_client():
    """Return an authenticated Secret Manager client using Dapla credentials."""
    try:
        from dapla_auth_client import AuthClient
        from google.cloud import secretmanager
    except ImportError as e:
        raise ImportError(
            "dapla-auth-client and google-cloud-secret-manager are required. "
            "Install them with: pip install dapla-auth-client google-cloud-secret-manager"
        ) from e

    return secretmanager.SecretManagerServiceClient(
        credentials=AuthClient.fetch_google_credentials()
    )


def create_secret(
    project_id: str,
    shortname: str,
    value: str,
    exists_ok: bool = True,
) -> None:
    """Create a secret and store its value in Google Secret Manager.

    Uses Dapla (dapla-auth-client) credentials for authentication.
    If the secret already exists and exists_ok is True, a new version
    is added with the provided value.

    Args:
        project_id: The GCP project ID (your Dapla team project),
            e.g. 'ssb-myteam-prod'.
        shortname: The name of the secret, e.g. 'nib-username'.
        value: The secret value to store, e.g. the actual username or password.
        exists_ok: If True, adds a new version when the secret already exists.
            If False, raises an error when the secret already exists.
            Defaults to True.

    Raises:
        ImportError: If dapla-auth-client or google-cloud-secret-manager is not installed.
        google.api_core.exceptions.AlreadyExists: If the secret exists and exists_ok is False.

    Example:
        >>> create_secret(
        ...     project_id="ssb-myteam-prod",
        ...     shortname="nib-username",
        ...     value="my_actual_username",
        ... )
    """
    from google.api_core.exceptions import AlreadyExists

    client = _get_client()
    parent = f"projects/{project_id}"
    secret_path = f"{parent}/secrets/{shortname}"

    # Create the secret resource (without a value)
    try:
        client.create_secret(
            request={
                "parent": parent,
                "secret_id": shortname,
                "secret": {"replication": {"automatic": {}}},
            }
        )
        print(f"Secret '{shortname}' created in project '{project_id}'.")
    except AlreadyExists:
        if not exists_ok:
            raise
        print(f"Secret '{shortname}' already exists. Adding a new version.")

    # Add the secret value as a new version
    client.add_secret_version(
        request={
            "parent": secret_path,
            "payload": {"data": value.encode("UTF-8")},
        }
    )
    print(f"Secret version added for '{shortname}'.")


def create_credentials(
    project_id: str,
    username_secret_id: str,
    username: str,
    password_secret_id: str,
    password: str,
    exists_ok: bool = True,
) -> None:
    """Create username and password secrets in Google Secret Manager.

    Uses Dapla (dapla-auth-client) credentials for authentication.
    Calls create_secret() for both the username and password.

    Args:
        project_id: The GCP project ID (your Dapla team project),
            e.g. 'ssb-myteam-prod'.
        username_secret_id: The secret name for the username,
            e.g. 'nib-username'.
        username: The actual username value to store.
        password_secret_id: The secret name for the password,
            e.g. 'nib-password'.
        password: The actual password value to store.
        exists_ok: If True, adds a new version when a secret already exists.
            Defaults to True.

    Raises:
        ImportError: If dapla-auth-client or google-cloud-secret-manager is not installed.

    Example:
        >>> create_credentials(
        ...     project_id="ssb-myteam-prod",
        ...     username_secret_id="nib-username",
        ...     username="my_actual_username",
        ...     password_secret_id="nib-password",
        ...     password="my_actual_password",
        ... )
    """
    create_secret(project_id, username_secret_id, username, exists_ok=exists_ok)
    create_secret(project_id, password_secret_id, password, exists_ok=exists_ok)
    print(
        f"Credentials stored successfully in project '{project_id}'.\n"
        f"  Username secret : {username_secret_id}\n"
        f"  Password secret : {password_secret_id}"
    )
