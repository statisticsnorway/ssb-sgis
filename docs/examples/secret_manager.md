# Using Google Secret Manager on Dapla

This guide shows how to store and retrieve credentials (e.g. for Norge i bilder)
using Google Secret Manager on Dapla.

---

## Prerequisites

Install the required packages:

```bash
pip install dapla-auth-client google-cloud-secret-manager
```

---

## Step 1: Store credentials in Secret Manager

Run this **once** from a Dapla Jupyter notebook to store your credentials:

```python
from sgis.maps.create_secrets import create_credentials

create_credentials(
    project_id="ssb-myteam-prod",        # your Dapla team GCP project ID
    username_secret_id="nib-username",   # name for the username secret
    username="my_nib_username",          # actual username value
    password_secret_id="nib-password",   # name for the password secret
    password="my_nib_password",          # actual password value
)
```

Expected output:

```
Secret 'nib-username' created in project 'ssb-myteam-prod'.
Secret version added for 'nib-username'.
Secret 'nib-password' created in project 'ssb-myteam-prod'.
Secret version added for 'nib-password'.
Credentials stored successfully in project 'ssb-myteam-prod'.
  Username secret : nib-username
  Password secret : nib-password
```

> **Note:** If the secret already exists, a new version is added automatically.
> Only the `latest` version is used by default when reading secrets.

---

## Step 2: Use credentials to load a WMTS tile layer

```python
from sgis.maps.wms import get_norge_i_bilder_wmts

tile = get_norge_i_bilder_wmts(
    project_id="ssb-myteam-prod",
    username_secret_id="nib-username",
    password_secret_id="nib-password",
)

# Add to a map
import folium

m = folium.Map(location=[59.91, 10.75], zoom_start=13)
tile.add_to(m)
folium.LayerControl().add_to(m)
m
```

---

## Step 3: Read credentials directly (advanced)

You can also fetch secrets manually using `get_credentials` or `get_secret_version`:

```python
from sgis.maps.secrets import get_credentials, get_secret_version

# Fetch both username and password at once
username, password = get_credentials(
    project_id="ssb-myteam-prod",
    username_secret_id="nib-username",
    password_secret_id="nib-password",
)

# Or fetch a single secret
username = get_secret_version(
    project_id="ssb-myteam-prod",
    shortname="nib-username",
    version_id="latest",   # or a specific version number e.g. "3"
)
```

---

## Finding your Dapla project ID

Run this in a Dapla Jupyter notebook:

```python
import subprocess
result = subprocess.run(
    ["gcloud", "config", "get-value", "project"],
    capture_output=True,
    text=True,
)
print(result.stdout.strip())
```

Or check it in **Dapla Lab → Team settings**. The project ID typically follows
the pattern `ssb-<team-name>-prod` or `ssb-<team-name>-test`.

---

## IAM permissions

Your Dapla user or service account needs the following GCP roles:

| Action              | Required Role                          |
|---------------------|----------------------------------------|
| Create secrets      | `roles/secretmanager.admin`            |
| Read secrets        | `roles/secretmanager.secretAccessor`   |

Contact your Dapla team admin if you don't have access.
