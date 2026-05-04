# %%
import sys
from pathlib import Path

src = str(Path(__file__).parent).replace("tests", "") + "src"

sys.path.insert(0, src)

from sgis.maps.create_secrets import create_credentials

create_credentials(
    project_id="ssb-myteam-prod",  # your Dapla team GCP project ID
    username_secret_id="nib-username",  # name for the username secret
    username="my_nib_username",  # actual username value
    password_secret_id="nib-password",  # name for the password secret
    password="my_nib_password",  # actual password value
)
