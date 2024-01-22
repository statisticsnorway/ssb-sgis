"""Function to check whether we are in Dapla without importing Dapla.

Function from: From https://github.com/statisticsnorway/ssb-altinn-python/blob/main/src/altinn/utils.py
"""

import os


def is_dapla() -> bool:
    """From https://github.com/statisticsnorway/ssb-altinn-python/blob/main/src/altinn/utils.py."""
    try:
        return os.environ["GCS_TOKEN_PROVIDER_KEY"] == "google"
    except KeyError:
        return False
