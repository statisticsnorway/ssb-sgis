"""Function to check whether we are in Dapla without importing Dapla.

Function from: From https://github.com/statisticsnorway/ssb-altinn-python/blob/main/src/altinn/utils.py
"""

import os


def is_dapla() -> bool:
    """Simply checks if an os environment variable contains the text 'dapla'."""
    return any("dapla" in key.lower() for key in os.environ)
