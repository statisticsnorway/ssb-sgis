"""Function to check whether we are in Dapla without importing Dapla.

Function from: From https://github.com/statisticsnorway/ssb-altinn-python/blob/main/src/altinn/utils.py
"""

import os


def is_dapla() -> bool:
    """From https://github.com/statisticsnorway/ssb-altinn-python/blob/main/src/altinn/utils.py."""
    jupyter_image_spec = os.environ.get("JUPYTER_IMAGE_SPEC")
    return bool(jupyter_image_spec and "dapla-jupyterlab" in jupyter_image_spec)
