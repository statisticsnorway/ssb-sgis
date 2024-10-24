import numpy as np


def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    red = red / 255
    nir = nir / 255

    ndvi_values = (nir - red) / (nir + red)
    ndvi_values[(red + nir) == 0] = 0

    return ndvi_values
