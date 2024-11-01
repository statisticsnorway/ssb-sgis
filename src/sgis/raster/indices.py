import numpy as np


def ndvi(red: np.ndarray, nir: np.ndarray, padding: int = 0) -> np.ndarray:
    ndvi_values = (nir - red + padding) / (nir + red + padding)
    ndvi_values[(red + nir) == 0] = 0

    return ndvi_values
