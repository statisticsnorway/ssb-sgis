import numpy as np


def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    red = red / 255
    nir = nir / 255

    # min_, max_ = np.min(red), np.max(red)
    # red = (red - min_) / (max_ - min_)

    # min_, max_ = np.min(nir), np.max(nir)
    # nir = (nir - min_) / (max_ - min_)

    ndvi_values = (nir - red) / (nir + red)
    ndvi_values[(red + nir) == 0] = 0

    return ndvi_values
