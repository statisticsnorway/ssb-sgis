from collections.abc import Callable

import numpy as np
import pandas as pd

from .raster import Raster


def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    # normalize red and nir arrays to 0-1 scale if needed
    # if red.max() > 1 and nir.max() > 1:
    #     red = red / 255
    #     nir = nir / 255
    # elif red.max() > 1 or nir.max() > 1:
    #     raise ValueError()
    red = red / 255
    nir = nir / 255

    ndvi_values = (nir - red) / (nir + red)
    ndvi_values[(red + nir) == 0] = 0

    return ndvi_values


def gndvi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return np.where((green + nir) == 0, 0, (nir - green) / (nir + green))


def water(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return np.where((green + nir) == 0, 0, (green - nir) / (green + nir))


def builtup(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return np.where((swir + nir) == 0, 0, (swir - nir) / (swir + nir))


def moisture(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return np.where((swir + nir) == 0, 0, (nir - swir) / (nir + swir))


def get_raster_pairs(
    cube,
    band_name1: str,
    band_name2: str,
) -> list[tuple[Raster, Raster]]:
    unique = pd.DataFrame({"tile": cube.tile, "date": cube.date}).drop_duplicates(
        ["tile", "date"]
    )

    raster_pairs = []
    for tile, date in zip(unique["tile"], unique["date"], strict=False):
        query = (cube.tile == tile) & (cube.date == date)
        band1 = cube.copy()[query & (cube.band == band_name1)]
        band2 = cube.copy()[query & (cube.band == band_name2)]
        if not len(band1) and not len(band2):
            continue
        if len(band1) > 1:
            raise ValueError("Cannot have more than one B4 band per tile.")
        if len(band2) > 1:
            raise ValueError("Cannot have more than one B8 band per tile.")
        if len(band1) != 1 and len(band2) != 1:
            raise ValueError("Must have one B4 and one B8 band per tile.")
        band1 = band1[0]
        band2 = band2[0]
        assert isinstance(band1, Raster), band1
        assert isinstance(band2, Raster), band2

        if band1.shape != band2.shape:
            raise ValueError("Rasters must have same shape")

        pair = band1, band2
        raster_pairs.append(pair)

    return raster_pairs


def index_calc_pair(
    raster_pair: tuple[Raster, Raster], index_formula: Callable
) -> Raster:
    """Calculate an index for one raster pair and return a single Raster."""
    r1, r2 = raster_pair
    assert isinstance(r1, Raster), r1
    assert isinstance(r2, Raster), r2

    if r1.values is None:
        r1 = r1.load()
    if r2.values is None:
        r2 = r2.load()

    r1_arr: np.ndarray = r1.values.astype(np.float16)
    r2_arr: np.ndarray = r2.values.astype(np.float16)

    out_array = index_formula(r1_arr, r2_arr)

    return Raster.from_array(
        out_array,
        crs=r1.crs,
        bounds=r1.bounds,
        name=index_formula.__name__,
        date=r1.date,
    )
