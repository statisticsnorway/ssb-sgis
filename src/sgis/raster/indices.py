from typing import Callable

import numpy as np

from .raster import Raster


def ndvi_formula(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return np.where((red + nir) == 0, 0, (nir - red) / (nir + red))


def gndvi_formula(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return np.where((green + nir) == 0, 0, (nir - green) / (nir + green))


def water_formula(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return np.where((green + nir) == 0, 0, (green - nir) / (green + nir))


def water_formula(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return np.where((swir + nir) == 0, 0, (nir - swir) / (nir + swir))


def builtup_formula(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return np.where((swir + nir) == 0, 0, (swir - nir) / (swir + nir))


def moisture_formula(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return np.where((swir + nir) == 0, 0, (nir - swir) / (nir + swir))


def get_raster_pairs(
    cube,
    band_name1: str,
    band_name2: str,
):
    cube._df["tile"] = cube.tile.values
    cube._df["date"] = cube.date.values

    unique = cube.df.drop_duplicates(["tile", "date"])

    raster_pairs = []
    for tile, date in zip(unique["tile"], unique["date"]):
        query = (cube.df["tile"] == tile) & (cube.df["date"] == date)
        band1 = cube.df.loc[query & (cube.name == band_name1), "raster"]
        band2 = cube.df.loc[query & (cube.name == band_name2), "raster"]
        if not len(band1) and not len(band2):
            continue
        if len(band1) > 1:
            raise ValueError("Cannot have more than one B4 band per tile.")
        if len(band2) > 1:
            raise ValueError("Cannot have more than one B8 band per tile.")
        if len(band1) != 1 and len(band2) != 1:
            raise ValueError("Must have one B4 and one B8 band per tile.")
        band1 = band1.iloc[0]
        band2 = band2.iloc[0]
        assert isinstance(band1, Raster), band1
        assert isinstance(band2, Raster), band2

        if band1.shape != band2.shape:
            raise ValueError("Rasters must have same shape")

        pair = band1, band2
        raster_pairs.append(pair)

    return raster_pairs


def index_calc_pair(
    raster_pair: tuple[Raster, Raster], index_formula: Callable, index_name: str
) -> Raster:
    """Calculate an index for one raster pair and return a single Raster."""
    r1, r2 = raster_pair
    assert isinstance(r1, Raster), r1
    assert isinstance(r2, Raster), r2

    if r1.array is None:
        r1 = r1.load()
    if r2.array is None:
        r2 = r2.load()

    r1_arr: np.ndarray = r1.array.astype(np.float16)
    r2_arr: np.ndarray = r2.array.astype(np.float16)

    out_array = index_formula(r1_arr, r2_arr)

    return Raster.from_array(
        out_array, crs=r1.crs, bounds=r1.bounds, name=index_name, date=r1.date
    )
