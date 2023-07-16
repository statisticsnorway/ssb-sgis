import functools
from typing import Callable

import numpy as np

from .cube import GeoDataCube, concat_cubes, starmap_concat
from .raster import Raster


def ndvi_formula(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    f = (nir - red) / (nir + red)
    return np.where((red + nir) == 0, 0, f)


def gndvi_formula(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    f = (nir - green) / (nir + green)
    return np.where((green + nir) == 0, 0, f)


def water_formula(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    f = (green - nir) / (green + nir)
    return np.where((green + nir) == 0, 0, f)


def water_formula(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    f = (nir - swir) / (nir + swir)
    return np.where((swir + nir) == 0, 0, f)


def builtup_formula(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    f = (swir - nir) / (swir + nir)
    return np.where((swir + nir) == 0, 0, f)


def moisture_formula(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    f = (nir - swir) / (nir + swir)
    return np.where((swir + nir) == 0, 0, f)


def moisture_index(
    cube: GeoDataCube, band_name_swir="B11", band_name_nir="B8", copy=True
):
    return index_calc(
        cube,
        band_name1=band_name_swir,
        band_name2=band_name_nir,
        func=moisture_formula,
        index_name="moisture_index",
        copy=copy,
    )


def water_index(cube: GeoDataCube, band_name_green="B3", band_name_nir="B8", copy=True):
    return index_calc(
        cube,
        band_name1=band_name_green,
        band_name2=band_name_nir,
        func=water_formula,
        index_name="water_index",
        copy=copy,
    )


def ndvi_index(cube: GeoDataCube, band_name_red="B4", band_name_nir="B8", copy=True):
    return index_calc(
        cube,
        band_name1=band_name_red,
        band_name2=band_name_nir,
        func=ndvi_formula,
        index_name="NDVI",
        copy=copy,
    )


def gndvi_index(cube: GeoDataCube, band_name_green="B3", band_name_nir="B8", copy=True):
    return index_calc(
        cube,
        band_name1=band_name_green,
        band_name2=band_name_nir,
        func=gndvi_formula,
        index_name="gndvi",
        copy=copy,
    )


def builtup_index(cube: GeoDataCube, band_name_red="B4", band_name_nir="B8", copy=True):
    return index_calc(
        cube,
        band_name1=band_name_red,
        band_name2=band_name_nir,
        func=builtup_formula,
        index_name="builtup_index",
        copy=copy,
    )


def index_calc(
    cube: GeoDataCube,
    band_name1: str,
    band_name2: str,
    func: Callable,
    index_name: str,
    copy=True,
):
    if copy:
        cube = cube.copy()

    cube._df["tile"] = cube.tile.values
    cube._df["date"] = cube.date.values

    unique = cube.df.drop_duplicates(["tile", "date"])

    raster_pairs = []
    for tile, date in zip(unique["tile"], unique["date"]):
        query = (cube.df["tile"] == tile) & (cube.df["date"] == date)
        red = cube.df.loc[query & (cube.band_name == band_name1), "raster"]
        nir = cube.df.loc[query & (cube.band_name == band_name2), "raster"]
        if len(red) > 1:
            raise ValueError("Cannot have more than one B4 band per tile.")
        if len(nir) > 1:
            raise ValueError("Cannot have more than one B8 band per tile.")
        if len(red) != 1 and len(nir) != 1:
            raise ValueError("Must have one B4 and one B8 band per tile.")
        red = red.iloc[0]
        nir = nir.iloc[0]
        assert isinstance(red, Raster), red
        assert isinstance(nir, Raster), nir

        if red.shape != nir.shape:
            raise ValueError("Rasters must have same shape")

        pair = red, nir
        raster_pairs.append(pair)

    index_calc_partial = functools.partial(
        index_calc_pair, func=func, index_name=index_name
    )

    if cube._chain is not None:
        partial_func = functools.partial(starmap_concat, func=index_calc_partial)
        cube._chain.append_cube_iter(partial_func, iterable=raster_pairs)
        return cube

    cubes = [index_calc_partial(*items) for items in raster_pairs]
    return concat_cubes(cubes, ignore_index=True)


def index_calc_pair(
    r1: Raster, r2: Raster, func: Callable, index_name: str
) -> GeoDataCube:
    """Calculate an index for one raster pair to create a one-row Cube."""
    assert isinstance(r1, Raster), r1
    assert isinstance(r2, Raster), r2

    if r1.array is None:
        r1 = r1.load()
    if r2.array is None:
        r2 = r2.load()

    r1_arr: np.ndarray = r1.array.astype(np.float16)
    r2_arr: np.ndarray = r2.array.astype(np.float16)

    out_array = func(r1_arr, r2_arr)

    raster = Raster.from_array(
        out_array, crs=r1.crs, bounds=r1.bounds, band_name=index_name, date=r1.date
    )

    return GeoDataCube(raster)
