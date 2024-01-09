import functools
import itertools
from numbers import Number
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyproj
from geopandas import GeoDataFrame

from ..geopandas_tools.conversion import to_shapely
from ..geopandas_tools.general import get_common_crs
from ..helpers import get_all_files, get_func_name, get_non_numpy_func_name, in_jupyter
from .base import RasterBase
from .indices import get_raster_pairs, index_calc_pair
from .merge import cube_merge
from .raster import Raster


def intersection_base(row: pd.Series, cube, **kwargs):
    cube = cube.copy()
    geom = row.pop("geometry")
    cube = cube.clip(geom, **kwargs)
    for key, value in row.items():
        cube.df[key] = value
    return cube


def _cube_merge(cubebounds, **kwargs):
    assert isinstance(cubebounds, dict)
    return cube_merge(cube=cubebounds["cube"], bounds=cubebounds["bounds"], **kwargs)


def _method_as_func(self, method, **kwargs):
    return getattr(self, method)(**kwargs)


def _astype_raster(raster, raster_type):
    """Returns raster as another raster type."""
    return raster_type(raster)


def _raster_from_path(path, raster_type, band_index, **kwargs):
    return raster_type.from_path(path, band_index=band_index, **kwargs)


def _from_gdf_func(gdf, raster_type, **kwargs):
    return raster_type.from_gdf(gdf, **kwargs)


def _to_gdf_func(raster, **kwargs):
    return raster.to_gdf(**kwargs)


def _write_func(raster, **kwargs):
    path = str(Path(raster._out_folder) / Path(raster._filename).stem) + ".tif"
    raster.write(path, **kwargs)
    raster.path = path
    return raster


def _clip_func(raster, mask, **kwargs):
    return raster.clip(mask, **kwargs)


def _clip_func(raster, mask, **kwargs):
    return raster.clip(mask, **kwargs)


def _load_func(raster, **kwargs):
    return raster.load(**kwargs)


def _to_crs_func(raster, **kwargs):
    return raster.to_crs(**kwargs)


def _set_crs_func(raster, **kwargs):
    return raster.set_crs(**kwargs)


def _array_astype_func(array, dtype):
    return array.astype(dtype)


def _add(raster, scalar):
    return raster + scalar


def _mul(raster, scalar):
    return raster * scalar


def _sub(raster, scalar):
    return raster - scalar


def _truediv(raster, scalar):
    return raster / scalar


def _floordiv(raster, scalar):
    return raster // scalar


def _pow(raster, scalar):
    return raster**scalar
