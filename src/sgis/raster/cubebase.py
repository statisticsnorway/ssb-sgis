import functools
import itertools
from numbers import Number
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyproj
from geopandas import GeoDataFrame

from ..geopandas_tools.general import get_common_crs, to_shapely
from ..helpers import get_all_files, get_func_name, get_non_numpy_func_name, in_jupyter
from .base import RasterBase
from .indices import get_raster_pairs, index_calc_pair
from .merge import cube_merge
from .raster import Raster


def intersection_base(row: pd.Series, cube, res: int, **kwargs):
    cube = cube.copy()
    geom = row.pop("geometry")
    cube = cube.clip(geom, res=res, **kwargs)
    for key, value in row.items():
        cube.df[key] = value
    return cube


class CubeBase(RasterBase):
    def clip_base(self, mask):
        if (
            hasattr(mask, "crs")
            and mask.crs
            and not pyproj.CRS(self.crs).equals(pyproj.CRS(mask.crs))
        ):
            raise ValueError("crs mismatch.")

        # first remove rows not within mask
        self._df = self._df.loc[self.boxes.intersects(to_shapely(mask))]

        return self

    def assign_datadict_to_rasters(self):
        for raster, (_, row) in zip(self.df["raster"], self.df.iterrows()):
            raster._datadict = row.drop("raster").to_dict()

        return self

    def write_base(self, subfolder_col, filename, root):
        if self._chain is None:
            self.check_for_array()

        if self.df["name"].isna().any():
            raise ValueError(
                "Cannot have missing values in 'name' column when writing."
            )

        if self.df["name"].duplicated().any():
            raise ValueError("Cannot have duplicate names when writing files.")

        self.validate_self_df(self.df)

        if subfolder_col:
            for raster, folder in zip(self.df["raster"], self.df[subfolder_col]):
                raster._out_folder = str(Path(root) / Path(folder))
        else:
            for raster in self.df["raster"]:
                raster._out_folder = str(Path(root))

        if filename in self.BASE_CUBE_COLS:
            return self

        for raster, name in zip(self.df["raster"], self.df[filename]):
            raster._filename = name

        return self

    def get_index_mapper(self, df):
        idx_mapper = dict(enumerate(df.index))
        idx_name = df.index.name
        return idx_mapper, idx_name

    def zonal_func(self, poly_iter, array_func, aggfunc, func_names):
        i, polygon = poly_iter
        clipped = self.clipmerge(polygon)
        assert len(clipped) == 1
        array = clipped[0].array
        if array_func:
            array = array_func(array)
        flat_array = array.flatten()
        no_nans = flat_array[~np.isnan(flat_array)]
        data = {}
        for f, name in zip(aggfunc, func_names, strict=True):
            num = f(no_nans)
            data[name] = num
        return pd.DataFrame(data, index=[i])

    def _index_calc(
        self,
        band_name1,
        band_name2,
        index_formula: Callable,
        index_name: str,
        copy=True,
    ):
        cube = self.copy() if copy else self

        raster_pairs: list[tuple[Raster, Raster]] = get_raster_pairs(
            cube, band_name1=band_name1, band_name2=band_name2
        )

        index_calc = functools.partial(
            index_calc_pair, index_formula=index_formula, index_name=index_name
        )

        rasters = [index_calc(items) for items in raster_pairs]

        return cube.__class__(rasters)


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
