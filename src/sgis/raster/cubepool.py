import functools
import multiprocessing
from typing import Any, Callable, Iterable

import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame, Series
from pandas.api.types import is_list_like
from rasterio import merge
from rasterio.enums import MergeAlg
from shapely import Geometry

from ..geopandas_tools.bounds import make_grid, to_bbox
from ..geopandas_tools.general import get_common_crs, to_shapely
from ..geopandas_tools.to_geodataframe import to_gdf
from ..helpers import (
    get_all_files,
    get_func_name,
    get_non_numpy_func_name,
    get_numpy_func,
    in_jupyter,
)
from ..multiprocessing.base import LocalFunctionError
from .cubebase import (
    CubeBase,
    _add,
    _array_astype_func,
    _astype_raster,
    _clip_func,
    _cube_merge,
    _floordiv,
    _from_gdf_func,
    _load_func,
    _method_as_func,
    _mul,
    _pow,
    _raster_from_path,
    _set_crs_func,
    _sub,
    _to_crs_func,
    _to_gdf_func,
    _truediv,
    _write_func,
    intersection_base,
)
from .cubechain import CubeChain
from .elevationraster import ElevationRaster
from .explode import explode
from .indices import get_raster_pairs, index_calc_pair, ndvi_formula
from .merge import cube_merge, merge_by_bounds
from .raster import Raster
from .sentinel import Sentinel2
from .zonal import make_geometry_iterrows, prepare_zonal, zonal_func, zonal_post


def rasters_to_cube(rasters, cube):
    return cube.__class__(rasters)


def make_iterrows(df):
    return [row for _, row in df.iterrows()]


def concat_cubes(
    cube_list,
    ignore_index: bool = False,
):
    """TODO: TEMP..."""
    cube = cube_list[0].__class__()
    cube._crs = get_common_crs(cube_list)

    cube.df = pd.concat([cube.df for cube in cube_list], ignore_index=ignore_index)

    return cube


class CubePool(CubeBase):
    def __init__(self, cube, processes) -> None:
        self.cube = cube
        self.processes = processes
        self.funcs = []
        self.names = []
        self.in_types = []
        self.out_types = []

        self.write_in_chain = False

    def execute(self):
        if not self.funcs:
            raise ValueError("Execution chain is of length 0.")

        cube = self.cube

        with multiprocessing.get_context("spawn").Pool(self.processes) as pool:
            for func, in_type, out_type in self:
                if in_type == "cube":
                    out = func(cube)
                elif in_type == "array":
                    out = pool.map(func, cube.arrays)
                elif in_type == "raster":
                    out = pool.map(func, cube.df["raster"])
                elif in_type == "iterable":
                    try:
                        partial_func = functools.partial(func, cube=cube)
                        out = pool.map(partial_func, iterable=out)
                    except TypeError:
                        out = pool.map(func, iterable=out)
                elif in_type == "other":
                    try:
                        out = func(out, cube=cube)
                    except TypeError:
                        out = func(out)
                    except UnboundLocalError:
                        out = func()
                else:
                    raise ValueError(out_type)

                if out_type == "cube":
                    cube = out
                elif out_type == "array":
                    cube.arrays = out
                elif out_type == "raster":
                    cube.df["raster"] = out
                elif out_type not in ["iterable", "other"]:
                    raise ValueError(out_type)

                cube.update_df()

        if out_type in ["iterable", "other"]:
            return out

        return cube

    def append_func(self, func, in_type: str, out_type: str, **kwargs):
        self.chekkkk()
        self._append_method_or_func(func, **kwargs)
        self.in_types.append(in_type)
        self.out_types.append(out_type)

    def append_cube_func(self, func, **kwargs):
        self.append_func(func, in_type="cube", out_type="cube", **kwargs)

    def append_raster_func(self, func, **kwargs):
        self.append_func(func, in_type="raster", out_type="raster", **kwargs)

    def append_array_func(self, func, **kwargs):
        self.append_func(func, in_type="array", out_type="array", **kwargs)

    """def append_raster_func(self, func, out_type="cube", **kwargs):
        self.chekkkk()
        self._append_method_or_func(func, **kwargs)
        self.in_types.append("raster")
        self.out_types.append(out_type)

    def append_cube_func(self, func, out_type="cube", **kwargs):
        self.chekkkk()
        self._append_method_or_func(func, **kwargs)
        self.in_types.append("cube")
        self.out_types.append(out_type)

    def append_iter_func(self, func, out_type="cube", **kwargs):
        self.chekkkk()
        self._append_func(func, **kwargs)
        self.in_types.append("iter")
        self.out_types.append(out_type)

    def append_wrapping_up(self, func, out_type="cube", **kwargs):
        self.chekkkk()
        self._append_func(func, **kwargs)
        self.in_types.append("wrapping_up")
        self.out_types.append(out_type)

    def append_make_iter(self, func, out_type="cube", **kwargs):
        self.chekkkk()
        self._append_func(func, **kwargs)
        self.in_types.append("make_iter")
        self.out_types.append(out_type)"""

    def chekkkk(self):
        if hasattr(self, "_not_returning_cube"):
            raise ValueError(
                "Cannot keep chaining to pool after a non-GeoDataCube is returned."
            )

    def _append_method_or_func(self, methfunc: str | Callable, **kwargs):
        if isinstance(methfunc, str):
            self._append_func(_method_as_func, method=methfunc, **kwargs)
        elif callable(methfunc):
            self._append_func(methfunc, **kwargs)
        else:
            raise TypeError(methfunc)

    def _append_func(self, func, **kwargs):
        assert callable(func)
        func_name = get_func_name(func)
        if "write" in func_name:
            if self.write_in_chain:
                raise ValueError("Cannot keep chain going after writing files.")
            else:
                self.write_in_chain = True

        if "to_gdf" in func_name:
            if self.to_gdf_in_chain:
                raise ValueError("Cannot keep chain going after to_gdf.")
            else:
                self.to_gdf_in_chain = True

        func = functools.partial(func, **kwargs)
        self.funcs.append(func)

    def __bool__(self):
        if len(self.funcs):
            return True
        return False

    def __len__(self):
        return len(self.funcs)

    def __iter__(self):
        return iter(
            [
                (func, in_type, out_type)
                for func, in_type, out_type in zip(
                    self.funcs,
                    self.in_types,
                    self.out_types,
                    strict=True,
                )
            ]
        )

    def raster_astype(self, raster_type: type):
        self.append_raster_func(_astype_raster, raster_type=raster_type)
        return self

    def astype(self, dtype: type):
        self.append_array_func(_array_astype_func, dtype=dtype)
        return self

    def as_mimimum_dtype(self):
        self.append_cube_func("as_mimimum_dtype")
        return self

    def zonal(
        self,
        polygons: GeoDataFrame,
        aggfunc: str | Callable | list[Callable | str],
        array_func: Callable | None = None,
        by_date: bool = True,
        dropna: bool = True,
    ) -> GeoDataFrame:
        idx_mapper, idx_name = self.get_index_mapper(polygons)
        polygons, aggfunc, func_names = prepare_zonal(polygons, aggfunc)
        self.append_func(
            make_geometry_iterrows,
            gdf=polygons,
            in_type="other",
            out_type="iterable",
        )

        self.append_func(
            zonal_func,
            array_func=array_func,
            aggfunc=aggfunc,
            func_names=func_names,
            by_date=by_date,
            in_type="iterable",
            out_type="other",
        )
        self.append_func(
            zonal_post,
            polygons=polygons,
            idx_mapper=idx_mapper,
            idx_name=idx_name,
            dropna=dropna,
            in_type="other",
            out_type="other",
        )

        return self

    def ndvi(self, band_name_red, band_name_nir):
        return self._index_calc_pool(
            band_name1=band_name_red,
            band_name2=band_name_nir,
            index_formula=ndvi_formula,
            index_name="ndvi",
        )

    def _index_calc_pool(
        self,
        band_name1,
        band_name2,
        index_formula: Callable,
        index_name: str,
    ):
        self.append_func(
            get_raster_pairs,
            band_name1=band_name1,
            band_name2=band_name2,
            in_type="cube",
            out_type="iterable",
        )
        """index_calc = functools.partial(
            index_calc_pair, index_formula=index_formula, index_name=index_name
        )
        def _index_calc(raster_pair: tuple[Raster, Raster])"""
        self.append_func(
            index_calc_pair,
            index_formula=index_formula,
            index_name=index_name,
            in_type="iterable",
            out_type="other",
        )

        self.append_func(rasters_to_cube, in_type="other", out_type="cube")
        return self

    def gradient(self, degrees: bool = False):
        if not all(isinstance(r, ElevationRaster) for r in self.cube):
            raise TypeError("raster_type must be ElevationRaster.")
        self.append_raster_func("gradient", degrees=degrees)
        return self

    def array_map(self, func: Callable, **kwargs):
        """Maps each raster array to a function.

        The function must take a numpu array as first positional argument,
        and return a single numpy array. The function should be defined in
        the leftmost indentation level. If in Jupyter, the function also
        have to be defined in and imported from another file.
        """

        if func.__module__ == "__main__" and in_jupyter():
            raise LocalFunctionError(func)

        self.append_array_func(func, **kwargs)
        return self

    def query(self, query: str, **kwargs):
        self.append_cube_func("query", query=query, **kwargs)
        return self

    def load(self, res: int | None = None, **kwargs):
        self.append_raster_func("load", res=res, **kwargs)
        return self

    def clip(self, mask, res: int | None = None, **kwargs):
        self.append_cube_func("clip_base", mask=mask)
        self.append_raster_func("clip", mask=mask, res=res, **kwargs)
        return self

    def intersection(self, df, res: int | None = None, **kwargs):
        self.append_func(
            make_iterrows,
            df=df,
            in_type="other",
            out_type="iterable",
        )
        self.append_func(
            intersection_base,
            res=res,
            **kwargs,
            in_type="iterable",
            out_type="other",
        )

        self.append_func(
            concat_cubes,
            ignore_index=True,
            in_type="other",
            out_type="cube",
        )
        return self

    def gradient(self, degrees: bool = False, copy: bool = False):
        if not all(isinstance(r, ElevationRaster) for r in self.cube):
            raise TypeError("raster_type must be ElevationRaster.")
        self.append_raster_func("gradient", degrees=degrees, copy=copy)
        return self

    def sample(
        self, n=1, buffer=1000, mask=None, crop=True, copy: bool = True, **kwargs
    ):
        pass

    def write(
        self,
        root: str,
        filename: str = "raster_id",
        subfolder_col: str | None = None,
        **kwargs,
    ):
        """Writes arrays as tif files and df with file info.

        This method should be run after the rasters have been clipped, merged or
        its array values have been recalculated.

        Args:
            subfolders: Column of the cube's df to use as subfolder below root.
                Must be a string column. Missing values will be placed in root.

        """
        self.append_cube_func(
            "write_base", subfolder_col=subfolder_col, filename=filename, root=root
        )
        self.append_raster_func(_write_func, **kwargs)
        return self

    def to_gdf(self, ignore_index: bool = False, concat: bool = True):
        self.append_cube_func("assign_datadict_to_rasters")
        self.append_raster_func("to_gdf")

        if concat:
            self.append_func(
                pd.concat, ignore_index=ignore_index, in_type="other", out_type="other"
            )
        return self

    def filter_by_location(self, other):
        self.append_cube_func("filter_by_location")
        return self

    def to_crs(self, crs):
        self.append_raster_func("to_crs", crs=crs)
        return self

    def set_crs(self, crs, allow_override: bool = False):
        self.append_raster_func("set_crs", crs=crs, allow_override=allow_override)
        return self

    def explode(self, ignore_index: bool = False):
        self.append_cube_func(explode, ignore_index=ignore_index)
        return self

    def merge(
        self,
        **kwargs,
    ):
        self.append_cube_func(cube_merge, **kwargs)
        return self

    def merge_by_bounds(
        self,
        **kwargs,
    ):
        self.append_cube_func(merge_by_bounds, **kwargs)
        return self

    def dissolve_bands(self, aggfunc):
        if not callable(aggfunc) and not isinstance(aggfunc, str):
            raise TypeError("Can only supply a single aggfunc")

        aggfunc = get_numpy_func(aggfunc)

        if aggfunc.__module__ == "__main__" and in_jupyter():
            raise LocalFunctionError(aggfunc)

        self.append_array_func(aggfunc, axis=0)
        return self

    def run_raster_method(self, method: str, **kwargs):
        """Run Raster methods."""
        if not all(hasattr(r, method) for r in self.cube):
            raise AttributeError(f"Raster has no method {method!r}.")

        method_as_func = functools.partial(_method_as_func, method=method, **kwargs)

        self.append_raster_func(method_as_func)
        return self

    def __mul__(self, scalar):
        self.append_array_func(_mul, scalar=scalar)
        return self

    def __add__(self, scalar):
        self.append_array_func(_add, scalar=scalar)
        return self

    def __sub__(self, scalar):
        self.append_array_func(_sub, scalar=scalar)
        return self

    def __truediv__(self, scalar):
        self.append_array_func(_truediv, scalar=scalar)
        return self

    def __floordiv__(self, scalar):
        self.append_array_func(_floordiv, scalar=scalar)
        return self

    def __pow__(self, scalar):
        self.append_array_func(_pow, scalar=scalar)
        return self
