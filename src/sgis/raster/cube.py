import functools
import glob
import itertools
import multiprocessing
import os
import re
import uuid
import warnings
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame, Series
from pandas.api.types import is_list_like
from rasterio import merge
from rasterio.enums import MergeAlg
from shapely import Geometry

from ..geopandas_tools.bounds import make_grid, to_bbox
from ..geopandas_tools.general import get_common_crs, to_shapely
from ..geopandas_tools.to_geodataframe import to_gdf
from ..helpers import get_all_files
from ..io.dapla import check_files, is_dapla, read_geopandas, write_geopandas
from ..multiprocessing.multiprocessingmapper import MultiProcessingMapper
from .base import RasterBase
from .cubechain import CubeChain
from .elevationraster import ElevationRaster
from .explode import explode
from .merge import cube_merge, merge_by_bounds
from .raster import Raster, get_numpy_func
from .sentinel import Sentinel2


# TODO: hvordan blir kolonnene etter merge/retile?
# etter retile:
# - band_name må beholdes
# - name må være band_name+tile - altså groupby+grid_id
#     - er det noe annet enn band_name man vil groupe by? Nei?
# - må lagres i mappene de lå i, altså folder_name
# -
# - Merge by [band_name, folder_name] ??

# name:
# - raster: tile+band_name+date

# TODO: raster has property navn, mens band_name kan settes?

CANON_RASTER_TYPES = {
    "Raster": Raster,
    "ElevationRaster": ElevationRaster,
    "Sentinel2": Sentinel2,
}

CUBE_DF_NAME = "cube_df.parquet"


class GeoDataCube(RasterBase):
    _chain = None
    _executing = False
    _read_in_chain = True

    def __init__(
        self,
        data: Raster | Iterable[Raster] | None = None,
        df: DataFrame | None = None,
        root: str | None = None,
        copy: bool = False,
    ) -> None:
        self._arrays = None
        self._crs = None
        self._hash = uuid.uuid4()
        self.root = root

        if data is None:
            self._df = self.get_cube_template()
            return

        if isinstance(data, GeoDataCube):
            for key, value in data.__dict__.items():
                self[key] = value
            return

        if isinstance(data, Raster):
            data = [data]
        if not is_list_like(data) and all(isinstance(r, Raster) for r in data):
            raise TypeError("'data' must be a Raster instance or an iterable.")

        if copy:
            data = [raster.copy() for raster in data]
        else:
            # take a copy only if there are gdfs with the same id
            if sum(r1 is r2 for r1 in data for r2 in data) > len(data):
                data = [raster.copy() for raster in data]

        if df is not None and len(df) != len(data):
            raise ValueError("'df' must be same length as data.")

        self._df = self.make_cube_df(data, df=df, root=root)
        self._crs = get_common_crs(self.df["raster"])

    @classmethod
    def from_root(
        cls,
        root: str | Path,
        *,
        band_index: int | list[int] | None = None,
        raster_dtype: Raster = Raster,
        check_for_df: bool = True,
        contains: str | None = None,
        endswith: str = ".tif",
        regex: str | None = None,
        **kwargs,
    ):
        kwargs = {
            "raster_dtype": raster_dtype,
        } | kwargs
        if is_dapla():
            paths = list(check_files(root, contains=contains)["path"])
        else:
            paths = get_all_files(root)

        dfs = [path for path in paths if path.endswith(CUBE_DF_NAME)]

        if contains:
            paths = [path for path in paths if contains in path]
        if endswith:
            paths = [path for path in paths if path.endswith(endswith)]
        if regex:
            regex = re.compile(regex)
            paths = [path for path in paths if re.search(regex, path)]

        if not paths:
            raise ValueError("Found no files matching the pattern.")

        if not check_for_df or not len(dfs):
            return cls.from_paths(paths, band_index=band_index, root=root, **kwargs)

        folders_with_df = {Path(path).parent for path in dfs if path}
        if len(dfs) != len(folders_with_df):
            raise ValueError(
                "More than one cube_df.parquet path found in at least one folder."
            )

        cubes = [cls.from_cube_df(df, **kwargs) for df in dfs]

        paths_in_folders_without_df = [
            path for path in paths if Path(path).parent not in folders_with_df
        ]

        if paths_in_folders_without_df:
            cubes += [
                cls.from_paths(
                    paths_in_folders_without_df,
                    band_index=band_index,
                    root=root,
                    **kwargs,
                )
            ]

        if len(cubes) == 1:
            return cubes[0]

        cube = concat_cubes(cubes, ignore_index=True)

        cube._from_cube_df = True
        return cube

    @classmethod
    def from_paths(
        cls,
        paths: list[str | Path],
        *,
        root: str | None = None,
        raster_dtype: Raster = Raster,
        band_index: int | tuple[int] | None = None,
        **kwargs,
    ):
        if not isinstance(raster_dtype, type):
            raise TypeError("raster_dtype must be Raster or a subclass.")

        if not issubclass(raster_dtype, Raster):
            raise TypeError("raster_dtype must be Raster or a subclass.")

        if not is_list_like(paths) and not all(
            isinstance(path, (str, Path)) for path in paths
        ):
            raise TypeError

        rasters = [
            raster_dtype.from_path(
                path,
                band_index=band_index,
            )
            for path in paths
        ]

        return cls(
            rasters,
            root=root,
            **kwargs,
        )

    @classmethod
    def from_gdf(
        cls,
        gdf: GeoDataFrame | list[GeoDataFrame],
        columns: str | list[str],
        res: int,
        processes: int,
        tilesize: int | None = None,
        tiles: GeoSeries | None = None,
        raster_dtype: Raster = Raster,
        fill=0,
        all_touched=False,
        merge_alg=MergeAlg.replace,
        default_value=1,
        dtype=None,
        **kwargs,
    ):
        if tiles is None and tilesize is None:
            raise ValueError("Must specify either 'tilesize' or 'tiles'.")

        if isinstance(gdf, GeoDataFrame):
            gdf = [gdf]
        if not all(isinstance(frame, GeoDataFrame) for frame in gdf):
            raise TypeError

        if tiles is None:
            crs = get_common_crs(gdf)
            total_bounds = shapely.unary_union(
                [shapely.box(*frame.total_bounds) for frame in gdf]
            )
            tiles = make_grid(total_bounds, gridsize=tilesize, crs=crs)

        tiles["tile_idx"] = range(len(tiles))

        partial_func = functools.partial(
            _from_gdf_func,
            columns=columns,
            res=res,
            fill=fill,
            all_touched=all_touched,
            merge_alg=merge_alg,
            default_value=default_value,
            dtype=dtype,
            raster_dtype=raster_dtype,
            **kwargs,
        )

        def get_gdf_list(gdf):
            return [gdf.loc[gdf["tile_idx"] == i] for i in gdf["tile_idx"].unique()]

        rasters = []

        if processes > 1:
            with multiprocessing.get_context("spawn").Pool(processes) as p:
                for frame in gdf:
                    frame = frame.overlay(tiles, keep_geom_type=True)
                    gdfs = get_gdf_list(frame)
                    rasters += p.map(partial_func, gdfs)
        elif processes < 1:
            raise ValueError("processes must be an integer 1 or greater.")
        else:
            for frame in gdf:
                frame = frame.overlay(tiles, keep_geom_type=True)
                gdfs = get_gdf_list(frame)
                rasters += [partial_func(gdf) for gdf in gdfs]

        return cls(rasters)

    @classmethod
    def from_cube_df(
        cls,
        df: DataFrame | str | Path,
        raster_dtype: Raster = Raster,
    ):
        raster_dtype = cls.get_raster_dtype(raster_dtype)

        if isinstance(df, (str, Path)):
            df = read_geopandas(df) if is_dapla() else gpd.read_parquet(df)

        if isinstance(df, DataFrame):
            raster_attrs, other_attrs = cls._prepare_gdf_for_raster(df)
            rasters = [
                raster_dtype.from_dict(dict(row[1])) for row in raster_attrs.iterrows()
            ]
            cube = cls(rasters, df=other_attrs)
            cube._from_cube_df = True
            return cube

        elif all(isinstance(x, (str, Path, DataFrame)) for x in df):
            cubes = [cls.from_cube_df(x, raster_dtype=raster_dtype) for x in df]
            cube = concat_cubes(cubes, ignore_index=True)
            cube._from_cube_df = True
            return cube

        raise TypeError("df must be DataFrame or file path to a parquet file.")

    def update_rasters(self, *args):
        if not all(isinstance(arg, str) for arg in args):
            raise TypeError("Arguments must be strings.")
        if not all(arg in self.df for arg in args):
            raise KeyError

        rasters = self._df["raster"]
        updates = self._df[[*args]]

        self._df["raster"] = [
            raster.update(dict(row[1]))
            for raster, row in zip(rasters, updates.iterrows())
        ]

        return self

    @staticmethod
    def get_raster_dtype(raster_dtype):
        if not isinstance(raster_dtype, type):
            if isinstance(raster_dtype, str) and raster_dtype in CANON_RASTER_TYPES:
                return CANON_RASTER_TYPES[raster_dtype]
            else:
                raise TypeError("'raster_dtype' must be Raster or a subclass.")

        if not issubclass(raster_dtype, Raster):
            raise TypeError("'raster_dtype' must be Raster or a subclass.")

        return raster_dtype

    def astype(self, dtype: type):
        if self._is_chaining():
            self._chain.append_raster_func(_astype_raster, dtype=dtype)
            # self._chain.append_method("astype", dtype=dtype)
            return self

        dtype = self.get_raster_dtype(dtype)

        self._df["raster"] = [dtype(raster) for raster in self]

        return self

    def astype_array(self, dtype):
        if self._chain is not None:
            self._chain.append_array_func(_array_astype_func, dtype=dtype)
            return self
        self.check_for_array("dtype can be set as a parameter in load and clip.")
        self.arrays = [_array_astype_func(arr, dtype=dtype) for arr in self.arrays]
        return self
        self._delegate_array_func(_array_astype_func, dtype=dtype)

    def zonal(
        self,
        polygons: GeoDataFrame,
        aggfunc: str | Callable | list[Callable | str],
        raster_calc_func: Callable | None = None,
        dropna: bool = True,
    ) -> GeoDataFrame:
        self.check_not_chain()
        kwargs = {
            "polygons": polygons,
            "aggfunc": aggfunc,
            "raster_calc_func": raster_calc_func,
            "dropna": dropna,
        }
        if self.mapper:
            gdfs: list[GeoDataFrame] = self.mapper.map(
                _zonal_func, self.arrays, **kwargs
            )
        else:
            gdfs: list[GeoDataFrame] = [
                _zonal_func(arr, **kwargs) for arr in self.arrays
            ]

        out = []
        for i, gdf in zip(self._df.index, gdfs):
            gdf["raster_index"] = i
            out.append(gdf)
        return pd.concat(out, ignore_index=True)

    def chain(self, *, processes: int, copy: bool = True):
        """TODO: navn pool? multiprocessing_chain?"""
        if self._chain is not None and len(self._chain):
            warnings.warn("A chain is already started. Starting a new one.")

        if copy:
            self = self.copy()
        self.mapper = MultiProcessingMapper(processes=processes)
        self.processes = processes
        self._chain = CubeChain()
        return self

    def execute(self):
        if self._chain is None:
            raise ValueError("Execution chain hasn't been created.")
        if not self._chain:
            raise ValueError("Execution chain is of length 0.")

        self._executing = True

        with multiprocessing.get_context("spawn").Pool(self.processes) as pool:
            for func, typ, iterable in self._chain:
                if typ == "cube":
                    self = func(self)
                elif typ == "array":
                    self.arrays = pool.map(func, self.arrays)
                elif typ == "raster":
                    self.df["raster"] = pool.map(func, self.df["raster"])
                elif typ == "cube_iter":
                    self = func(pool=pool, iterable=iterable)
                elif typ == "raster_iter":
                    self.df["raster"] = func(pool=pool, iterable=iterable)
                else:
                    raise ValueError

        self._chain = None
        self._executing = False
        return self

    def map(self, func: Callable, **kwargs):
        """Maps each raster array to a function.

        The function must take a numpu array as first positional argument,
        and return a single numpy array. The function should be defined in
        the leftmost indentation level. If in Jupyter, the function also
        have to be defined in and imported from another file.
        """
        if self._chain is not None:
            try:
                self.mapper.validate_execution(func)
            except AttributeError:
                pass
            self._chain.append_array_func(func, **kwargs)
            return self
        self.check_for_array()
        self.arrays = [func(arr, **kwargs) for arr in self.arrays]
        return self
        return self._delegate_array_func(func, **kwargs)

    def query(self, query: str, **kwargs):
        self.df = self.df.query(query, **kwargs)
        return self

    def load(self, res: int | None = None, copy: bool = True, **kwargs):
        if self._chain:
            # self._chain.append_raster_func(_load_func, res=res, **kwargs)
            self._chain.append_raster_method("load", res=res, **kwargs)
            self._read_in_chain = True
            return self

        if copy:
            self = self.copy()

        self.df["raster"] = self.raster_method("load", res=res, **kwargs)
        return self

    def clip(self, mask, res: int | None = None, copy: bool = True, **kwargs):
        if (
            hasattr(mask, "crs")
            and mask.crs
            and not pyproj.CRS(self.crs).equals(pyproj.CRS(mask.crs))
        ):
            raise ValueError("crs mismatch.")

        # first remove rows not within mask
        self._df = self._df.loc[self.boxes.intersects(to_shapely(mask))]

        if not len(self._df):
            return self

        if self._chain is not None:
            # self._chain.append_raster_func(_clip_func, mask=mask, res=res, **kwargs)
            self._chain.append_raster_method("clip", mask=mask, res=res, **kwargs)
            self._read_in_chain = True
            return self

        if copy:
            self = self.copy()

        self.df["raster"] = self.raster_method("clip", mask=mask, res=res, **kwargs)
        return self

    def retile(self, tilesize: int, res: int, band_id="band_name"):
        """"""
        # TODO: blir det konflikter i mp???
        grid = (
            make_grid(self.total_bounds, gridsize=tilesize, crs=self.crs)
            .sjoin(to_gdf(self.unary_union, crs=self.crs))
            .drop("index_right", axis=1)
        )

        grid["grid_id"] = [
            f"{int(minx)}_{int(miny)}" for minx, miny, _, _ in grid.bounds.values
        ]

        self._df = self.df.reset_index(drop=True)

        joined = self.boxes.to_frame().sjoin(grid)

        def get_intersecting_rasters(cube, grid_idx, joined):
            intersecting = joined[joined["index_right"] == grid_idx]
            cube = cube.copy()
            cube._df = cube.df.loc[intersecting.index]
            return cube

        cubes = [get_intersecting_rasters(self, i, joined) for i in grid.index]
        bounds = list(grid.geometry)
        args = [item for item in zip(cubes, bounds)]
        kwargs = {"by": band_id, "res": res}

        if self._chain is not None:
            merge_func = functools.partial(starmap_concat, func=cube_merge, **kwargs)
            self._chain.append_cube_iter(merge_func, iterable=args, **kwargs)
            self._read_in_chain = True
            return self

        cubes = [cube_merge(*items, **kwargs) for items in args]
        cube = concat_cubes(cubes, ignore_index=True)

        return cube

    def ndvi(self, band_name_red, band_name_nir, copy=True):
        return ndvi(
            self, band_name_red=band_name_red, band_name_nir=band_name_nir, copy=copy
        )

    def sample(
        self, n=1, buffer=1000, mask=None, crop=True, copy: bool = True, **kwargs
    ):
        self.check_not_array()
        self.check_not_chain()

        if self._is_chaining():
            self._chain.append_cube_iter(_sample_func, **kwargs)
            self._read_in_chain = True
            return self

        if mask is not None:
            points = (
                GeoSeries(self.unary_union, crs=self.crs)
                .clip(mask, keep_geom_type=False)
                .sample_points(n)
            )
        else:
            points = GeoSeries(self.unary_union, crs=self.crs).sample_points(n)
        buffered = points.buffer(buffer)

        boxes = [shapely.box(*arr) for arr in buffered.bounds.values]

        if self._chain is None:
            self.df["raster"] = [
                _clip_func(r, mask=mask, crop=crop, **kwargs) for r in self
            ]

        if copy:
            return self.copy()._delegate_raster_func(
                _clip_func, mask=mask, crop=crop, **kwargs
            )
        else:
            return self._delegate_raster_func(
                _clip_func, mask=mask, crop=crop, **kwargs
            )

        return RandomCubeSample(cube=self, masks=boxes, crop=crop, **kwargs)

    def write(self, root: str, by_subfolder: bool = True, **kwargs):
        """Writes arrays as tif files and df with file info.

        This method should be run after the rasters have been clipped, merged or
        its array values have been recalculated.
        """
        if self._chain is None:
            self.check_for_array()

        if self.df["name"].isna().any():
            raise ValueError(
                "Cannot have missing values in 'name' column when writing."
            )

        self.validate_cube_df(self.df)

        # trigger df.setter
        self.df = self._df

        if by_subfolder:
            # folders = [Path(root) / r.subfolder for r in self]
            folders = [
                Path(root) / subfolder if subfolder else Path(root)
                for subfolder in self.df["subfolder"]
            ]
        else:
            folders = [Path(root) for _ in self]

        rasters = list(self)
        args = [item for item in zip(rasters, folders)]

        if self._chain is not None:
            write_func = functools.partial(starmap_concat, func=_write_func, **kwargs)
            self._chain.append_raster_iter(write_func, iterable=args, **kwargs)
            self._read_in_chain = True
            return self

        rasters = [starmap_concat(*items, **kwargs) for items in args]
        self.df["raster"] = rasters

        return self

        if self._is_chaining():
            self._chain.append_raster_func(_write_func, folder=folder, **kwargs)
            return self

        # make sure we write based on cube df rather than outdated raster attributes
        self.update_rasters("band_index", "name")

        self = self._delegate_raster_func(_write_func, folder=folder, **kwargs)

        self.write_df(folder)

        return self

    def write_df(self, folder: str):
        gdf: GeoDataFrame = self._prepare_df_for_parquet()

        if is_dapla():
            write_geopandas(gdf, Path(folder) / CUBE_DF_NAME)
        else:
            gdf.to_parquet(Path(folder) / CUBE_DF_NAME)

        return self

    def to_gdf(self, ignore_index: bool = False, concat: bool = True):
        if self._chain is None:
            self.check_for_array()
        gdf_list = []
        for i, raster in enumerate(self._df["raster"]):
            row = self._df.iloc[[i]].drop("raster", axis=1)
            gdf = raster.to_gdf()
            idx = row.index[0]
            gdf.index = np.repeat(idx, len(gdf))
            for col in row.columns:
                gdf[col] = {idx: row[col].iloc[0]}
            gdf_list.append(gdf)
        if not concat:
            return gdf_list
        return GeoDataFrame(
            pd.concat(gdf_list, ignore_index=ignore_index),
            geometry="geometry",
            crs=self.crs,
        )

    def filter_by_location(self, other):
        self._df = self._df[self.boxes.interesects(other)]
        return self

    def intersects(self, other):
        self.check_not_chain()
        return self.boxes.intersects(other)

    def to_crs(self, crs):
        if self._chain is not None:
            self._chain.append_raster_func(_to_crs_func, crs=crs)
            return self
        self.df["raster"] = [_to_crs_func(r, crs=crs) for r in self]
        self._warped_crs = crs
        return self

    def set_crs(self, crs, allow_override: bool = False):
        if self._chain is not None:
            self._chain.append_raster_func(
                _set_crs_func, crs=crs, allow_override=allow_override
            )
            return self
        self.df["raster"] = [
            _set_crs_func(r, crs=crs, allow_override=allow_override) for r in self
        ]
        self._warped_crs = crs
        return self

    def explode(self, ignore_index: bool = False):
        if self._is_chaining():
            self._chain.append_cube_func(explode, ignore_index=ignore_index)
            return self

        return explode(self, ignore_index=ignore_index)

    def clipmerge(
        self,
        mask,
        by: str | list[str] | None = None,
        res=None,
        raster_dtype=None,
        aggfunc="first",
        copy: bool = True,
        **kwargs,
    ):
        return self.merge(
            by=by,
            bounds=mask,
            res=res,
            raster_dtype=raster_dtype,
            aggfunc=aggfunc,
            copy=copy,
            **kwargs,
        )

    def merge(
        self,
        by: str | list[str] | None = None,
        bounds=None,
        res=None,
        aggfunc="first",
        copy: bool = True,
        **kwargs,
    ):
        # self.check_not_array()
        kwargs = {
            "by": by,
            "bounds": bounds,
            "res": res,
            "aggfunc": aggfunc,
        } | kwargs

        if self._is_chaining():
            self._chain.append_cube_func(cube_merge, **kwargs)
            self._read_in_chain = True
            return self

        return cube_merge(self, **kwargs)

    def merge_by_bounds(
        self,
        bounds=None,
        res=None,
        aggfunc="first",
        **kwargs,
    ):
        """Merge rasters with the same bounds to a 3 dimensional array.

        The 'name' column will be updated to the bounds as a string.
        """

        kwargs = {
            "bounds": bounds,
            "res": res,
            "aggfunc": aggfunc,
            **kwargs,
        }
        if self._is_chaining():
            self._chain.append_cube_func(
                merge_by_bounds,
                **kwargs,
            )
            self._read_in_chain = True
            return self

        return merge_by_bounds(
            self,
            # bounds=bounds,
            # res=res,
            # aggfunc=aggfunc,
            **kwargs,
        )

    def dissolve_bands(self, aggfunc):
        if self._chain is None:
            self.check_for_array()
        if not callable(aggfunc) and not isinstance(aggfunc, str):
            raise TypeError("Can only supply a single aggfunc")

        aggfunc = get_numpy_func(aggfunc)

        if self._chain is not None:
            try:
                self.mapper.validate_execution(aggfunc)
            except AttributeError:
                pass
            self._chain.append_array_func(aggfunc, axis=0)
            return self

        self.arrays = [aggfunc(arr, axis=0) for arr in self.arrays]
        return self
        self = self._delegate_array_func(aggfunc, axis=0)
        self._df["band_index"] = self.band_index
        return self

    def dissolve_by_bounds(self, aggfunc):
        if self._chain is None:
            self.check_for_array()
        if not callable(aggfunc) and not isinstance(aggfunc, str):
            raise TypeError("Can only supply a single aggfunc")

        aggfunc = get_numpy_func(aggfunc)

        self._df["tile"] = self.tile.values

        if self._chain is not None:
            try:
                self.mapper.validate_execution(aggfunc)
            except AttributeError:
                pass
            self._chain.append_array_func(aggfunc, axis=0)
            return self
        self.arrays = [aggfunc(arr, axis=0) for arr in self.arrays]
        return self
        self = self._delegate_array_func(aggfunc, axis=0)
        self._df["band_index"] = self.band_index
        return self

    def _grouped_merge(
        self, group: DataFrame, bounds, raster_dtype, res, _as_3d=False, **kwargs
    ):
        # if res is None:
        #   res = group["res"].min()
        if bounds is None:
            bounds = (
                group["minx"].min(),
                group["miny"].min(),
                group["maxx"].max(),
                group["maxy"].max(),
            )
        exploded = group.explode(column="band_index")
        band_index = tuple(exploded["band_index"].sort_values().unique())
        arrays = []
        for idx in band_index:
            paths = exploded.loc[exploded["band_index"] == idx, "path"]
            array, transform = merge.merge(
                list(paths), indexes=(idx,), bounds=bounds, res=res, **kwargs
            )
            # merge doesn't allow single index (numpy error), so changing afterwards
            if len(array.shape) == 3:
                assert array.shape[0] == 1
                array = array[0]
            arrays.append(array)
        array = np.array(arrays)
        assert len(array.shape) == 3
        if not _as_3d and array.shape[0] == 1:
            array = array[0]

        return raster_dtype.from_array(
            array,
            transform=transform,
            crs=self.crs,
        )

    def _merge_all(self, bounds, res, raster_dtype, **kwargs):
        raster = self._grouped_merge(
            self.df,
            bounds=bounds,
            res=res,
            raster_dtype=raster_dtype,
            **kwargs,
        )
        cube = GeoDataCube([raster])
        cube = self._add_attributes_from_self(cube)
        return cube

    def min(self):
        self.check_not_chain()
        return min(x.min() for x in self)

    def max(self):
        self.check_not_chain()
        return max(x.max() for x in self)

    def _add_attributes_from_self(self, obj):
        for key, value in self.__dict__.items():
            if key == "_df":
                continue
            obj[key] = value
        return obj

    def raster_attribute(self, attribute: str) -> Series:
        """Get a Raster attribute returned as values of a Series."""
        return Series(
            [getattr(r, attribute) for r in self],
            index=self._df.index,
            name=attribute,
        )

    def raster_method(self, method: str, **kwargs):
        """Run Raster methods."""
        if not all(hasattr(r, method) for r in self):
            raise AttributeError(f"Raster has no method {method!r}.")

        method_as_func = functools.partial(_method_as_func, method=method, **kwargs)

        if self._chain is not None:
            self._chain.append_raster_func(method_as_func)
            return self

        self.df["raster"] = [method_as_func(r) for r in self]
        return self

    @staticmethod
    def validate_mapper(mapper):
        if not isinstance(mapper, MultiProcessingMapper):
            raise TypeError

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, new_df):
        self.validate_cube_df(new_df)
        self._df = new_df
        return self._df

    @property
    def raster_dtype(self):
        return Series(
            [r.__class__ for r in self],
            index=self._df.index,
            name="raster_dtype",
        )

    def most_common_dtype(self):
        return list(self.raster_dtype.value_counts().index)[0]

    @property
    def arrays(self) -> list[np.ndarray]:
        return self.raster_attribute("array")

    @arrays.setter
    def arrays(self, new_arrays: list[np.ndarray]):
        if len(new_arrays) != len(self._df):
            arr, df = len(new_arrays), len(self._df)
            raise ValueError(
                f"Number of arrays ({arr}) must be same as length as df ({df})."
            )
        if not all(isinstance(arr, np.ndarray) for arr in new_arrays):
            raise ValueError("Must be list of numpy ndarrays")

        if self.df.index.is_unique:
            self.df["raster"] = {
                i: raster.update(array=arr)
                for (i, raster), arr in zip(self._df["raster"].items(), new_arrays)
            }
        self._df["__i"] = range(len(self._df))
        mapper = {
            i: raster.update(array=arr)
            for i, raster, arr in zip(self._df["__i"], self._df["raster"], new_arrays)
        }
        self._df["raster"] = self._df["__i"].map(mapper)
        self._df = self._df.drop("__i", axis=1)

    @property
    def name(self) -> Series:
        return self.raster_attribute("name")

    @property
    def date(self) -> Series:
        return self.raster_attribute("date")

    @property
    def subfolder(self) -> Series:
        return self.raster_attribute("subfolder")

    @property
    def band_index(self) -> Series:
        return self.raster_attribute("band_index")

    @property
    def band_name(self) -> Series:
        return self.raster_attribute("band_name")

    @property
    def area(self) -> Series:
        return self.raster_attribute("area")

    @property
    def length(self) -> Series:
        return self.raster_attribute("length")

    @property
    def height(self) -> Series:
        return self.raster_attribute("height")

    @property
    def width(self) -> Series:
        return self.raster_attribute("width")

    @property
    def shape(self) -> Series:
        return self.raster_attribute("shape")

    @property
    def count(self) -> Series:
        return self.raster_attribute("count")

    @property
    def res(self) -> Series:
        return self.raster_attribute("res")

    @property
    def crs(self) -> pyproj.CRS:
        return self._warped_crs if hasattr(self, "_warped_crs") else self._crs

    @property
    def unary_union(self) -> Geometry:
        return shapely.unary_union([shapely.box(*r.bounds) for r in self])

    @property
    def centroid(self) -> Series:
        return self.raster_attribute("centroid")

    @property
    def tile(self) -> Series:
        return self.raster_attribute("tile")

    @property
    def is_tiled(self) -> bool:
        return len(set(self.bounds)) > 1

    @property
    def bounds(self) -> Series:
        return DataFrame(
            [r.bounds for r in self],
            index=self.df.index,
            columns=["minx", "miny", "maxx", "maxy"],
        )

    @property
    def minx(self) -> Series:
        return Series(
            [r.bounds[0] for r in self],
            index=self._df.index,
            name="minx",
        )

    @property
    def miny(self) -> Series:
        return Series(
            [r.bounds[1] for r in self],
            index=self._df.index,
            name="miny",
        )

    @property
    def maxx(self) -> Series:
        return Series(
            [r.bounds[2] for r in self],
            index=self._df.index,
            name="maxx",
        )

    @property
    def maxy(self) -> Series:
        return Series(
            [r.bounds[3] for r in self],
            index=self._df.index,
            name="maxy",
        )

    @property
    def boxes(self) -> GeoSeries:
        """GeoSeries of each raster's bounds as polygon."""
        return GeoSeries(
            [shapely.box(*r.bounds) for r in self],
            index=self._df.index,
            name="boxes",
            crs=self.crs,
        )

    @property
    def total_bounds(self) -> tuple[float, float, float, float]:
        bounds = self.bounds
        minx = bounds["minx"].min()
        miny = bounds["miny"].min()
        maxx = bounds["maxx"].max()
        maxy = bounds["maxy"].max()
        return minx, miny, maxx, maxy

    def _push_raster_col(self):
        col = self._df["raster"]
        self._df = self["df"].reindex(
            columns=[c for c in self["df"].columns if c != col] + [col]
        )
        return self

    def copy(self, deep=True):
        """Returns a (deep) copy of the class instance and its rasters.

        Args:
            deep: Whether to return a deep or shallow copy. Defaults to True.
        """
        copied = deepcopy(self) if deep else copy(self)

        df = copied.df.copy(deep=deep)

        df["__i"] = range(len(df))
        for i, raster in zip(df["__i"], df["raster"]):
            df.loc[df["__i"] == i, "raster"] = raster.copy(deep=deep)

        copied._df = df.drop("__i", axis=1)

        return copied

    @classmethod
    def validate_cube_df(cls, df):
        if type(df) not in (DataFrame, Series):
            raise TypeError

        for col in cls.BASE_CUBE_COLS:
            if col not in df:
                raise ValueError(f"Column {col!r} cannot be removed from df.")

        for col in ["raster", "band_index"]:
            if df[col].isna().any():
                raise ValueError(f"Column {col!r} cannot have missing values.")

    @classmethod
    def get_cube_template(cls) -> DataFrame:
        return pd.DataFrame(columns=cls.BASE_CUBE_COLS)

    def make_cube_df(self, rasters: Iterable[Raster], df=None, root=None) -> DataFrame:
        if not all(isinstance(r, Raster) for r in rasters):
            raise TypeError("rasters should be an iterable of Rasters.")

        if df is None:
            df = pd.DataFrame()

        def get_subfolder(path, root):
            if path is None:
                return None
            subfolder = str(Path(path).parent)
            if root is not None:
                subfolder = subfolder.replace(str(Path(root)), "")
            return subfolder or None

        df["name"] = [r.name for r in rasters]
        df["path"] = [r.path for r in rasters]
        df["subfolder"] = [get_subfolder(r.path, root) for r in rasters]
        df["band_index"] = [r.band_index for r in rasters]
        df["band_name"] = [r.band_name for r in rasters]
        df["raster"] = list(rasters)

        return df.replace({None: pd.NA})

    def _is_chaining(self):
        return self._chain is not None and not self._executing

    def _prepare_df_for_parquet(self):
        """Remove column raster and add geometry column box."""
        if not all(col in self.df for col in self.BASE_CUBE_COLS):
            raise ValueError(f"Must have all columns {', '.join(self.BASE_CUBE_COLS)}")
        df = self._df.drop(columns=["raster"])
        df[self.CUBE_GEOM_COL] = self.boxes

        return GeoDataFrame(df, geometry=self.CUBE_GEOM_COL, crs=self.crs)

    @classmethod
    def _prepare_gdf_for_raster(cls, gdf: GeoDataFrame) -> tuple[DataFrame, DataFrame]:
        must_have = [col for col in cls.BASE_CUBE_COLS if col != "raster"]
        if not isinstance(gdf, GeoDataFrame):
            raise TypeError("'df' must be GeoDataFrame with image bounds as geometry.")
        if not all(col in gdf for col in must_have):
            raise ValueError(f"Must have all columns {', '.join(must_have)}")
        if gdf._geometry_column_name != cls.CUBE_GEOM_COL:
            raise AttributeError(f"Must have geometry column {cls.CUBE_GEOM_COL!r}")

        gdf = gdf.copy()
        gdf["bounds"] = [geom.bounds for geom in gdf[cls.CUBE_GEOM_COL]]
        gdf["crs"] = gdf.crs
        raster_cols = [col for col in gdf if col in cls.ALLOWED_KEYS]
        other_cols = [
            col for col in gdf if col not in cls.ALLOWED_KEYS + [cls.CUBE_GEOM_COL]
        ]
        return gdf[raster_cols], gdf[other_cols]

    @classmethod
    def get_raster_dict(cls, df):
        return df[[key for key in cls.ALLOWED_KEYS if key in df]].to_dict()

    """def _delegate_raster_func(self, func, **kwargs):
        self._df["raster"] = [func(r, **kwargs) for r in self]
        return self"""

    def _delegate_raster_func(self, func, **kwargs):
        return [func(r, **kwargs) for r in self]

    def _delegate_array_func(self, func, **kwargs):
        self.arrays = [func(arr, **kwargs) for arr in self.arrays]
        return self

    def check_for_array(self, text=""):
        mess = "Arrays are not loaded. " + text
        if self.arrays.isna().all():
            raise ValueError(mess)

    def check_not_array(self):
        mess = super().check_not_array_mess()
        if self.arrays.notna().any():
            raise ValueError(mess)
        if self._read_in_chain:
            raise ValueError(mess)

    def check_not_chain(self):
        if self._chain is not None:
            raise ValueError("Cannot use this method in a chain.")

    def __hash__(self):
        return hash(self._hash)

    def __iter__(self):
        return iter(self._df["raster"])

    def __len__(self):
        return len(self._df)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(df=" "\n" f"{self._df.__repr__()}" "\n)"

    def __setattr__(self, __name: str, __value) -> None:
        return super().__setattr__(__name, __value)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        if not isinstance(key, str):
            return self._df["raster"][key]
        return getattr(self, key)

    def __mul__(self, scalar):
        if self._chain is not None:
            self._chain.append_array_func(_mul, scalar=scalar)
            return self
        self.check_for_array()
        self.arrays = [_mul(arr, scalar=scalar) for arr in self.arrays]
        return self

    def __add__(self, scalar):
        if self._chain is not None:
            self._chain.append_array_func(_add, scalar=scalar)
            return self

        self.check_for_array()
        self.arrays = [_add(arr, scalar=scalar) for arr in self.arrays]
        return self
        return self._delegate_array_func(_add, scalar=scalar)

    def __sub__(self, scalar):
        if self._chain is not None:
            self._chain.append_array_func(_sub, scalar=scalar)
            return self
        self.check_for_array()
        self.arrays = [_sub(arr, scalar=scalar) for arr in self.arrays]
        return self
        return self._delegate_array_func(_sub, scalar=scalar)

    def __truediv__(self, scalar):
        if self._chain is not None:
            self._chain.append_array_func(_truediv, scalar=scalar)
            return self
        self.check_for_array()
        self.arrays = [_truediv(arr, scalar=scalar) for arr in self.arrays]
        return self
        return self._delegate_array_func(_truediv, scalar=scalar)

    def __floordiv__(self, scalar):
        if self._chain is not None:
            self._chain.append_array_func(_floordiv, scalar=scalar)
            return self
        self.check_for_array()
        self.arrays = [_floordiv(arr, scalar=scalar) for arr in self.arrays]
        return self
        return self._delegate_array_func(_floordiv, scalar=scalar)

    def __pow__(self, scalar):
        if self._chain is not None:
            self._chain.append_array_func(_pow, scalar=scalar)
            return self
        self.check_for_array()
        self.arrays = [_pow(arr, scalar=scalar) for arr in self.arrays]
        return self
        return self._delegate_array_func(_pow, scalar=scalar)


class RandomCubeSample:
    def __init__(self, cube, masks: list[Geometry], crop=True, **kwargs):
        if not isinstance(cube, GeoDataCube):
            raise TypeError

        cube.copy().clip(box)
        return RandomCubeSample(cube, self.clip(boxes, crop=crop, **kwargs))

    @property
    def index(self):
        pass

    @property
    def arrays(self):
        pass

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)


"""
b4 = cube.query("band_name == 'B4'")
b8 = cube.query("band_name == 'B8'")


def bndvi_func(b4, b8):
    return (b8 - b4) / (b8 + b4)


cube_math(b4, b8, func=bndvi_func)
"""


def cube_math(self, other: GeoDataCube, func: Callable, copy=True):
    if copy:
        self = self.copy()
        other = other.copy()

    self._df["tile"] = self.tile.values
    other._df["tile"] = other.tile.values

    common_tiles = [
        tile for tile in self.df["tile"].unique() if tile in other.df["tile"].unique()
    ]

    if not len(common_tiles):
        raise ValueError("No common tiles.")

    cube_pairs = []
    for tile in common_tiles:
        self_tile = self.query("tile == @tile")
        other_tile = self.query("tile == @tile")

        if len(self_tile) != 1 or len(other_tile) != 1:
            raise ValueError("The cubes can have only one raster per tile.")

        if self_tile.shape.iloc[0] != other_tile.shape.iloc[0]:
            raise ValueError("Rasters must have same shape.")

        pair = self_tile, other_tile
        cube_pairs.append(pair)

    if cube._chain is not None:
        cube._chain.append_raster_iter(func, iterable=cube_pairs)
        return cube

    cubes = [func(*items) for items in cube_pairs]
    return concat_cubes(cubes, ignore_index=True)


def cube_math(cube1: GeoDataCube, cube2: GeoDataCube, func: Callable, copy=True):
    if copy:
        cube1 = cube1.copy()
        cube2 = cube2.copy()

    cube1._df["tile"] = cube1.tile.values
    cube2._df["tile"] = cube2.tile.values

    common_tiles = [
        tile for tile in cube1.df["tile"].unique() if tile in cube2.df["tile"].unique()
    ]

    if not len(common_tiles):
        raise ValueError("No common tiles.")

    cube_pairs = []
    for tile in common_tiles:
        cube1_tile = cube1.query("tile == @tile")
        cube2_tile = cube1.query("tile == @tile")

        if len(cube1_tile) != 1 or len(cube2_tile) != 1:
            raise ValueError("The cubes can have only one raster per tile.")

        if cube1_tile.shape.iloc[0] != cube2_tile.shape.iloc[0]:
            raise ValueError("Rasters must have same shape.")

        pair = cube1_tile, cube2_tile
        cube_pairs.append(pair)

    if cube._chain is not None:
        cube._chain.append_raster_iter(func, iterable=cube_pairs)
        return cube

    cubes = [func(*items) for items in cube_pairs]
    return concat_cubes(cubes, ignore_index=True)


def concat_cubes(
    cube_list: list[GeoDataCube],
    ignore_index: bool = False,
):
    if not all(isinstance(cube, GeoDataCube) for cube in cube_list):
        raise TypeError

    cube = GeoDataCube()
    cube._crs = get_common_crs(cube_list)

    cube.df = pd.concat([cube.df for cube in cube_list], ignore_index=ignore_index)

    return cube


def starmap_concat(pool, func, iterable, **kwargs):
    partial_func = functools.partial(func, **kwargs)
    cubes = pool.starmap(partial_func, iterable)
    return concat_cubes(cubes, ignore_index=True)


"""Method-to-function to use as mapping function."""


def _cube_merge(cubebounds, **kwargs):
    assert isinstance(cubebounds, dict)
    return cube_merge(cube=cubebounds["cube"], bounds=cubebounds["bounds"], **kwargs)


def _method_as_func(self, method, **kwargs):
    return getattr(self, method)(**kwargs)


def _astype_raster(raster, dtype):
    """Returns raster as another raster type."""
    return dtype(raster)


def _from_gdf_func(gdf, raster_dtype, **kwargs):
    return raster_dtype.from_gdf(gdf, **kwargs)


def _write_func(raster, folder, **kwargs):
    path = str(Path(folder) / Path(raster.name).stem) + ".tif"
    raster.write(path, **kwargs)
    raster.path = path
    return raster


def _clip_func(raster, mask, **kwargs):
    return raster.clip(mask, **kwargs)


def _clip_func(raster, mask, **kwargs):
    return raster.clip(mask, **kwargs)


def _load_func(raster, **kwargs):
    return raster.load(**kwargs)


def _zonal_func(raster, **kwargs):
    clipmerge()
    return raster.zonal(**kwargs)


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
