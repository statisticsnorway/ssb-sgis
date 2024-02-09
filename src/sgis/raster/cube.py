import functools
import itertools
import multiprocessing
import os
import re
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
import shapely
import xarray as xr
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame, Series
from pandas.api.types import is_dict_like, is_list_like
from rasterio import merge as rasterio_merge
from rioxarray.merge import merge_arrays
from rtree.index import Index, Property
from shapely import Geometry
from typing_extensions import Self  # TODO: imperter fra typing når python 3.11

from ..geopandas_tools.bounds import get_total_bounds, make_grid
from ..geopandas_tools.conversion import (
    crs_to_string,
    is_bbox_like,
    to_bbox,
    to_shapely,
)
from ..geopandas_tools.general import get_common_crs
from ..geopandas_tools.overlay import clean_overlay
from ..helpers import dict_zip_intersection, get_all_files, get_numpy_func
from ..io._is_dapla import is_dapla
from ..io.opener import opener
from ..parallel.parallel import Parallel
from .raster import Raster


try:
    from torchgeo.datasets.geo import RasterDataset
    from torchgeo.datasets.utils import BoundingBox
except ImportError:

    class BoundingBox:
        """Placeholder."""

        def __init__(self, *args, **kwargs):
            raise ImportError("missing optional dependency 'torchgeo'")

    class RasterDataset:
        """Placeholder."""

        def __init__(self, *args, **kwargs):
            raise ImportError("missing optional dependency 'torchgeo'")


try:
    import torch
    from torchgeo.datasets.utils import disambiguate_timestamp
except ImportError:

    class torch:
        """Placeholder."""

        class Tensor:
            pass


try:
    from ..io.dapla_functions import read_geopandas
except ImportError:
    pass

try:
    from dapla import FileClient, write_pandas
except ImportError:
    pass

from .bands import Sentinel2
from .base import ALLOWED_KEYS, NESSECARY_META, get_index_mapper
from .cubebase import _from_gdf_func, _method_as_func, _raster_from_path, _write_func
from .indices import get_raster_pairs, index_calc_pair
from .zonal import make_geometry_iterrows, prepare_zonal, zonal_func, zonal_post


class DataCube:
    """Experimental.

    Examples
    --------

    >>> cube = sg.DataCube.from_root(...)
    >>> clipped = cube.clip(mask).merge(by="date")
    >>>
    """

    CUBE_DF_NAME = "cube_df.parquet"

    CANON_RASTER_TYPES = {
        "Raster": Raster,
        "Sentinel2": Sentinel2,
    }

    separate_files = True
    is_image = True

    def __init__(
        self,
        data: Iterable[Raster] | None = None,
        crs: Any | None = None,
        res: int | None = None,
        nodata: int | None = None,
        copy: bool = False,
        parallelizer: Optional[Parallel] = None,
    ) -> None:
        self._arrays = None
        self._res = res
        self.parallelizer = parallelizer

        # hasattr check to allow class attribute
        if not hasattr(self, "_nodata"):
            self._nodata = nodata

        if isinstance(data, DataCube):
            for key, value in data.__dict__.items():
                setattr(self, key, value)
            return
        elif data is None:
            self.data = data
            self._crs = None
            return
        elif not is_list_like(data) and all(isinstance(r, Raster) for r in data):
            raise TypeError(
                "'data' must be a Raster instance or an iterable."
                f"Got {type(data)}: {data}"
            )
        else:
            data = list(data)

        if copy:
            data = [raster.copy() for raster in data]
        else:
            # take a copy only if there are gdfs with the same memory address
            if sum(r1 is r2 for r1 in data for r2 in data) < len(data):
                data = [raster.copy() for raster in data]

        self.data = data

        nodatas = {r.nodata for r in self}
        if self.nodata is None and len(nodatas) > 1:
            raise ValueError(
                "Must specify 'nodata' when the images have different nodata values. "
                f"Got {', '.join([str(x) for x in nodatas])}"
            )

        resolutions = {r.res for r in self}
        if self._res is None and len(resolutions) > 1:
            raise ValueError(
                "Must specify 'res' when the images have different resolutions. "
                f"Got {', '.join([str(x) for x in resolutions])}"
            )
        elif res is None and len(resolutions):
            self._res = resolutions.pop()

        if crs:
            self._crs = pyproj.CRS(crs)
            if not all(self._crs.equals(pyproj.CRS(r.crs)) for r in self.data):
                self = self.to_crs(self._crs)
        try:
            self._crs = get_common_crs(self.data)
        except (ValueError, IndexError):
            self._crs = None

    @classmethod
    def from_root(
        cls,
        root: str | Path,
        *,
        res: int | None = None,
        raster_type: Raster = Raster,
        check_for_df: bool = True,
        contains: str | None = None,
        endswith: str = ".tif",
        regex: str | None = None,
        parallelizer: Optional[Parallel] = None,
        file_system=None,
        **kwargs,
    ):
        kwargs = {
            "raster_type": raster_type,
            "res": res,
        } | kwargs

        if is_dapla():
            if file_system is None:
                file_system = FileClient.get_gcs_file_system()
            glob_pattern = str(Path(root) / "**")
            paths: list[str] = file_system.glob(glob_pattern)
            if contains:
                paths = [path for path in paths if contains in path]

        else:
            paths = get_all_files(root)

        dfs = [path for path in paths if path.endswith(cls.CUBE_DF_NAME)]

        if contains:
            paths = [path for path in paths if contains in path]
        if endswith:
            paths = [path for path in paths if path.endswith(endswith)]
        if regex:
            regex = re.compile(regex)
            paths = [path for path in paths if re.search(regex, path)]
        if raster_type.filename_regex is not None:
            # regex = raster_type.filename_regex
            # paths = [path for path in paths if re.search(regex, Path(path).name)]
            regex = re.compile(raster_type.filename_regex, re.VERBOSE)
            paths = [
                path
                for path in paths
                if re.match(regex, os.path.basename(path))
                or re.search(regex, os.path.basename(path))
            ]

        if not check_for_df or not len(dfs):
            return cls.from_paths(
                paths,
                # indexes=indexes,
                parallelizer=parallelizer,
                **kwargs,
            )

        folders_with_df: set[Path] = {Path(path).parent for path in dfs if path}

        cubes: list[DataCube] = [cls.from_cube_df(df, **kwargs) for df in dfs]

        paths_in_folders_without_df = [
            path for path in paths if Path(path).parent not in folders_with_df
        ]

        if paths_in_folders_without_df:
            cubes += [
                cls.from_paths(
                    paths_in_folders_without_df,
                    parallelizer=parallelizer,
                    **kwargs,
                )
            ]

        return concat_cubes(cubes, res=res)

    @classmethod
    def from_paths(
        cls,
        paths: Iterable[str | Path],
        *,
        res: int | None = None,
        raster_type: Raster = Raster,
        parallelizer: Optional[Parallel] = None,
        file_system=None,
        **kwargs,
    ):
        crs = kwargs.pop("crs", None)

        if not paths:
            return cls(crs=crs, parallelizer=parallelizer, res=res)

        kwargs = dict(raster_type=raster_type, res=res) | kwargs

        if file_system is None and is_dapla():
            kwargs |= {"file_system": FileClient.get_gcs_file_system()}

        if parallelizer is None:
            rasters: list[Raster] = [
                _raster_from_path(path, **kwargs) for path in paths
            ]
        else:
            rasters: list[Raster] = parallelizer.map(
                _raster_from_path,
                paths,
                kwargs=kwargs,
            )

        return cls(rasters, copy=False, crs=crs, res=res)

    @classmethod
    def from_gdf(
        cls,
        gdf: GeoDataFrame | Iterable[GeoDataFrame],
        columns: str | Iterable[str],
        res: int,
        parallelizer: Optional[Parallel] = None,
        tile_size: int | None = None,
        grid: GeoSeries | None = None,
        raster_type: Raster = Raster,
        **kwargs,
    ):
        """

        Args:
            grid: A grid.
            **kwargs: Keyword arguments passed to Raster.from_gdf.
        """
        if grid is None and tile_size is None:
            raise ValueError("Must specify either 'tile_size' or 'grid'.")

        if isinstance(gdf, GeoDataFrame):
            gdf = [gdf]
        elif not all(isinstance(frame, GeoDataFrame) for frame in gdf):
            raise TypeError("gdf must be one or more GeoDataFrames.")

        if grid is None:
            crs = get_common_crs(gdf)
            total_bounds = shapely.unary_union(
                [shapely.box(*frame.total_bounds) for frame in gdf]
            )
            grid = make_grid(total_bounds, gridsize=tile_size, crs=crs)

        grid["tile_idx"] = range(len(grid))

        partial_func = functools.partial(
            _from_gdf_func,
            columns=columns,
            res=res,
            raster_type=raster_type,
            **kwargs,
        )

        def to_gdf_list(gdf: GeoDataFrame) -> list[GeoDataFrame]:
            return [gdf.loc[gdf["tile_idx"] == i] for i in gdf["tile_idx"].unique()]

        rasters = []

        if processes > 1:
            rasters = parallelizer.map(
                clean_overlay, gdf, args=(grid,), kwargs=dict(keep_geom_type=True)
            )
            with multiprocessing.get_context("spawn").Pool(processes) as p:
                for frame in gdf:
                    frame = frame.overlay(grid, keep_geom_type=True)
                    gdfs = to_gdf_list(frame)
                    rasters += p.map(partial_func, gdfs)
        elif processes < 1:
            raise ValueError("processes must be an integer 1 or greater.")
        else:
            for frame in gdf:
                frame = frame.overlay(grid, keep_geom_type=True)
                gdfs = to_gdf_list(frame)
                rasters += [partial_func(gdf) for gdf in gdfs]

        return cls(rasters, res=res)

    @classmethod
    def from_cube_df(cls, df: DataFrame | str | Path, res: int | None = None):
        if isinstance(df, (str, Path)):
            df = read_geopandas(df) if is_dapla() else gpd.read_parquet(df)

        # recursive
        if not is_dict_like(df) and all(
            isinstance(x, (str, Path, DataFrame)) for x in df
        ):
            cubes = [cls.from_cube_df(x) for x in df]
            cube = concat_cubes(cubes, res=res)
            return cube

        if isinstance(df, dict):
            df = DataFrame(df)
        elif not isinstance(df, DataFrame):
            raise TypeError("df must be DataFrame or file path to a parquet file.")

        try:
            raster_types = [cls.CANON_RASTER_TYPES[x] for x in df["type"]]
        except KeyError:
            for x in df["type"]:
                try:
                    cls.CANON_RASTER_TYPES[x]
                except KeyError:
                    raise ValueError(
                        f"Cannot convert raster type '{x}' to a Raster instance."
                    )

        rasters: list[Raster] = [
            raster_type.from_dict(meta)
            for raster_type, (_, meta) in zip(
                raster_types, df[NESSECARY_META].iterrows()
            )
        ]
        return cls(rasters)

    def to_gdf(
        self, column: str | None = None, ignore_index: bool = False, concat: bool = True
    ) -> GeoDataFrame:
        gdfs = self.run_raster_method("to_gdf", column=column, return_self=False)

        if concat:
            return pd.concat(gdfs, ignore_index=ignore_index)
        return gdfs

    def to_xarray(self) -> xr.Dataset:
        return xr.Dataset({i: r.to_xarray() for i, r in enumerate(self.data)})

    def zonal(
        self,
        polygons: GeoDataFrame,
        aggfunc: str | Callable | list[Callable | str],
        array_func: Callable | None = None,
        by_date: bool | None = None,
        dropna: bool = True,
    ) -> GeoDataFrame:
        idx_mapper, idx_name = get_index_mapper(polygons)
        polygons, aggfunc, func_names = prepare_zonal(polygons, aggfunc)
        poly_iter = make_geometry_iterrows(polygons)

        if by_date is None:
            by_date: bool = all(r.date is not None for r in self)

        if not self.parallelizer:
            aggregated: list[DataFrame] = [
                zonal_func(
                    poly,
                    cube=self,
                    array_func=array_func,
                    aggfunc=aggfunc,
                    func_names=func_names,
                    by_date=by_date,
                )
                for poly in poly_iter
            ]
        else:
            aggregated: list[DataFrame] = self.parallelizer.map(
                zonal_func,
                poly_iter,
                kwargs=dict(
                    cube=self,
                    array_func=array_func,
                    aggfunc=aggfunc,
                    func_names=func_names,
                    by_date=by_date,
                ),
            )

        return zonal_post(
            aggregated,
            polygons=polygons,
            idx_mapper=idx_mapper,
            idx_name=idx_name,
            dropna=dropna,
        )

    def gradient(self, degrees: bool = False) -> Self:
        self.data = self.run_raster_method("gradient", degrees=degrees)
        return self

    def map(self, func: Callable, return_self: bool = True, **kwargs) -> Self:
        """Maps each raster array to a function.

        The function must take a numpy array as first positional argument,
        and return a single numpy array. The function should be defined in
        the leftmost indentation level. If in Jupyter, the function also
        have to be defined in and imported from another file.
        """
        self._check_for_array()
        if self.parallelizer:
            data = self.parallelizer.map(func, self.arrays, kwargs=kwargs)
        else:
            data = [func(arr, **kwargs) for arr in self.arrays]
        if not return_self:
            return data
        self.arrays = data
        return self

    def raster_map(self, func: Callable, return_self: bool = True, **kwargs) -> Self:
        """Maps each raster to a function.

        The function must take a Raster object as first positional argument,
        and return a single Raster object. The function should be defined in
        the leftmost indentation level. If in Jupyter, the function also
        have to be defined in and imported from another file.
        """
        if self.parallelizer:
            data = self.parallelizer.map(func, self, kwargs=kwargs)
        else:
            data = [func(r, **kwargs) for r in self]
        if not return_self:
            return data
        self.data = data
        return self

    def load(self, copy: bool = True, **kwargs) -> Self:
        if self.crs is None:
            self._crs = get_common_crs(self.data)

        cube = self.copy() if copy else self

        cube.data = cube.run_raster_method("load", **kwargs)

        return cube

    def intersects(self, other, copy: bool = True) -> Self:
        other = to_shapely(other)
        cube = self.copy() if copy else self
        cube = cube[cube.boxes.intersects(other)]
        return cube

    def sfilter(self, other, copy: bool = True) -> Self:
        other = to_shapely(other)
        cube = self.copy() if copy else self
        cube.data = [raster for raster in self if raster.unary_union.intersects(other)]
        return cube

    def clip(
        self, mask: GeoDataFrame | GeoSeries | Geometry, copy: bool = True, **kwargs
    ) -> Self:
        if self.crs is None:
            self._crs = get_common_crs(self.data)

        if (
            hasattr(mask, "crs")
            and mask.crs
            and not pyproj.CRS(self.crs).equals(pyproj.CRS(mask.crs))
        ):
            raise ValueError("crs mismatch.")

        cube = self.copy() if copy else self

        cube = cube.sfilter(to_shapely(mask), copy=False)

        cube.data = cube.run_raster_method("clip", mask=mask, **kwargs)
        return cube

    def clipmerge(self, mask, **kwargs) -> Self:
        return clipmerge(self, mask, **kwargs)

    def merge_by_bounds(self, by: str | list[str] | None = None, **kwargs) -> Self:
        return merge_by_bounds(self, by=by, **kwargs)

    def merge(self, by: str | list[str] | None = None, **kwargs) -> Self:
        return merge(self, by=by, **kwargs)

    def explode(self) -> Self:
        def explode_one_raster(raster: Raster) -> list[Raster]:
            property_values = {key: getattr(raster, key) for key in raster.properties}

            all_meta = {
                key: value
                for key, value in (
                    raster.__dict__ | raster.meta | property_values
                ).items()
                if key in ALLOWED_KEYS and key not in ["array", "indexes"]
            }
            if raster.array is None:
                return [
                    raster.__class__.from_dict({"indexes": i} | all_meta)
                    for i in raster.indexes_as_tuple()
                ]
            else:
                return [
                    raster.__class__.from_dict(
                        {"array": array, "indexes": i + 1} | all_meta
                    )
                    for i, array in enumerate(raster.array_list())
                ]

        self.data = list(
            itertools.chain.from_iterable(
                [explode_one_raster(raster) for raster in self]
            )
        )
        return self

    def dissolve_bands(self, aggfunc, copy: bool = True) -> Self:
        self._check_for_array()
        if not callable(aggfunc) and not isinstance(aggfunc, str):
            raise TypeError("Can only supply a single aggfunc")

        cube = self.copy() if copy else self

        aggfunc = get_numpy_func(aggfunc)

        cube = cube.map(aggfunc, axis=0)
        return cube

    def write(
        self,
        root: str,
        file_format: str = "tif",
        **kwargs,
    ) -> None:
        """Writes arrays as tif files and df with file info.

        This method should be run after the rasters have been clipped, merged or
        its array values have been recalculated.

        Args:

        """
        self._check_for_array()

        if any(raster.name is None for raster in self):
            raise ValueError("")

        paths = [
            (Path(root) / raster.name).with_suffix(f".{file_format}") for raster in self
        ]

        if self.parallelizer:
            self.parallelizer.starmap(_write_func, zip(self, paths), kwargs=kwargs)
        else:
            [_write_func(raster, path, **kwargs) for raster, path in zip(self, paths)]

    def write_df(self, folder: str) -> None:
        df = pd.DataFrame(self.meta)

        folder = Path(folder)
        if not folder.is_dir():
            raise ValueError()

        if is_dapla():
            write_pandas(df, folder / self.CUBE_DF_NAME)
        else:
            df.to_parquet(folder / self.CUBE_DF_NAME)

    def calculate_index(
        self,
        index_func: Callable,
        band_name1: str,
        band_name2: str,
        copy=True,
        **kwargs,
    ) -> Self:
        cube = self.copy() if copy else self

        raster_pairs: list[tuple[Raster, Raster]] = get_raster_pairs(
            cube, band_name1=band_name1, band_name2=band_name2
        )

        kwargs = dict(index_formula=index_func) | kwargs

        if self.parallelizer:
            rasters = self.parallelizer.map(
                index_calc_pair, raster_pairs, kwargs=kwargs
            )
        else:
            rasters = [index_calc_pair(items, **kwargs) for items in raster_pairs]

        return cube.__class__(rasters)

    def reproject_match(self) -> Self:
        pass

    def to_crs(self, crs, copy: bool = True) -> Self:
        cube = self.copy() if copy else self
        cube.data = [r.to_crs(crs) for r in cube]
        cube._warped_crs = crs
        return cube

    def set_crs(self, crs, allow_override: bool = False, copy: bool = True) -> Self:
        cube = self.copy() if copy else self
        cube.data = [r.set_crs(crs, allow_override=allow_override) for r in cube]
        cube._warped_crs = crs
        return cube

    def min(self) -> Series:
        return Series(
            self.run_raster_method("min"),
            name="min",
        )

    def max(self) -> Series:
        return Series(
            self.run_raster_method("max"),
            name="max",
        )

    def raster_attribute(self, attribute: str) -> Series | GeoSeries:
        """Get a Raster attribute returned as values in a pandas.Series."""
        data = [getattr(r, attribute) for r in self]
        if any(isinstance(x, Geometry) for x in data):
            return GeoSeries(data, name=attribute)
        return Series(data, name=attribute)

    def run_raster_method(
        self, method: str, *args, copy: bool = True, return_self=False, **kwargs
    ) -> Self:
        """Run a Raster method for each raster in the cube."""
        if not all(hasattr(r, method) for r in self):
            raise AttributeError(f"Raster has no method {method!r}.")

        method_as_func = functools.partial(
            _method_as_func, method=method, *args, **kwargs
        )

        cube = self.copy() if copy else self

        return cube.raster_map(method_as_func, return_self=return_self)

    @property
    def meta(self) -> list[dict]:
        return [raster.meta for raster in self]

    # @property
    # def cube_df_meta(self) -> dict[list]:
    #     return {
    #         "path": [r.path for r in self],
    #         "indexes": [r.indexes for r in self],
    #         "type": [r.__class__.__name__ for r in self],
    #         "bounds": [r.bounds for r in self],
    #         "crs": [crs_to_string(r.crs) for r in self],
    #     }

    @property
    def data(self) -> list[Raster]:
        return self._data

    @data.setter
    def data(self, data: list[Raster]):
        self.index = Index(interleaved=False, properties=Property(dimension=3))

        if data is None or not len(data):
            self._data = []
            return
        if not all(isinstance(x, Raster) for x in data):
            types = {type(x).__name__ for x in data}
            raise TypeError(f"data must be Raster. Got {', '.join(types)}")
        self._data = list(data)

        for i, raster in enumerate(self._data):
            if raster.date and raster.date_format:
                try:
                    mint, maxt = disambiguate_timestamp(raster.date, raster.date_format)
                except NameError:
                    mint, maxt = 0, 1
            else:
                mint, maxt = 0, 1
            # important: torchgeo has a different order of the bbox than shapely and geopandas
            minx, miny, maxx, maxy = raster.bounds
            self.index.insert(i, (minx, maxx, miny, maxy, mint, maxt))

    @property
    def arrays(self) -> list[np.ndarray]:
        return [raster.array for raster in self]

    @arrays.setter
    def arrays(self, new_arrays: list[np.ndarray]):
        if len(new_arrays) != len(self):
            raise ValueError(
                f"Number of arrays ({len(new_arrays)}) must be same as length as cube ({len(self)})."
            )
        if not all(isinstance(arr, np.ndarray) for arr in new_arrays):
            raise ValueError("Must be list of numpy ndarrays")

        self.data = [raster.update(array=arr) for raster, arr in zip(self, new_arrays)]

    @property
    def raster_type(self) -> Series:
        return Series(
            [r.__class__ for r in self],
            name="raster_type",
        )

    @property
    def band(self) -> Series:
        return Series(
            [r.band for r in self],
            name="band",
        )

    @property
    def dtype(self) -> Series:
        return Series(
            [r.dtype for r in self],
            name="dtype",
        )

    @property
    def nodata(self) -> int | None:
        return self._nodata

    @property
    def path(self) -> Series:
        return self.raster_attribute("path")

    @property
    def name(self) -> Series:
        return self.raster_attribute("name")

    @property
    def date(self) -> Series:
        return self.raster_attribute("date")

    @property
    def indexes(self) -> Series:
        return self.raster_attribute("indexes")

    # @property
    # def raster_id(self) -> Series:
    #     return self.raster_attribute("raster_id")

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
    def res(self) -> int:
        return self._res

    @res.setter
    def res(self, value):
        self._res = value

    @property
    def crs(self) -> pyproj.CRS:
        crs = self._warped_crs if hasattr(self, "_warped_crs") else self._crs
        if crs is not None:
            return crs
        try:
            get_common_crs(self.data)
        except ValueError:
            return None

    @property
    def unary_union(self) -> Geometry:
        return shapely.unary_union([shapely.box(*r.bounds) for r in self])

    @property
    def centroid(self) -> GeoSeries:
        return GeoSeries(
            [r.centroid for r in self],
            name="centroid",
            crs=self.crs,
        )

    @property
    def tile(self) -> Series:
        return self.raster_attribute("tile")

    @property
    def boxes(self) -> GeoSeries:
        """GeoSeries of each raster's bounds as polygon."""
        return GeoSeries(
            [shapely.box(*r.bounds) if r.bounds is not None else None for r in self],
            name="boxes",
            crs=self.crs,
        )

    @property
    def total_bounds(self) -> tuple[float, float, float, float]:
        return tuple(x for x in self.boxes.total_bounds)

    @property
    def bounds(self) -> BoundingBox:
        """Pytorch bounds of the index.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) of the dataset
        """
        return BoundingBox(*self.index.bounds)

    def copy(self, deep=True) -> Self:
        """Returns a (deep) copy of the class instance and its rasters.

        Args:
            deep: Whether to return a deep or shallow copy. Defaults to True.
        """
        copied = deepcopy(self) if deep else copy(self)
        copied.data = [raster.copy() for raster in copied]
        return copied

    def _check_for_array(self, text="") -> None:
        mess = "Arrays are not loaded. " + text
        if all(raster.array is None for raster in self):
            raise ValueError(mess)

    def __getitem__(
        self, item: slice | int | Series | Sequence | Callable | Geometry | BoundingBox
    ) -> Self | Raster:
        """

        Examples
        --------
        >>> cube = sg.DataCube.from_root(testdata, endswith=".tif", crs=25833).load()

        List slicing:

        >>> cube[1:3]
        >>> cube[3:]

        Single integer returns a Raster, not a cube.

        >>> cube[1]

        Boolean conditioning based on cube properties and pandas boolean Series:

        >>> cube[(cube.length > 0) & (cube.path.str.contains("FRC_B"))]
        >>> cube[lambda x: (x.length > 0) & (x.path.str.contains("dtm"))]

        """
        copy = self.copy()
        if isinstance(item, slice):
            copy.data = copy.data[item]
            return copy
        elif isinstance(item, int):
            return copy.data[item]
        elif callable(item):
            item = item(copy)
        elif isinstance(item, BoundingBox):
            return cube_to_torch(self, item)

        elif isinstance(item, (GeoDataFrame, GeoSeries, Geometry)) or is_bbox_like(
            item
        ):
            item = to_shapely(item)
            copy.data = [
                raster for raster in copy.data if raster.bounds.intersects(item)
            ]
            return copy

        copy.data = [
            raster
            for raster, condition in zip(copy.data, item, strict=True)
            if condition
        ]

        return copy

    def __setattr__(self, attr, value):
        if (
            attr in ["data", "_data"]
            or not is_list_like(value)
            or not hasattr(self, "data")
        ):
            return super().__setattr__(attr, value)
        if len(value) != len(self.data):
            raise ValueError(
                "custom cube attributes must be scalar or same length as number of rasters. "
                f"Got self.data {len(self)} and new attribute {len(value)}"
            )
        return super().__setattr__(attr, value)

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)})"

    # def __mul__(self, scalar) -> Self:
    #     return self.map(_mul, scalar=scalar)

    # def __add__(self, scalar) -> Self:
    #     return self.map(_add, scalar=scalar)

    # def __sub__(self, scalar) -> Self:
    #     return self.map(_sub, scalar=scalar)

    # def __truediv__(self, scalar) -> Self:
    #     return self.map(_truediv, scalar=scalar)

    # def __floordiv__(self, scalar) -> Self:
    #     return self.map(_floordiv, scalar=scalar)

    # def __pow__(self, scalar) -> Self:
    #     return self.map(_pow, scalar=scalar)


def concat_cubes(cube_list: list[DataCube], res: int | None = None) -> DataCube:
    if not all(isinstance(cube, DataCube) for cube in cube_list):
        raise TypeError

    return DataCube(
        list(itertools.chain.from_iterable([cube.data for cube in cube_list])), res=res
    )


def clipmerge(cube: DataCube, mask, **kwargs) -> DataCube:
    return merge(cube, bounds=mask, **kwargs)


def merge(
    cube: DataCube,
    by=None,
    bounds=None,
    **kwargs,
) -> DataCube:
    if not all(r.array is None for r in cube):
        raise ValueError("Arrays can't be loaded when calling merge.")

    bounds = to_bbox(bounds) if bounds is not None else bounds

    if by is None:
        return _merge(
            cube,
            bounds=bounds,
            **kwargs,
        )

    elif isinstance(by, str):
        by = [by]
    elif not is_list_like(by):
        raise TypeError("'by' should be string or list like.", by)

    df = DataFrame(
        {"i": range(len(cube)), "tile": cube.tile} | {x: getattr(cube, x) for x in by}
    )

    grouped_indices = df.groupby(by)["i"].unique()
    indices = Series(range(len(cube)))

    return concat_cubes(
        [
            _merge(
                cube[indices.isin(idxs)],
                bounds=bounds,
            )
            for idxs in grouped_indices
        ],
        res=cube.res,
    )


def merge_by_bounds(
    cube: DataCube,
    by=None,
    bounds=None,
    **kwargs,
) -> DataCube:
    if isinstance(by, str):
        by = [by, "tile"]
    elif by is None:
        by = ["tile"]
    else:
        by = by + ["tile"]

    return merge(
        cube,
        by=by,
        bounds=bounds,
        **kwargs,
    )


def _merge(cube, **kwargs) -> DataCube:
    if cube.crs is None:
        cube._crs = get_common_crs(cube.data)

    indexes = cube[0].indexes_as_tuple()

    datasets = [load_raster(raster.path) for raster in cube]
    array, transform = rasterio_merge.merge(datasets, indexes=indexes, **kwargs)
    cube.data = [Raster.from_array(array, crs=cube.crs, transform=transform)]

    return cube

    if all(arr is None for arr in cube.arrays):
        datasets = [load_raster(raster.path) for raster in cube]
        array, transform = rasterio_merge.merge(datasets, indexes=indexes, **kwargs)
        cube.data = [Raster.from_array(array, crs=cube.crs, transform=transform)]
        return cube

    bounds = kwargs.pop("bounds", None)

    if bounds:
        xarrays = [
            r.to_xarray().transpose("y", "x")
            for r in cube.explode()
            if r.intersects(bounds)
        ]
    else:
        xarrays = [r.to_xarray().transpose("y", "x") for r in cube.explode()]

    if len(xarrays) > 1:
        merged = merge_arrays(
            xarrays,
            bounds=bounds,
            res=cube.res,
            nodata=cube.nodata,
            **kwargs,
        )
    else:
        try:
            merged = xarrays[0]
        except IndexError:
            cube.data = []
            return cube

    array = merged.to_numpy()

    raster = cube[0].__class__
    out_bounds = bounds or cube.total_bounds
    cube.data = [raster.from_array(array, bounds=out_bounds, crs=cube.crs)]

    return cube


def load_raster(path):
    with opener(path) as file:
        return rasterio.open(file)


def numpy_to_torch(array: np.ndarray) -> torch.Tensor:
    # fix numpy dtypes which are not supported by pytorch tensors
    if array.dtype == np.uint16:
        array = array.astype(np.int32)
    elif array.dtype == np.uint32:
        array = array.astype(np.int64)

    return torch.tensor(array)


def cube_to_torch(cube: DataCube, query: BoundingBox):
    bbox = shapely.box(*to_bbox(query))
    if cube.separate_files:
        cube = cube.sfilter(bbox).explode().load()
    else:
        cube = cube.clipmerge(bbox).explode()

    data: torch.Tensor = torch.cat([numpy_to_torch(array) for array in cube.arrays])

    key = "image" if cube.is_image else "mask"
    sample = {key: data, "crs": cube.crs, "bbox": query}
    return sample
