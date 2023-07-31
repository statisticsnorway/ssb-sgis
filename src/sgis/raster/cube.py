import functools
import multiprocessing
import re
import uuid
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
import xarray as xr
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame, Series
from pandas.api.types import is_list_like
from rasterio.enums import MergeAlg
from shapely import Geometry

from ..geopandas_tools.bounds import make_grid
from ..geopandas_tools.general import get_common_crs, to_shapely
from ..helpers import (
    dict_zip_intersection,
    get_all_files,
    get_func_name,
    get_non_numpy_func_name,
    get_numpy_func,
)
from ..io._is_dapla import is_dapla
from .raster import Raster


try:
    from ..io.dapla import check_files, read_geopandas, write_geopandas
except ImportError:
    pass
from .cubebase import (
    CubeBase,
    _add,
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
from .cubepool import CubePool
from .elevationraster import ElevationRaster
from .explode import explode
from .indices import ndvi_formula
from .merge import cube_merge, merge_by_bounds
from .sample import RandomCubeSample
from .sentinel import Sentinel2
from .zonal import make_geometry_iterrows, prepare_zonal, zonal_func, zonal_post


# TODO: hvordan blir kolonnene etter merge/retile?
# etter retile:
# - name må beholdes
# - name må være name+tile - altså groupby+grid_id
#     - er det noe annet enn name man vil groupe by? Nei?
# - må lagres i mappene de lå i, altså folder_name
# -
# - Merge by [name, folder_name] ??

# name:
# - raster: tile+name+date

# TODO: raster has property navn, mens name kan settes?


CANON_RASTER_TYPES = {
    "Raster": Raster,
    "ElevationRaster": ElevationRaster,
    "Sentinel2": Sentinel2,
}

CUBE_DF_NAME = "cube_df.parquet"


class GeoDataCube(CubeBase):
    def __init__(
        self,
        data: Raster | Iterable[Raster] | None = None,
        df: DataFrame | None = None,
        root: str | None = None,
        crs: Any | None = None,
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

        self._df = df if df is not None else pd.DataFrame()
        self._df["raster"] = list(data)
        self.update_df()

        if crs:
            self._crs = pyproj.CRS(crs)
            if not all(self._crs.equals(pyproj.CRS(r.crs)) for r in data):
                self = self.to_crs(self._crs)
        elif hasattr(self, "_test") and self._test:
            try:
                self._crs = get_common_crs(self.df["raster"])
            except ValueError:
                pass
        else:
            self._crs = get_common_crs(self.df["raster"])

    @classmethod
    def from_root(
        cls,
        root: str | Path,
        *,
        band_index: int | Iterable[int] | None = None,
        raster_type: Raster = Raster,
        check_for_df: bool = True,
        contains: str | None = None,
        endswith: str = ".tif",
        regex: str | None = None,
        processes: int | None = None,
        **kwargs,
    ):
        kwargs = {
            "raster_type": raster_type,
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
            return cls.from_paths(
                paths, band_index=band_index, root=root, processes=processes, **kwargs
            )

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
                    processes=processes,
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
        paths: Iterable[str | Path],
        *,
        root: str | None = None,
        raster_type: Raster = Raster,
        band_index: int | tuple[int] | None = None,
        processes: int | None = None,
        **kwargs,
    ):
        crs = kwargs.pop("crs", None)

        if not isinstance(raster_type, type):
            raise TypeError("raster_type must be Raster or a subclass.")

        if not issubclass(raster_type, Raster):
            raise TypeError("raster_type must be Raster or a subclass.")

        if not is_list_like(paths) and not all(
            isinstance(path, (str, Path)) for path in paths
        ):
            raise TypeError

        func = functools.partial(
            _raster_from_path,
            raster_type=raster_type,
            band_index=band_index,
            **kwargs,
        )

        if processes is None:
            rasters = [func(path) for path in paths]
        else:
            with multiprocessing.get_context("spawn").Pool(processes) as pool:
                rasters = pool.map(func, paths)

        return cls(
            rasters,
            root=root,
            copy=False,
            crs=crs,
        )

    @classmethod
    def from_gdf(
        cls,
        gdf: GeoDataFrame | Iterable[GeoDataFrame],
        columns: str | Iterable[str],
        res: int,
        processes: int,
        tilesize: int | None = None,
        tiles: GeoSeries | None = None,
        raster_type: Raster = Raster,
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
            raster_type=raster_type,
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
        raster_type: Raster = Raster,
    ):
        raster_type = cls.get_raster_type(raster_type)

        if isinstance(df, (str, Path)):
            df = read_geopandas(df) if is_dapla() else gpd.read_parquet(df)

        if isinstance(df, DataFrame):
            raster_attrs, other_attrs = cls._prepare_gdf_for_raster(df)
            rasters = [
                raster_type.from_dict(dict(row[1])) for row in raster_attrs.iterrows()
            ]
            cube = cls(rasters, df=other_attrs)
            cube._from_cube_df = True
            return cube

        elif all(isinstance(x, (str, Path, DataFrame)) for x in df):
            cubes = [cls.from_cube_df(x, raster_type=raster_type) for x in df]
            cube = concat_cubes(cubes, ignore_index=True)
            cube._from_cube_df = True
            return cube

        raise TypeError("df must be DataFrame or file path to a parquet file.")

    def update_rasters(self, *args):
        # TODO: droppe?
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

    def add_meta_to_df(self, *args):
        # TODO: droppe?
        for arg in args:
            self._df[arg] = self.raster_attribute(arg).values
        return self

    def update_df(self):
        # TODO: internal?
        for col in self.BASE_CUBE_COLS:
            if col == "raster":
                self._df[col] = [r for r in self]
            try:
                self._df[col] = self.raster_attribute(col).values
            except AttributeError:
                pass
        self._df = self._df.replace({None: pd.NA})

        other_cols = list(self.df.columns.difference(self.BASE_CUBE_COLS))
        self._df = self._df[self.BASE_CUBE_COLS + other_cols]

    @staticmethod
    def get_raster_type(raster_type):
        # TODO: internal?
        if not isinstance(raster_type, type):
            if isinstance(raster_type, str) and raster_type in CANON_RASTER_TYPES:
                return CANON_RASTER_TYPES[raster_type]
            else:
                raise TypeError("'raster_type' must be Raster or a subclass.")

        if not issubclass(raster_type, Raster):
            raise TypeError("'raster_type' must be Raster or a subclass.")

        return raster_type

    def most_common_raster_type(self):
        # TODO: internal?
        try:
            return list(self.raster_type.value_counts().index)[0]
        except IndexError:
            return list(self.raster_type.value_counts().index)

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
        poly_iter = make_geometry_iterrows(polygons)

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

        return zonal_post(
            aggregated,
            polygons=polygons,
            idx_mapper=idx_mapper,
            idx_name=idx_name,
            dropna=dropna,
        )

    def gradient(self, degrees: bool = False, copy: bool = False):
        cube = self.copy() if copy else self
        if not all(isinstance(r, ElevationRaster) for r in cube):
            raise TypeError("raster_type must be ElevationRaster.")
        cube.df["raster"] = cube.run_raster_method("gradient", degrees=degrees)
        return cube

    def pool(self, processes: int, copy: bool = True) -> CubePool:
        cube = self.copy() if copy else self
        return CubePool(cube, processes=processes)

    def array_map(self, func: Callable, **kwargs):
        """Maps each raster array to a function.

        The function must take a numpu array as first positional argument,
        and return a single numpy array. The function should be defined in
        the leftmost indentation level. If in Jupyter, the function also
        have to be defined in and imported from another file.
        """
        return self._delegate_array_func(func, **kwargs)

    def raster_map(self, func: Callable, **kwargs):
        """Maps each raster to a function.

        The function must take a Raster object as first positional argument,
        and return a single Raster object. The function should be defined in
        the leftmost indentation level. If in Jupyter, the function also
        have to be defined in and imported from another file.
        """
        return self._delegate_raster_func(func, **kwargs)

    def query(self, query: str, copy: bool = True, **kwargs):
        cube = self.copy() if copy else self
        cube.df = cube.df.query(query, **kwargs)
        return cube

    def load(self, res: int | None = None, copy: bool = True, **kwargs):
        cube = self.copy() if copy else self

        cube.df["raster"] = cube.run_raster_method("load", res=res, **kwargs)
        return cube

    def clip(self, mask, res: int | None = None, copy: bool = True, **kwargs):
        cube = self.copy() if copy else self

        cube = cube.clip_base(mask)

        cube.df["raster"] = cube.run_raster_method("clip", mask=mask, res=res, **kwargs)
        return cube

    def intersection(self, df, res: int | None = None, **kwargs):
        cubes = []
        for _, row in df.iterrows():
            cube = intersection_base(row, cube=self, res=res, **kwargs)
            cubes.append(cube)

        return concat_cubes(cubes, ignore_index=True)

    def ndvi(self, band_name_red, band_name_nir, copy=True):
        return self._index_calc(
            band_name1=band_name_red,
            band_name2=band_name_nir,
            index_formula=ndvi_formula,
            index_name="ndvi",
            copy=copy,
        )

    def sample(self, n=1, buffer=1000, mask=None, **kwargs):
        return RandomCubeSample(self, n=n, buffer=buffer, mask=mask, **kwargs)

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
        self.write_base(subfolder_col=subfolder_col, filename=filename, root=root)

        return self._delegate_raster_func(_write_func, **kwargs)

    def write_df(self, folder: str):
        gdf: GeoDataFrame = self._prepare_df_for_parquet()

        if is_dapla():
            write_geopandas(gdf, Path(folder) / CUBE_DF_NAME)
        else:
            gdf.to_parquet(Path(folder) / CUBE_DF_NAME)

        return self

    def to_gdf(self, ignore_index: bool = False, concat: bool = True) -> GeoDataFrame:
        gdfs = self.run_raster_method("to_gdf")
        if concat:
            return pd.concat(gdfs, ignore_index=ignore_index)
        return gdfs

    def to_xarray(self, index_col: str | None = None) -> xr.Dataset:
        if index_col and self.df[index_col].duplicated().any():
            raise ValueError("Cannot have duplicate indices.")
        if index_col:
            arrays = {
                i: r.to_xarray() for i, r in zip(self.df[index_col], self.df["raster"])
            }
        else:
            arrays = {i: r.to_xarray() for i, r in enumerate(self.df["raster"])}
        return xr.Dataset(arrays)

    def reproject_match(self):
        pass

    def filter_by_location(self, other, copy: bool = True):
        other = to_shapely(other)
        cube = self.copy() if copy else self
        cube._df = cube._df[cube.boxes.interesects(other)]
        return cube

    def to_crs(self, crs, copy: bool = True):
        cube = self.copy() if copy else self
        cube.df["raster"] = [_to_crs_func(r, crs=crs) for r in cube]
        cube._warped_crs = crs
        return cube

    def set_crs(self, crs, allow_override: bool = False, copy: bool = True):
        cube = self.copy() if copy else self
        cube.df["raster"] = [
            _set_crs_func(r, crs=crs, allow_override=allow_override) for r in cube
        ]
        cube._warped_crs = crs
        return cube

    def explode(self, ignore_index: bool = False):
        return explode(self, ignore_index=ignore_index)

    """def explode(self, column=None, ignore_index: bool = False, **kwargs):
        # If no column is specified then default to the active geometry column
        if column is None:
            return super().explode(column, ignore_index=ignore_index, **kwargs)
        return explode(self, ignore_index=ignore_index)"""

    def clipmerge(
        self,
        mask,
        by: str | list[str] | None = None,
        res=None,
        aggfunc="first",
        copy: bool = True,
        **kwargs,
    ):
        return self.merge(
            by=by,
            bounds=mask,
            res=res,
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
        dropna: bool = False,
        copy: bool = True,
        **kwargs,
    ):
        cube = self.copy() if copy else self
        kwargs = {
            "by": by,
            "bounds": bounds,
            "res": res,
            "aggfunc": aggfunc,
            "dropna": dropna,
        } | kwargs

        return cube_merge(cube, **kwargs)

    def merge_by_bounds(
        self,
        bounds=None,
        res=None,
        aggfunc="first",
        dropna: bool = False,
        copy: bool = True,
        **kwargs,
    ):
        """Merge rasters with the same bounds to a 3 dimensional array.

        The 'name' column will be updated to the bounds as a string.
        """

        cube = self.copy() if copy else self
        kwargs = {
            "bounds": bounds,
            "res": res,
            "aggfunc": aggfunc,
            "dropna": dropna,
            **kwargs,
        }
        return merge_by_bounds(cube, **kwargs)

    def dissolve_bands(self, aggfunc, copy: bool = True):
        self.check_for_array()
        if not callable(aggfunc) and not isinstance(aggfunc, str):
            raise TypeError("Can only supply a single aggfunc")

        cube = self.copy() if copy else self

        aggfunc = get_numpy_func(aggfunc)

        cube = cube._delegate_array_func(aggfunc, axis=0)
        cube.update_df()
        return cube

    def min(self):
        return min(x.min() for x in self)

    def max(self):
        return max(x.max() for x in self)

    def raster_attribute(self, attribute: str) -> Series:
        """Get a Raster attribute returned as values of a Series."""
        return Series(
            [getattr(r, attribute) for r in self],
            index=self._df.index,
            name=attribute,
        )

    def run_raster_method(self, method: str, *args, copy: bool = True, **kwargs):
        """Run Raster methods."""
        if not all(hasattr(r, method) for r in self):
            raise AttributeError(f"Raster has no method {method!r}.")

        method_as_func = functools.partial(
            _method_as_func, method=method, *args, **kwargs
        )

        cube = self.copy() if copy else self

        cube.df["raster"] = [method_as_func(r) for r in cube]
        return cube

    @property
    def raster(self):
        return self._raster

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, new_df):
        self.validate_cube_df(new_df)
        self._df = new_df
        self.update_df()
        return self._df

    @property
    def meta(self):
        cube_df = DataFrame(index=self._df.index)
        for col in self.ALL_ATTRS:
            try:
                cube_df[col] = self.raster_attribute(col).values
            except Exception:
                pass
        return cube_df

    @property
    def raster_type(self):
        return Series(
            [r.__class__ for r in self],
            index=self._df.index,
            name="raster_type",
        )

    @property
    def dtype(self):
        return Series(
            [r.dtype for r in self],
            index=self._df.index,
            name="dtype",
        )

    @property
    def nodata(self) -> Series:
        return self.raster_attribute("nodata")

    @property
    def arrays(self) -> Series:
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
    def path(self) -> Series:
        return self.raster_attribute("path")

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
    def name(self) -> Series:
        return self.raster_attribute("name")

    @property
    def raster_id(self) -> Series:
        return self.raster_attribute("raster_id")

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

    def equals(self, other, verbose: bool = False) -> bool:
        if not isinstance(other, GeoDataCube):
            raise NotImplementedError

        equal_length = len(self) == len(other)
        equal_attributes = sum(r.equals(r2) for r, r2 in zip(self, other))

        if equal_length and equal_attributes == len(self):
            return True
        if not verbose:
            return False

        if len(self) != len(other):
            print("len:", len(self), len(other))

        print(f"Number of equal Rasters: {equal_attributes} of {len(self)}")

        print("unequal attributes")
        for i, (r1, r2) in enumerate(zip(self, other)):
            print("Row", i)

            for key, value1, value2 in dict_zip_intersection(r1.__dict__, r2.__dict__):
                try:
                    equals = value1 == value2
                except ValueError:
                    if isinstance(value1, np.ndarray):
                        equals = np.all(np.array_equal(value1, value2)).all()
                    else:
                        equals = (value1).equals(value2).all()
                print(equals)
                if not equals:
                    print(key, value1, value2)

        return False

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

    def check_for_array(self, text=""):
        mess = "Arrays are not loaded. " + text
        if self.arrays.isna().all():
            raise ValueError(mess)

    def __hash__(self):
        return hash(self._hash)

    def __iter__(self):
        return iter(self._df["raster"])

    def __len__(self):
        return len(self._df)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(df=\n" f"{self._df.__repr__()}\n" ")"

    def __setattr__(self, __name: str, __value) -> None:
        return super().__setattr__(__name, __value)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        if not isinstance(key, str):
            return self._df["raster"].iloc[key]
        return getattr(self, key)

    def _delegate_raster_func(self, func, **kwargs):
        self.df["raster"] = [func(r, **kwargs) for r in self]
        return self

    def _delegate_array_func(self, func, **kwargs):
        self.check_for_array()
        self.arrays = [func(arr, **kwargs) for arr in self.arrays]
        return self

    def __mul__(self, scalar):
        return self._delegate_array_func(_mul, scalar=scalar)

    def __add__(self, scalar):
        return self._delegate_array_func(_add, scalar=scalar)

    def __sub__(self, scalar):
        return self._delegate_array_func(_sub, scalar=scalar)

    def __truediv__(self, scalar):
        return self._delegate_array_func(_truediv, scalar=scalar)

    def __floordiv__(self, scalar):
        return self._delegate_array_func(_floordiv, scalar=scalar)

    def __pow__(self, scalar):
        return self._delegate_array_func(_pow, scalar=scalar)


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
