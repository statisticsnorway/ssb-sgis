import functools
import glob
import itertools
import multiprocessing
import re
import uuid
import warnings
from copy import copy, deepcopy
from pathlib import Path, WindowsPath
from typing import Any, Callable, Iterable


try:
    import dapla as dp
except ModuleNotFoundError:
    pass

import numpy as np
import pandas as pd
import pyproj
import shapely
from affine import Affine
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame, Series
from pandas.api.types import is_list_like
from rasterio import merge
from shapely import Geometry

from ..geopandas_tools.bounds import is_bbox_like, to_bbox
from ..geopandas_tools.general import to_shapely
from ..geopandas_tools.to_geodataframe import to_gdf
from ..io.dapla import check_files
from ..multiprocessing.multiprocessingmapper import MultiProcessingMapper
from .base import RasterBase
from .cubechain import CubeChain
from .elevationraster import ElevationRaster
from .raster import Raster, get_numpy_func


CANON_RASTER_TYPES = {
    "Raster": Raster,
    "ElevationRaster": ElevationRaster,
}

CUBE_DF_NAME = "cube_df.parquet"


# class RandomCubeSampler:
class RandomCubeSample:
    def __init__(self, cube, n):
        self.cube = cube
        self.samples

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


class BaseCubeTemplate:
    filename_glob = "*"
    filename_regex = None

    all_bands: list[str] | None = None
    rgb_bands: list[str] | None = None

    band_indexes: int | tuple[int] | None = None
    nodata: float | int | None = None

    raster_dtype = Raster
    dtype = np.uint8

    def __init__(self, rasters):
        self.rasters = rasters

    @classmethod
    def names(path):
        if path is None:
            return None
        return Path(path).stem


class Sentinel2:
    filename_glob = "T*_*_B02_*m.*"
    filename_regex = r"""
        ^T(?P<tile>\d{2}[A-Z]{3})
        _(?P<date>\d{8}T\d{6})
        _(?P<band>B[018][\dA])
        _(?P<resolution>\d{2}m)
        \..*$
    """
    date_format = "%Y%m%dT%H%M%S"

    # https://gisgeography.com/sentinel-2-bands-combinations/
    all_bands = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B10",
        "B11",
        "B12",
    ]
    rgb_bands = ["B04", "B03", "B02"]

    separate_files = True

    @classmethod
    def names(path):
        return Path(path).stem


class GeoDataCube(RasterBase):
    _chain = None
    _executing = False

    def __init__(
        self,
        data: Raster | Iterable[Raster] | None = None,
        df: DataFrame | None = None,
        copy: bool = False,
        dapla: bool = False,
    ) -> None:
        self._arrays = None
        self._crs = None
        self._hash = uuid.uuid4()

        # so this can be set on class as well as instance
        if not hasattr(self, "dapla"):
            self.dapla = dapla

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

        self._df = self.get_cube_template(data)

        """elif len(df) != len(data):
            raise ValueError("'df' must be same length as data.")
        else:
            # concat columns
            self._df = pd.concat(
                [df, pd.DataFrame({"raster": data}, index=df.index)], axis=1
            )"""

        self._crs = self._get_crs()

    @classmethod
    def from_root(
        cls,
        root: str | Path,
        *,
        band_indexes: int | list[int] | None = None,
        raster_dtype: Raster = Raster,
        check_for_df: bool = True,
        cube_names="folder",
        contains: str | None = None,
        endswith: str = ".tif",
        regex: str | None = None,
        template: Any = BaseCubeTemplate,
        dapla: bool = False,
        **kwargs,
    ):
        kwargs = {
            "dapla": dapla,
            "raster_dtype": raster_dtype,
        } | kwargs
        if dapla:
            files = list(check_files(root, contains=contains)["path"])
        else:
            files = [file for file in glob.glob(str(Path(root)) + "/*")]

        dfs = [file for file in files if file.endswith(CUBE_DF_NAME)]

        if contains:
            files = [file for file in files if contains in file]
        if endswith:
            files = [file for file in files if file.endswith(endswith)]
        if regex:
            regex = re.compile(regex)
            files = [file for file in files if re.search(regex, file)]

        if not files:
            raise ValueError("Found no files matching the pattern.")

        if not check_for_df or not len(dfs):
            return cls.from_paths(files, band_indexes=band_indexes, **kwargs)

        folders_with_df = {Path(file).parent for file in dfs if file}
        if len(dfs) != len(folders_with_df):
            raise ValueError(
                "More than one cube_df.parquet file found in at least one folder."
            )

        cubes = [cls.from_df(df, **kwargs) for df in dfs]

        files_in_folders_without_df = [
            file for file in files if Path(file).parent not in folders_with_df
        ]

        if files_in_folders_without_df:
            cubes = cubes + [
                cls.from_paths(
                    files_in_folders_without_df, band_indexes=band_indexes, **kwargs
                )
            ]

        if len(cubes) == 1:
            return cubes[0]

        cube = concat_cubes(cubes, ignore_index=True)

        def get_folder(path):
            try:
                return Path(Path(path).parent).stem
            except TypeError:
                return None

        if cube_names == "folder":
            cube.df["cube_name"] = [get_folder(path) for path in cube.df["path"]]

        cube._from_df = True
        return cube

    @classmethod
    def from_paths(
        cls,
        paths: list[str | Path],
        *,
        raster_dtype: Raster = Raster,
        band_indexes: int | tuple[int] | None = None,
        dapla: bool = False,
        template: Any = BaseCubeTemplate,
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
                band_indexes=band_indexes,
                dapla=dapla,
            )
            for path in paths
        ]

        return cls(
            rasters,
            dapla=dapla,
            **kwargs,
        )

    @classmethod
    def from_gdf(
        cls,
        gdfs: GeoDataFrame | list[GeoDataFrame],
        tilesize: int,
    ):
        if isinstance(gdf, GeoDataFrame):
            gdf = [gdf]
        if not all(isinstance(frame, GeoDataFrame) for frame in gdf):
            raise TypeError

    @classmethod
    def from_df(
        cls,
        df: DataFrame | str | Path,
        raster_dtype: Raster = Raster,
        dapla: bool = False,
    ):
        raster_dtype = cls.get_raster_dtype(raster_dtype)

        if isinstance(df, (str, Path)):
            df = dp.read_pandas(df) if dapla else pd.read_parquet(df)

        if isinstance(df, DataFrame):
            rasters = [raster_dtype.from_dict(dict(row[1])) for row in df.iterrows()]
            cube = cls(rasters, dapla=dapla)
            cube._from_df = True
            return cube

        elif all(isinstance(x, (str, Path, DataFrame)) for x in df):
            names = [cls.get_name(x) for x in df]
            cubes = [cls.from_df(x, raster_dtype=raster_dtype, dapla=dapla) for x in df]
            cube = concat_cubes(cubes, cube_index=names, ignore_index=True)
            cube._from_df = True
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

    def update_df(self):
        df = self._df

        # for col in ["width", "height", "transform", "bounds", "path", "name"]:
        for col in ["bounds", "path", "name"]:
            try:
                new_values = [getattr(row["raster"], col) for _, row in df.iterrows()]
            except AttributeError as e:
                if col not in str(e):
                    raise e
            df[col.lstrip("_")] = new_values

        for col in [
            "band_indexes",
            "crs",
        ]:
            df[col] = getattr(self, col)

        self._df = df
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
        self._write_in_chain()
        if self._chain is not None and not self._executing:
            self._chain.append_method("astype", dtype=dtype)
            return self

        dtype = self.get_raster_dtype(dtype)

        self._df["raster"] = [dtype(raster) for raster in self._df["raster"]]

        return self

    def astype_array(self, dtype):
        self.check_for_array("dtype can be set as a parameter in load and clip.")
        self._write_in_chain()
        if self._chain is not None:
            self._chain.append_array_func(_array_astype_func, dtype=dtype)
            return self

        self._delegate_array_func(_array_astype_func, dtype=dtype)

    def zonal(
        self,
        polygons: GeoDataFrame,
        aggfunc: str | Callable | list[Callable | str],
        raster_calc_func: Callable | None = None,
        dropna: bool = True,
    ) -> GeoDataFrame:
        self._check_not_chain()
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

    def chain(self, *, processes: int):
        """TODO: navn pool? multiprocessing_chain?"""
        if self._chain is not None and len(self._chain):
            warnings.warn("A chain is already started. Starting a new one.")
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
            for func, _, typ, kwargs in self._chain:
                if typ == "raster":
                    func = functools.partial(func, **kwargs)
                    self._df["raster"] = pool.map(func, self._df["raster"])
                elif typ == "array":
                    func = functools.partial(func, **kwargs)
                    self.arrays = pool.map(func, self.arrays)
                elif hasattr(self, func):
                    self = getattr(self, func)(**kwargs)
                else:
                    raise ValueError

        self.update_df()
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
        self.check_for_array()
        self._write_in_chain()
        if self._chain is not None:
            try:
                self.mapper.validate_execution(func)
            except AttributeError:
                pass
            self._chain.append_array_func(func, **kwargs)
            return self
        return self._delegate_array_func(func, **kwargs)

    def load(self, **kwargs):
        # TODO til felles laveste res???
        self.check_not_array()
        if self._chain:
            self._chain.append_raster_func(_load_func, **kwargs)
            return self
        self._delegate_raster_func(_load_func, **kwargs)
        self.update_df()
        return self

    def clip(self, mask, crop=True, **kwargs):
        # TODO til felles laveste res???
        self.check_not_array()

        if (
            hasattr(mask, "crs")
            and mask.crs
            and not pyproj.CRS(self.crs).equals(pyproj.CRS(mask.crs))
        ):
            raise ValueError("crs mismatch.")

        # first remove rows not within mask
        self._df = self._df.loc[self.boxes.isin(to_shapely(mask))]

        if not len(self._df):
            return self

        if self._chain is not None:
            self._chain.append_raster_func(_clip_func, mask=mask, crop=crop, **kwargs)
            return self

        self._delegate_raster_func(_clip_func, mask=mask, crop=crop, **kwargs)
        self.update_df()

        return self

    def sample(self, size=256, length=1, mask=None, crop=True, **kwargs):
        self.check_not_array()

        if self._chain:
            self._chain.append_method("sample", **kwargs)

        if mask is not None:
            points = GeoSeries(self.unary_union).clip(mask).sample_points(size=length)
        else:
            points = GeoSeries(self.unary_union).sample_points(size=length)
        buffered = points.buffer(size / self.res[0])
        boxes = to_gdf(
            [shapely.box(*arr) for arr in buffered.bounds.values], crs=self.crs
        )

        return self.clip(boxes, crop=crop, **kwargs)

    def write(self, folder: str, **kwargs):
        """Writes arrays as tif files and df with file info.

        This method should be run after the rasters have been clipped, merged or
        its array values have been recalculated.
        """
        self.check_for_array()

        if self._df["name"].isna().any():
            raise ValueError(
                "Cannot have NA values in the 'name' column when writing files."
            )

        self.verify_cube_df(self._df)

        if self._chain is not None and not self._executing:
            self._chain.append_raster_func(_write_func, folder=folder, **kwargs)
            return self

        # the only two attributes where cube.df is to be trusted over Raster
        self.update_rasters("band_indexes", "path")

        self = self._delegate_raster_func(_write_func, folder=folder, **kwargs)
        self.update_df()

        df = self._prepare_df_for_parquet()

        if self.dapla:
            dp.write_pandas(df, Path(folder) / CUBE_DF_NAME)
        else:
            df.drop("raster", axis=1).to_parquet(Path(folder) / CUBE_DF_NAME)

        return self

    def _prepare_df_for_parquet(self):
        df = self._df.drop(columns=["raster"])
        df["crs"] = df["crs"].astype(str)
        try:
            df["transform"] = df["transform"].apply(tuple)
        except KeyError:
            pass
        if not all(col in df for col in self.BASE_CUBE_COLS + ["bounds"]):
            raise ValueError(
                f"Must have all columns {', '.join(self.BASE_CUBE_COLS + ['bounds'])}"
            )
        return df

    def to_gdf(self, ignore_index: bool = False, concat: bool = True):
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
        self._check_not_chain()
        return self.boxes.intersects(other)

    def to_crs(self, crs):
        if self._chain is not None:
            self._chain.append_raster_func(_to_crs_func, crs=crs)
            return self
        self.update_df()
        self._warped_crs = crs
        return self

    def set_crs(self, crs, allow_override: bool = False):
        if self._chain is not None:
            self._chain.append_raster_func(
                _set_crs_func, crs=crs, allow_override=allow_override
            )
            return self
        self = self._delegate_raster_func(
            _set_crs_func, crs=crs, allow_override=allow_override
        )
        self.update_df()
        self._warped_crs = crs
        return self

    def reindex_by_band(self, **kwargs):
        self.update_rasters("band_indexes")

    def explode(self, ignore_index: bool = False):
        if self._chain and not self._executing:
            self._chain.append_method("explode", ignore_index=ignore_index)
            return self

        df = self._df

        # Raster object is mutable, so dupicates after explode must be copied
        df["id"] = df["band_indexes"].map(id)
        df["duplicate_id"] = df["id"].duplicated()

        df = df.explode(column="band_indexes", ignore_index=ignore_index)
        df["__i"] = range(len(df))
        filt = lambda x: x["__i"] == i

        for i, band_idx, raster in zip(df["__i"], df["band_indexes"], df["raster"]):
            row = df[filt]

            if row["duplicate_id"] is True:
                raster = raster.copy()

            if len(raster.shape) == 3 and raster.array is not None:
                raster.array = raster.array[band_idx - 1]

            raster._band_indexes = band_idx

            df.loc[filt, "raster"] = raster

        self._df = df.drop(["__i", "id", "duplicate_id"], axis=1)

        return self

    def merge(
        self,
        by: str | list[str] | None = None,
        bounds=None,
        res=None,
        raster_dtype=None,
        aggfunc="first",
        **kwargs,
    ):
        self.check_not_array()
        if self._chain is not None and not self._executing:
            self._chain.append_method(
                "merge",
                by=by,
                bounds=bounds,
                res=res,
                raster_dtype=raster_dtype,
                aggfunc=aggfunc,
                **kwargs,
            )
            return self

        if bounds is None:
            self._df[["minx", "miny", "maxx", "maxy"]] = self.bounds.values
        elif is_bbox_like(bounds):
            bounds = to_bbox(bounds)
        else:
            raise TypeError("bounds should be bbox like.")

        if raster_dtype is None:
            raster_dtype = list(self.raster_dtype.unique())
            if len(raster_dtype) > 1:
                raise ValueError
            raster_dtype = raster_dtype[0]
        else:
            raster_dtype = self.get_raster_dtype(raster_dtype)

        if res is None:
            self._df["res"] = self.res

        if by is None:
            return self._merge_all(bounds, res, raster_dtype, **kwargs)

        if isinstance(by, str):
            by = [by]
        if not is_list_like(by):
            raise TypeError("'by' should be string or list like.")

        unique = self._df[by + ["raster"]].drop_duplicates(by).set_index(by)

        if len(unique) == len(self._df):
            self._df = unique.reset_index()
            return self

        unique["raster"] = self._df.groupby(by).apply(
            lambda x: self._grouped_merge(
                x,
                bounds=bounds,
                res=res,
                raster_dtype=raster_dtype,
                **kwargs,
            )
        )

        remaining_cols = self._df.columns.difference(by + ["raster"])
        unique[remaining_cols] = self._df.groupby(by)[remaining_cols].agg(aggfunc)

        self._df = unique.reset_index().drop(
            ["minx", "miny", "maxx", "maxy", "res"], axis=1, errors="ignore"
        )

        self.update_df()

        return self

    def merge_by_band(
        self,
        bounds=None,
        res=None,
        raster_dtype=None,
        aggfunc="first",
        **kwargs,
    ):
        """Merge rasters with the same band_index to a large 2d array."""
        if self._chain is not None and not self._executing:
            self._chain.append_method(
                "merge_by_band",
                bounds=bounds,
                res=res,
                raster_dtype=raster_dtype,
                aggfunc=aggfunc,
                **kwargs,
            )
            return self
        self = self.explode().merge(
            by="band_indexes",
            bounds=bounds,
            res=res,
            raster_dtype=raster_dtype,
            aggfunc=aggfunc,
            **kwargs,
        )
        self.update_df()
        return self

    def merge_by_bounds(
        self,
        bounds=None,
        res=None,
        raster_dtype=None,
        aggfunc="first",
        **kwargs,
    ):
        """Merge rasters with the same bounds to a 3 dimensional array."""

        if self._chain is not None and not self._executing:
            self._chain.append_method(
                "merge_by_bounds",
                bounds=bounds,
                res=res,
                raster_dtype=raster_dtype,
                aggfunc=aggfunc,
                **kwargs,
            )
            return self

        self._df["tile"] = self.bounds_tuples

        self = self.merge(
            by="tile",
            bounds=bounds,
            res=res,
            raster_dtype=raster_dtype,
            _as_3d=True,
            **kwargs,
        )

        self._df = self._df.drop(columns=["tile"])

        self._df["band_indexes"] = self.band_indexes

        return self

    def dissolve_bands(self, aggfunc):
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
        self = self._delegate_array_func(aggfunc, axis=0)
        self.update_df()
        return self

    def _grouped_merge(
        self, group: DataFrame, bounds, raster_dtype, res, _as_3d=False, **kwargs
    ):
        if res is None:
            res = group["res"].min()
        if bounds is None:
            bounds = (
                group["minx"].min(),
                group["miny"].min(),
                group["maxx"].max(),
                group["maxy"].max(),
            )
        exploded = group.explode(column="band_indexes")
        band_indexes = tuple(exploded["band_indexes"].sort_values().unique())
        arrays = []
        for idx in band_indexes:
            paths = exploded.loc[exploded["band_indexes"] == idx, "path"]
            array, transform = merge.merge(
                list(paths), indexes=(idx,), bounds=bounds, res=res, **kwargs
            )
            # merge doesn't allow single index (error in numpy), so changing afterwards
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
        self._check_not_chain()
        return min([x.min() for x in self._df["raster"]])

    def max(self):
        self._check_not_chain()
        return max([x.max() for x in self._df["raster"]])

    def _add_attributes_from_self(self, obj):
        for key, value in self.__dict__.items():
            if key == "_df":
                continue
            obj[key] = value
        return obj

    def _get_attribute_series(self, attribute: str) -> pd.Series:
        return pd.Series(
            [getattr(r, attribute) for r in self._df["raster"]],
            index=self._df.index,
            name=attribute,
        )

    @staticmethod
    def validate_mapper(mapper):
        if not isinstance(mapper, MultiProcessingMapper):
            raise TypeError

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, new_df):
        self.verify_cube_df(new_df)
        self._df = new_df
        return self._df

    @property
    def raster_dtype(self):
        return pd.Series(
            [r.__class__ for r in self._df["raster"]],
            index=self._df.index,
            name="raster_dtype",
        )

    @property
    def arrays(self) -> list[np.ndarray]:
        return self._get_attribute_series("array")

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
    def band_indexes(self) -> pd.Series:
        return pd.Series(
            [r.band_indexes_as_tuple() for r in self._df["raster"]],
            index=self._df.index,
            name="band_indexes",
        )

    @property
    def area(self):
        return self._get_attribute_series("area")

    @property
    def length(self):
        return self._get_attribute_series("length")

    @property
    def height(self):
        return self._get_attribute_series("height")

    @property
    def width(self):
        return self._get_attribute_series("width")

    @property
    def shape(self):
        return self._get_attribute_series("shape")

    @property
    def count(self):
        return self._get_attribute_series("count")

    @property
    def res(self):
        return self._get_attribute_series("res")

    @property
    def crs(self):
        return self._warped_crs if hasattr(self, "_warped_crs") else self._crs

    @property
    def unary_union(self) -> Geometry:
        return shapely.unary_union([shapely.box(*r.bounds) for r in self])

    @property
    def centroid(self):
        return self._get_attribute_series("centroid")

    @property
    def bounds(self) -> Series:
        return DataFrame(
            [r.bounds for r in self._df["raster"]],
            index=self._df.index,
            columns=["minx", "miny", "maxx", "maxy"],
        )

    @property
    def bounds_tuples(self) -> Series:
        # TODO: bedre navn
        return Series(
            [r.bounds for r in self._df["raster"]],
            index=self._df.index,
            name="bounds_tuples",
        )

    @property
    def boxes(self) -> GeoSeries:
        """GeoSeries of each raster's bounds as polygon."""
        return GeoSeries(
            [shapely.box(*r.bounds) for r in self._df["raster"]],
            index=self._df.index,
            name="boxes",
            crs=self.crs,
        )

    @property
    def total_bounds(self) -> tuple[float, float, float, float]:
        bounds = self.bounds
        minx = bounds["minx"].min()
        miny = bounds["miny"].min()
        maxx = bounds["maxx"].min()
        maxy = bounds["maxy"].min()
        return minx, miny, maxx, maxy

    def _set_band_indexes(self):
        self._df["band_indexes"] = (
            self.bounds.to_frame()
            .assign(band_indexes=1)
            .groupby("bounds")["band_indexes"]
            .transform("cumsum")
        )

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
        self._check_not_chain()
        copied = deepcopy(self) if deep else copy(self)

        df = copied.df.copy(deep=deep)

        df["__i"] = range(len(df))
        for i, raster in zip(df["__i"], df["raster"]):
            df.loc[df["__i"] == i, "raster"] = raster.copy(deep=deep)

        copied._df = df.drop("__i", axis=1)

        return copied

    @classmethod
    def verify_cube_df(cls, df):
        if not isinstance(df, (DataFrame, Series)):
            raise TypeError
        # check that nessecary columns are not removed
        cls.verify_dict(df)
        if "raster" not in df:
            raise ValueError("Must have column 'raster'")

        for col in cls.BASE_RASTER_PROPERTIES + cls.NEED_ONE_ATTR:
            if col not in df:
                continue
            if df[col].isna().any():
                raise ValueError(f"Column {col} cannot have missing values.")

    @classmethod
    def get_name(cls, obj: str | Path | DataFrame) -> str | None:
        if isinstance(obj, (str, Path)):
            return Path(obj).parent
        if not isinstance(obj, DataFrame):
            raise TypeError

        if "path" in obj:
            folders = list(obj["path"].apply(lambda x: Path(x).parent).unique())
            if len(folders) == 1:
                return folders[0]
            else:
                return None

    @classmethod
    def get_cube_template(
        cls, rasters: Iterable[Raster] | None = None, with_meta: bool = False
    ):
        cols = cls.BASE_CUBE_COLS
        if with_meta:
            cols = cols + cls.PROFILE_ATTRS
        df = pd.DataFrame(columns=cols + ["raster"])
        if rasters is not None:
            if not all(isinstance(r, Raster) for r in rasters):
                raise TypeError("rasters should be an iterable of Rasters.")
            return pd.concat(
                [
                    df,
                    DataFrame(
                        {
                            "name": [r.name for r in rasters],
                            "path": [r.path for r in rasters],
                            "band_indexes": [x.band_indexes for x in rasters],
                            "bounds": [x.bounds for x in rasters],
                            "raster": rasters,
                        }
                    ),
                ]
            )

    @classmethod
    def get_raster_dict(cls, df):
        return df[[key for key in cls.ALLOWED_KEYS if key in df]].to_dict()

    def _delegate_raster_func(self, func, **kwargs):
        self._df["raster"] = [func(r, **kwargs) for r in self._df["raster"]]
        return self

    def _delegate_array_func(self, func, **kwargs):
        self.arrays = [func(arr, **kwargs) for arr in self.arrays]
        return self

    def _get_crs(self):
        crs = list({r.crs for r in self if r.crs})
        if not crs:
            return None
        if len(crs) > 1:
            raise ValueError("'crs' mismatch.")
        return pyproj.CRS(crs[0])

    def check_for_array(self, text=""):
        mess = "Arrays are not loaded. " + text
        if self.arrays.isna().all():
            raise ValueError(mess)

    def check_not_array(self):
        mess = super().check_not_array_mess()
        if self.arrays.notna().any():
            raise ValueError(mess)
        if not self._chain:
            return
        loading_is_chained = any(
            method in self._chain.names
            for method in ["_load_func", "_clip_func", "sample", "merge"]
        )
        if loading_is_chained:
            raise ValueError(mess)

    def _check_not_chain(self):
        if self._chain is not None:
            raise ValueError("Cannot use this method in a chain.")

    def _write_in_chain(self):
        if self._chain and "_write_func" in self._chain.names:
            raise ValueError("Cannot keep chain going after writing files.")

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
        self.check_for_array()
        self._write_in_chain()
        if self._chain is not None:
            self._chain.append_array_func(_mul, scalar=scalar)
            return self
        return self._delegate_array_func(_mul, scalar=scalar)

    def __add__(self, scalar):
        self.check_for_array()
        self._write_in_chain()
        if self._chain is not None:
            self._chain.append_array_func(_add, scalar=scalar)
            return self
        return self._delegate_array_func(_add, scalar=scalar)

    def __sub__(self, scalar):
        self.check_for_array()
        self._write_in_chain()
        if self._chain is not None:
            self._chain.append_array_func(_sub, scalar=scalar)
            return self
        return self._delegate_array_func(_sub, scalar=scalar)

    def __truediv__(self, scalar):
        self.check_for_array()
        self._write_in_chain()
        if self._chain is not None:
            self._chain.append_array_func(_truediv, scalar=scalar)
            return self
        return self._delegate_array_func(_truediv, scalar=scalar)

    def __floordiv__(self, scalar):
        self.check_for_array()
        self._write_in_chain()
        if self._chain is not None:
            self._chain.append_array_func(_floordiv, scalar=scalar)
            return self
        return self._delegate_array_func(_floordiv, scalar=scalar)

    def __pow__(self, scalar):
        self.check_for_array()
        self._write_in_chain()
        if self._chain is not None:
            self._chain.append_array_func(_pow, scalar=scalar)
            return self
        return self._delegate_array_func(_pow, scalar=scalar)


def concat_cubes(
    cube_list: list[GeoDataCube],
    cube_index: list | np.ndarray | None = None,
    ignore_index: bool = False,
):
    if not all(isinstance(cube, GeoDataCube) for cube in cube_list):
        raise TypeError

    if cube_index is None:
        cube_index = np.arange(0, len(cube_list))
    elif not hasattr(cube_index, "__iter__"):
        raise TypeError("cube_index must be an iterable.")
    elif len(cube_index) != len(cube_list):
        raise ValueError("cube_index must be of same length as cube_list.")

    crs = list({cube.crs for cube in cube_list if cube.crs})
    if not crs:
        crs = None
    if len(crs) > 1:
        raise ValueError("'crs' mismatch.")
    crs = pyproj.CRS(crs[0])

    cubes: list[GeoDataCube] = []
    for idx, cube in zip(cube_index, cube_list):
        cube.df["cube_name"] = idx
        cubes.append(cube)

    cube = GeoDataCube()
    cube.df = pd.concat([cube.df for cube in cubes], ignore_index=ignore_index)
    return cube
    rasters: list[Raster] = [x for cube in cube_list for x in cube.df["raster"]]

    rasters: list[Raster] = []
    for idx, cube in zip(cube_list, cube_index):
        cube.df["cube_name"] = idx
        for raster in cube:
            raster.cube = idx
            rasters.append(raster)

    if ignore_index:
        indexes = np.arange(0, len(rasters))
    else:
        indexes: list[int | Any] = list(
            itertools.chain.from_iterable([cube.df.index for cube in cube_list])
        )

    df = pd.concat([cube.df for cube in cube_list], ignore_index=ignore_index).drop(
        "raster", axis=1
    )

    cube = GeoDataCube(rasters, df=df)

    return cube


"""Method-to-function to use as mapping function."""


def _write_func(raster, folder, **kwargs):
    path = str(Path(folder) / Path(raster.name).stem) + ".tif"
    raster.write(path, **kwargs)
    raster.path = path
    return raster


def _clip_func(raster, mask, **kwargs):
    return raster.clip(mask, **kwargs)


def _load_func(raster, **kwargs):
    return raster.load(**kwargs)


def _zonal_func(raster, **kwargs):
    return raster.zonal(**kwargs)


def _to_crs_func(raster, **kwargs):
    return raster.to_crs(**kwargs)


def _set_crs_func(raster, **kwargs):
    return raster.set_crs(**kwargs)


def write_func(df, folder: str):
    names = {} if not names else names
    df["__i"] = range(len(df))
    filt = lambda x: x["__i"] == i
    for i, raster in zip(df["__i"], df["raster"]):
        row = df[filt]
        path = str(Path(folder) / Path(row["name"]).stem) + ".tif"

        values = self.get_raster_dict(row)
        raster.update(**values)
        raster.write(path)
        raster.path = path
        df.loc[filt, "path"] = path
    df = df.drop("__i", axis=1)


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
