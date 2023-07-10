import functools
import glob
import itertools
import uuid
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyproj
import shapely
from affine import Affine
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame, Series
from pandas.api.types import is_list_like
from shapely import Geometry

from ..geopandas_tools.to_geodataframe import to_gdf
from ..io.dapla import check_files
from ..multiprocessing.multiprocessingmapper import MultiProcessingMapper
from .base import RasterBase
from .elevationraster import ElevationRaster
from .raster import Raster


CANON_RASTER_TYPES = {
    "Raster": Raster,
    "ElevationRaster": ElevationRaster,
}


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

    """
    'load' will read all the images as numpy ndarrays.
    'clip' reads only the part of the images in the mask extent.
    'add_meta' will load all the
    """

    def __init__(
        self,
        rasters: Raster | list[Raster] | None = None,
        index=None,
        df=None,
        multiprocessingmapper: MultiProcessingMapper | None = None,
        copy=False,
        dapla=False,
    ) -> None:
        if isinstance(rasters, Raster):
            rasters = [rasters]
        if not is_list_like(rasters) and all(isinstance(r, Raster) for r in rasters):
            raise TypeError("'rasters' must be a Raster instance or an iterable.")

        if copy:
            rasters = [raster.copy() for raster in rasters]
        else:
            # take a copy only if there are gdfs with the same id
            if sum(r1 is r2 for r1 in rasters for r2 in rasters) > len(rasters):
                rasters = [raster.copy() for raster in rasters]

        if df is None:
            # concat rows
            df = self.get_cube_template()
            self._df = pd.concat([df, pd.DataFrame({"raster": rasters})])

        elif len(df) != len(rasters):
            raise ValueError("'df' must be same length as rasters.")
        else:
            # concat columns
            self._df = pd.concat(
                [df, pd.DataFrame({"raster": rasters}, index=df.index)], axis=1
            )

        crs = list({r.crs for r in self._df["raster"]})
        if len(crs) > 1:
            raise ValueError("'crs' mismatch.")
        elif len(crs) == 0:
            self._crs = None
        else:
            self._crs = crs[0]

        if multiprocessingmapper:
            self.validate_mapper(multiprocessingmapper)

        self.mapper = multiprocessingmapper

        self._hash = uuid.uuid4()

    def _get_crs(self):
        crs = list({r.crs for r in self._df["raster"]})
        if len(crs) > 1:
            raise ValueError("'crs' mismatch.")
        return crs[0]

    @classmethod
    def from_root(
        cls,
        root: str | Path,
        *,
        raster_dtype: Raster = Raster,
        contains: str | None = None,
        endswith: str | None = None,
        regex: str | None = None,
        band_indexes: int | list[int] | None = None,
        nodata: float | int | None = None,
        template: Any = BaseCubeTemplate,
        dapla: bool = False,
        name: str | None = None,
        **kwargs,
    ):
        if dapla:
            files = check_files(root, contains=contains)["path"]
        else:
            files = [file for file in glob.glob(str(Path(root)) + "/*")]
            if contains:
                files = [file for file in files if contains in file]
            if endswith:
                files = [file for file in files if file.endswith(endswith)]

        return cls.from_paths(
            files, template=template, raster_dtype=raster_dtype, **kwargs
        )

    @classmethod
    def from_paths(
        cls,
        paths: list[str | Path],
        *,
        raster_dtype: Raster = Raster,
        band_indexes: int | list[int] | None = None,
        nodata: float | int | None = None,
        dapla: bool = False,
        template: Any = BaseCubeTemplate,
        multiprocessingmapper: MultiProcessingMapper | None = None,
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

        if multiprocessingmapper:
            cls.validate_mapper(multiprocessingmapper)
            orig_processes = multiprocessingmapper.processes
            if multiprocessingmapper.processes > len(paths) / 50:
                multiprocessingmapper.processes = len(paths) / 50
            rasters = multiprocessingmapper.map(
                _init_raster_from_path,
                paths,
                raster_dtype=raster_dtype,
                dapla=dapla,
                **kwargs,
            )
            multiprocessingmapper.processes = orig_processes

        else:
            rasters = [
                raster_dtype.from_path(
                    path, band_indexes=band_indexes, nodata=nodata, dapla=dapla
                )
                for path in paths
            ]
        df = DataFrame(
            {
                "name": [r.name for r in rasters],
                "path": paths,
                # "band_indexes": [x.band_indexes_as_tuple() for x in rasters],
            }
        )

        return cls(
            rasters,
            df=df,
            multiprocessingmapper=multiprocessingmapper,
            dapla=dapla,
            **kwargs,
        )

    @classmethod
    def from_gdfs(cls, gdfs: list[GeoDataFrame]):
        if isinstance(gdfs, GeoDataFrame):
            gdfs = [gdfs]
        if not all(isinstance(gdf, GeoDataFrame) for gdf in gdfs):
            raise TypeError

    @classmethod
    def from_df(
        cls, df: DataFrame, raster_dtype: Raster = Raster, multiprocessingmapper=None
    ):
        raster_dtype = cls.get_raster_type(raster_dtype)

        rasters = [raster_dtype.from_dict(dict(row[1])) for row in df.iterrows()]

        return cls(rasters, multiprocessingmapper=multiprocessingmapper)

    def to_dict(self) -> dict[int, dict]:
        return {i: r.to_dict() for i, r in self.df["raster"].items()}

    def update_df(self):
        df = self._df
        # df["__i"] = range(len(df))

        for col in self.BASE_CUBE_COLS:
            try:
                new_values = {
                    i: getattr(row["raster"], col) for i, row in df.iterrows()
                }
            except AttributeError as e:
                if col not in str(e):
                    raise e
            df[col.lstrip("_")] = new_values
            """for i in df["__i"]:
                raster: Raster = df.loc[df["__i"] == i, "raster"].iloc[0]
                try:
                    # df.loc[df["__i"] == i, col] = getattr(raster, col)
                    df.loc[df["__i"] == i, col] = {i: getattr(raster, col)}
                except AttributeError:
                    pass"""

        self._df = df  # .drop("__i", axis=1)
        return self

    def to_df(self, path: str | Path):
        self.update_df()
        df = self.df.drop("raster", axis=1)
        if self.dapla:
            dp.write_pandas(df, path)
        elif ".parquet" in path:
            df.to_parquet(path)
        elif ".csv" in path:
            df.to_csv(path)
        elif ".json" in path:
            df.to_json(path)

    @staticmethod
    def get_raster_type(raster_dtype):
        if not isinstance(raster_dtype, type):
            if isinstance(raster_dtype, str) and raster_dtype in CANON_RASTER_TYPES:
                return CANON_RASTER_TYPES[raster_dtype]
            else:
                raise TypeError("'raster_dtype' must be Raster or a subclass.")

        if not issubclass(raster_dtype, Raster):
            raise TypeError("'raster_dtype' must be Raster or a subclass.")

        return raster_dtype

    def astype(self, dtype: type):
        dtype = self.get_raster_type(dtype)

        for i in self._df["_ii"]:
            pass

    def astype_array(self, dtype):
        self._delegate_array_func(_array_astype_func, dtype=dtype)

    def zonal(
        self,
        polygons: GeoDataFrame,
        aggfunc: str | Callable | list[Callable | str],
        raster_calc_func: Callable | None = None,
        dropna: bool = True,
    ) -> GeoDataFrame:
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

    def load(self, **kwargs):
        self._delegate_raster_func(_load_func, **kwargs)
        self.update_df()
        return self

    def map(self, func, **kwargs):
        return self._delegate_array_func(func, **kwargs)

    def clip(self, mask, **kwargs):
        if (
            hasattr(mask, "crs")
            and mask.crs
            and not pyproj.CRS(self.crs).equals(pyproj.CRS(mask.crs))
        ):
            raise ValueError("crs mismatch.")

        # first remove rows not within mask
        clipped = self.boxes.clip(mask)
        self._df = self._df.loc[self._df.index.isin(clipped.index)]

        if not len(self._df):
            return self

        self._delegate_raster_func(_clip_func, mask=mask, **kwargs)
        self.update_df()
        return self

    def write_df(self, path: str):
        """Write the DataFrame of the Cube with all metadata but not the rastercolumn."""
        pass

    def write(self, folder: str):
        """Writes all arrays as tif files in given folder and updates path attribute.

        This method should be run after the rasters have been clipped, merged or
        its array values have been recalculated.
        """
        if self._df["name"].isna().any():
            raise ValueError("")
        self._df["__i"] = range(len(self._df))
        filt = lambda x: x["__i"] == i
        for i, raster in zip(self._df["__i"], self._df["raster"]):
            path = str(Path(folder) / Path(raster.name).stem) + ".tif"
            values = self.get_raster_dict(self._df[filt])
            raster.update(**values)
            raster.write(path)
            raster.path = path
            self._df.loc[filt, "path"] = path
        self._df = self._df.drop("__i", axis=1)

    def to_gdf(self, ignore_index: bool = False, mask=None):
        gdf_list = []
        for i, raster in enumerate(self._df["raster"]):
            row = self._df.iloc[[i]].drop("raster", axis=1)
            gdf = raster.to_gdf(mask=mask)
            gdf.index = np.repeat(i, len(gdf))
            for col in row.columns:
                gdf[col] = row[col].iloc[0]
            gdf_list.append(gdf)
        return GeoDataFrame(
            pd.concat(gdf_list, ignore_index=ignore_index),
            geometry="geometry",
            crs=self.crs,
        )

    def filter_by_bounds(self, other):
        self._df = self._df[self.boxes.interesects(other)]
        return self

    def intersects(self, other):
        return self.boxes.intersects(other)

    def to_crs(self, crs):
        self = self._delegate_raster_func(_to_crs_func, crs=crs)
        self._warped_crs = crs
        return self

    def set_crs(self, crs, allow_override: bool = False):
        self = self._delegate_raster_func(
            _set_crs_func, crs=crs, allow_override=allow_override
        )
        self._warped_crs = crs
        return self

    def reindex_by_band(self, **kwargs):
        pass

    def merge_by_band(self, **kwargs):
        if self._rasters_have_changed or any(
            r._raster_has_changed or r._array_has_changed() for r in self._df["raster"]
        ):
            raise RasterHasChangedError("merge_by_band")

        band_df = pd.DataFrame({"band": self._df["band"].unique()})
        new_df: list[pd.DataFrame] = [
            pd.DataFrame(columns=_df_template),
            band_df,
        ]
        for i, band_idx in band_df["band"].items():
            band_group = self._df.loc[self._df["band"] == band_idx]

            array, transform = merge.merge(
                list(band_group["path"]), indexes=band_idx, **kwargs
            )

            meta = {
                "res": self._get_single_res(band_group),
                "transform": transform,
                "crs": self.crs,
            }
            band_df.at[i, "raster"] = Raster(array=array, meta=meta)

            new_df.append(band_df)

        new_df = pd.concat(new_df)
        cube = self.copy()
        cube.df = new_df
        return cube

    def add_meta(self):
        """Adds metadata columns to the instance's df attribute."""
        self._delegate_raster_func(_add_meta)
        for col in self.ALL_ATTRS:
            try:
                self.df[col] = self._get_attribute_series(col)
            except AttributeError:
                pass
        return self

    def merge_tiles(self):
        for band in self._df["band"].unique():
            df = self._df[self._df["band"] == band]

            src_idx = src_indexes.loc[band]
            dst_idx = dst_indexes.loc[band]

            array = np.stack([raster.array for raster in df["raster"]], axis=1)
            res = df["res"].value_counts()[0]
            minx = np.min(bounds[0] for bounds in df["bounds"])
            miny = np.min(bounds[1] for bounds in df["bounds"])
            maxx = np.min(bounds[2] for bounds in df["bounds"])
            maxy = np.min(bounds[3] for bounds in df["bounds"])
            diffx = maxx - minx
            diffy = maxy - miny
            width, height = int(diffx / res), int(diffy / res)
            _transform = transform.from_bounds(minx, miny, maxx, maxy, width, height)
            raster = Raster.from_array(array, transform=_transform)
            df = pd.DataFrame(
                {
                    "cube": cube,
                    "raster": raster,
                    "tile": tile,
                    "src_index": src_idx,
                    "dst_index": dst_idx,
                    # "bounds": self.bounds,
                }
            )

    def merge_bands(self):
        df = self._df.copy()
        df[["src_index", "cube"]] = df[["src_index", "cube"]].astype(str)
        src_indexes = (
            df.groupby("tile")["src_index"].apply(list).str.join("-").str.strip("-")
        )
        cubes = df.groupby("tile")["cube"].unique().str.join("-").str.strip("-")

        for tile in self._df["tile"]:
            df = self._df[self._df["tile"] == tile]

            src_idx = src_indexes.loc[tile]
            cube = cubes.loc[tile]

            arrays = []
            for raster in df["raster"]:
                arrays += raster._to_2d_array_list(raster.array)

            array = np.stack(arrays, axis=0)
            res = df["res"].value_counts().iloc[0]
            minx = np.min([r.bounds[0] for r in df["raster"]])
            miny = np.min([r.bounds[1] for r in df["raster"]])
            maxx = np.min([r.bounds[2] for r in df["raster"]])
            maxy = np.min([r.bounds[3] for r in df["raster"]])
            diffx = maxx - minx
            diffy = maxy - miny
            width, height = int(diffx / res), int(diffy / res)
            _transform = transform.from_bounds(minx, miny, maxx, maxy, width, height)
            name = f"{cube}_{tile}_{src_idx}"
            raster = Raster.from_array(
                array,
                transform=_transform,
                res=res,
                crs=self.crs,
                name=name,
            )

            self._df = pd.DataFrame(
                {
                    "cube": cube,
                    "raster": raster,
                    "tile": tile,
                    "src_index": src_idx,
                    "name": raster.name,
                    "res": res,
                },
                index=[0],
            )
            # self._df["bounds"] = self.bounds
            return self

    def explode(self, ignore_index: bool = False):
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
                raster._set_array_stats()
            elif raster.array is None:
                raster.indexes = band_idx

            df.loc[filt, "raster"] = raster

        self._df = df.drop(["__i", "id", "duplicate_id"], axis=1)
        return self

    def merge(self, by: str | list[str] | None = None, **kwargs):
        if self._rasters_have_changed or any(
            r._raster_has_changed or r._array_has_changed() for r in self._df["raster"]
        ):
            raise RasterHasChangedError("merge")

        if by is None:
            return self._merge_all(**kwargs)

        if isinstance(by, str):
            by = [by]
        if not is_list_like(by):
            raise TypeError("'by' should be string or list like.")

        unique = self._df[by].drop_duplicates().set_index(by)

        def _get_merged_raster(group, **kwargs):
            res = self._get_single_res(group)
            bands = list(group["band"].unique())
            array, transform = merge.merge(list(group["path"]), indexes=bands, **kwargs)

            meta = {
                "res": res,
                "transform": transform,
                "crs": self.crs,
            }

            return Raster(array=array, meta=meta)

        unique["raster"] = self._df.groupby(by, as_index=False, dropna=False).apply(
            lambda x: _get_merged_raster(x, **kwargs)
        )

        self._df = unique.reset_index()

        return self

        if isinstance(by, str):
            by = [by]
        if not is_list_like(by):
            raise TypeError("'by' should be string or list like.")

        unique = self._df[by].drop_duplicates().reset_index(drop=True)
        new_df = [
            pd.DataFrame(columns=_df_template),
            unique,
        ]

        for i in unique.index:
            group = self._df.loc[[i]]

            array, transform = merge.merge(
                list(group["path"]), indexes=band_idx, **kwargs
            )

            meta = {
                "res": self._get_single_res(group),
                "transform": transform,
                "crs": self.crs,
            }
            band_df.at[i, "raster"] = Raster(array=array, meta=meta)

            new_df.append(band_df)

        new_df = pd.concat(new_df)
        cube = self.copy()
        cube.df = new_df
        return cube

    def merge_by_band(self, **kwargs):
        if self._rasters_have_changed or any(
            r._raster_has_changed or r._array_has_changed() for r in self._df["raster"]
        ):
            raise RasterHasChangedError("merge_by_band")

        band_df = pd.DataFrame({"band": self._df["band"].unique()})
        new_df: list[pd.DataFrame] = [
            pd.DataFrame(columns=_df_template),
            band_df,
        ]
        for i, band_idx in band_df["band"].items():
            band_group = self._df.loc[self._df["band"] == band_idx]

            res = []
            for raster in band_group["raster"]:
                res.append(raster.res)

            if len(set(res)) > 1:
                raise ValueError("res mismatch", res)
            res = res[0]

            array, transform = merge.merge(
                list(band_group["path"]), indexes=band_idx, **kwargs
            )

            meta = {
                "res": res,
                "transform": transform,
                "crs": self.crs,
            }
            band_df.at[i, "raster"] = Raster(array=array, meta=meta)

            new_df.append(band_df)

        new_df = pd.concat(new_df)
        cube = self.copy()
        cube.df = new_df
        return cube

    def _merge_all(self, **kwargs):
        if self._rasters_have_changed or any(
            r._raster_has_changed or r._array_has_changed() for r in self._df["raster"]
        ):
            raise RasterHasChangedError("merge_all")

        array, transform = merge.merge(list(self._df["path"]), **kwargs)
        meta = {
            "res": self._get_single_res(self._df),
            "transform": transform,
            "crs": self.crs,
        }
        new_df = pd.DataFrame(columns=_df_template, index=[0])
        new_df["raster"] = Raster(array=array, meta=meta)

        cube = self.copy()
        cube.df = new_df
        return cube

    def min(self):
        return min([x._min() for x in self._df["raster"]])

    def max(self):
        return max([x._max() for x in self._df["raster"]])

    def mean(self):
        return np.mean([x._mean() for x in self._df["raster"]])

    def _get_attribute_series(self, attribute: str) -> pd.Series:
        return pd.Series(
            {i: getattr(r, attribute) for i, r in self._df["raster"].items()},
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
    def _rasters_have_changed(self) -> bool:
        return any(r._raster_has_changed for r in self._df["raster"])

    @property
    def arrays(self) -> list[np.ndarray]:
        return self._get_attribute_series("array")

    @property
    def band_indexes(self) -> pd.Series:
        if "band_indexes" in self._df:
            return self._df["band_indexes"]
        return pd.Series(
            {i: r.band_indexes_as_tuple() for i, r in self._df["raster"].items()},
            name="band_indexes",
        )

    @property
    def cube_indexes(self):
        pass

    @property
    def tiles(self) -> Series:
        self._set_tile_values()
        return self._df["tile"]

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
        if hasattr(self, "_warped_crs"):
            return self._warped_crs
        return self._crs

    @property
    def unary_union(self) -> Geometry:
        return shapely.unary_union([shapely.box(*bounds) for bounds in self.bounds])

    @property
    def bounds(self) -> Series:
        return self._get_attribute_series("bounds")

    @property
    def boxes(self) -> GeoSeries:
        """GeoSeries of each raster's bounds as polygon."""
        return GeoSeries(
            {i: shapely.box(*r.bounds) for i, r in self._df["raster"].items()},
            name="boxes",
            crs=self.crs,
        )

    @property
    def total_bounds(self) -> tuple[float, float, float, float]:
        minx = np.min([r.bounds[0] for r in self._df["raster"]])
        miny = np.min([r.bounds[1] for r in self._df["raster"]])
        maxx = np.min([r.bounds[2] for r in self._df["raster"]])
        maxy = np.min([r.bounds[3] for r in self._df["raster"]])
        return minx, miny, maxx, maxy

    def _set_tile_values(self):
        mapper = {bounds: i for i, bounds in enumerate(self.bounds.unique())}
        self._df["tile"] = self.bounds.map(mapper)

    def _set_band_indexes(self):
        self._df["src_index"] = (
            self.bounds.to_frame()
            .assign(src_index=1)
            .groupby("bounds")["src_index"]
            .transform("cumsum")
        )

    def _push_raster_col(self):
        col = self._df["raster"]
        self._df = self["df"].reindex(
            columns=[c for c in self["df"].columns if c != col] + [col]
        )
        return self

    def copy(self, deep=True):
        """Returns a (deep) copy of the class instance.

        Args:
            deep: Whether to return a deep or shallow copy. Defaults to True.
        """
        if deep:
            copied = deepcopy(self)
        else:
            copied = copy(self)

        df = copied.df.copy(deep=deep)

        df["__i"] = range(len(df))
        for i, raster in zip(df["__i"], df["raster"]):
            df.loc[df["__i"] == i, "raster"] = raster.copy(deep=deep)

        copied._df = df.drop("__i", axis=1)

        return copied

    @classmethod
    def verify_cube_df(cls, df):
        if not isinstance(df, (pd.DataFrame, pd.Series)):
            raise TypeError
        # check that nessecary columns are not removed
        cls.verify_dict(df)

    @classmethod
    def get_cube_template(cls, with_meta: bool = False):
        cols = cls.BASE_CUBE_COLS
        if with_meta:
            cols = cols + cls.PROFILE_ATTRS
        return pd.DataFrame(columns=cols)

    @classmethod
    def get_raster_dict(cls, df):
        return df[[key for key in cls.ALLOWED_KEYS if key in df]].to_dict()

    def __hash__(self):
        return hash(self._hash)

    def __iter__(self):
        return iter(self._df["raster"])

    def __len__(self):
        return len(self._df["raster"])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(df=" "\n" f"{self._df.__repr__()}" "\n)"

    def __setattr__(self, __name: str, __value) -> None:
        return super().__setattr__(__name, __value)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def _delegate_raster_func(self, func, **kwargs):
        if self.mapper:
            self._df["raster"] = self.mapper.map(func, self._df["raster"], **kwargs)
        else:
            self._df["raster"] = [func(r, **kwargs) for r in self._df["raster"]]

        return self

    def _delegate_array_func(self, func, **kwargs):
        if self.mapper:
            arrays = self.mapper.map(func, self.arrays, **kwargs)
        else:
            arrays = [func(arr, **kwargs) for arr in self.arrays]

        for (i, raster), arr in zip(self._df["raster"].items(), arrays):
            raster.array = arr
            self._df.loc[i, "raster"] = raster

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

    crs = {cube.crs for cube in cube_list}
    if len(crs) > 1:
        raise ValueError("crs mismatch.")

    rasters: list[Raster] = [x for cube in cube_list for x in cube.df["raster"]]

    rasters: list[Raster] = []
    for i, cube in enumerate(cube_list):
        for raster in cube.df["raster"]:
            raster.cube = i
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

    cube = GeoDataCube(rasters, index=indexes, df=df)

    cube._set_tile_values()
    cube._set_band_indexes()

    return cube


def _init_raster_from_path(path, raster_dtype, **kwargs):
    return raster_dtype.from_path(path, **kwargs)


def _add_meta(raster, **kwargs):
    """Method-to-function to use as mapping function."""
    return raster.add_meta(**kwargs)


def _clip_func(raster, mask, **kwargs):
    """Method-to-function to use as mapping function."""
    # values = self.get_raster_dict(row)
    # raster.update(**values)
    return raster.clip(mask, **kwargs)


def _load_func(raster, **kwargs):
    """Method-to-function to use as mapping function."""
    # values = self.get_raster_dict(row)
    # raster.update(**values)
    return raster.load(**kwargs)


def _zonal_func(raster, **kwargs):
    """Method-to-function to use as mapping function."""
    return raster.zonal(**kwargs)


def _to_crs_func(raster, **kwargs):
    """Method-to-function to use as mapping function."""
    return raster.to_crs(**kwargs)


def _set_crs_func(raster, **kwargs):
    """Method-to-function to use as mapping function."""
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
