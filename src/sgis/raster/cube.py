import functools
import glob
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyproj
import shapely
from affine import Affine
from geopandas import GeoDataFrame
from pandas import DataFrame, Series
from pandas.api.types import is_list_like
from shapely import Geometry

from ..geopandas_tools.to_geodataframe import to_gdf
from ..io.dapla import check_files
from ..multiprocessing.multiprocessingmapper import MultiProcessingMapper
from .raster import Raster, RasterBase


def _clip_func(raster, mask, **kwargs):
    """Method-to-function to use as mapping function."""
    return raster.clip(mask, **kwargs)


def _zonal_func(raster, **kwargs):
    """Method-to-function to use as mapping function."""
    return raster.zonal(**kwargs)


def _to_crs_func(raster, **kwargs):
    """Method-to-function to use as mapping function."""
    return raster.to_crs(**kwargs)


def _set_crs_func(raster, **kwargs):
    """Method-to-function to use as mapping function."""
    return raster.set_crs(**kwargs)


def _load_func(raster, **kwargs):
    """Method-to-function to use as mapping function."""
    return raster.load(**kwargs)


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


class BaseCubeTemplate:
    filename_glob = "*"
    filename_regex = None

    all_bands: list[str] | None = None
    rgb_bands: list[str] | None = None

    indexes: int | tuple[int] | None = None
    nodata: float | int | None = None

    def __init__(self, rasters):
        self.rasters = rasters

    @classmethod
    def names(path):
        if path is None:
            return None
        return Path(path).stem

    def get_df(self):
        return DataFrame(
            {
                "name": [self.names(r.path) for r in self.rasters],
                "band": [],
            }
        )


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
    def __init__(
        self,
        rasters=None,
        index=None,
        df=None,
        use_multiprocessing: bool = True,
        context="spawn",
        dapla=True,
        processes=None,
    ) -> None:
        self.use_mp = use_multiprocessing

        self.df = self._df_template(rasters, df, index=index)

        if self.use_mp:
            self.mapper = MultiProcessingMapper(
                context=context, dapla=dapla, processes=processes
            )

        self._hash = uuid.uuid4()

        crs = list({r.crs for r in self.df["raster"]})
        if len(crs) > 1:
            raise ValueError("'crs' mismatch.")
        self._crs = crs

    @classmethod
    def _df_template(cls, rasters: list[Raster], df=None, index=None):
        if isinstance(rasters, Raster):
            rasters = [rasters]
        if not is_list_like(rasters) and all(isinstance(r, Raster) for r in rasters):
            raise TypeError

        raster_df = DataFrame({"raster": rasters}, index=index)

        if df is None:
            raster_df["name"] = pd.NA
            raster_df["path"] = pd.NA
            raster_df["band_indexes"] = pd.NA
            raster_df["band_type"] = pd.NA  # rbg, not
            raster_df["tile"] = pd.NA
            raster_df["cube"] = 0
            return raster_df
        elif len(df) != len(raster_df):
            raise ValueError

        return pd.concat([df, raster_df], axis=1)

    @classmethod
    def from_root(
        cls,
        root: str | Path,
        *,
        raster_type: Raster,
        contains: str | None = None,
        regex: str | None = None,
        indexes: int | list[int] | None = None,
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

        return cls.from_paths(
            files, template=template, raster_type=raster_type, **kwargs
        )

    @classmethod
    def from_paths(
        cls,
        paths: list[str | Path],
        *,
        raster_type: Raster,
        indexes: int | list[int] | None = None,
        nodata: float | int | None = None,
        dapla: bool = False,
        template: Any = BaseCubeTemplate,
        **kwargs,
    ):
        if not isinstance(raster_type, Raster):
            raise TypeError("raster_type must be Raster or a subclass.")

        rasters = [
            raster_type.from_path(path, indexes=indexes, nodata=nodata, dapla=dapla)
            for path in paths
        ]
        df = DataFrame(
            {
                "name": [r.name for r in rasters],
                "path": paths,
                "band_indexes": [x.indexes_as_tuple() for x in rasters],
            }
        )

        return cls(rasters, df=df, **kwargs)

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
        if self.use_mp:
            gdfs: list[GeoDataFrame] = self.mapper.map(
                _zonal_func, self.arrays, **kwargs
            )
        else:
            gdfs: list[GeoDataFrame] = [
                _zonal_func(arr, **kwargs) for arr in self.arrays
            ]

        out = []
        for i, gdf in zip(self.index, gdfs):
            gdf["raster_index"] = i
            out.append(gdf)
        return pd.concat(out, ignore_index=True)

    def load(self, **kwargs):
        return self._delegate_raster_func(_load_func, **kwargs)

    def map(self, func, **kwargs):
        return self._delegate_array_func(func, **kwargs)

    def clip(self, mask, **kwargs):
        if not pyproj.CRS(self.crs).equals(pyproj.CRS(mask.crs)):
            raise ValueError("crs mismatch.")

        # first remove rows not within mask
        clipped = to_gdf(self.unary_union, crs=self.crs).clip(mask)
        self.df = self.df.loc[self.df.index.isin(clipped.index)]

        return self._delegate_raster_func(_clip_func, mask=mask, **kwargs)

        if self.use_mp:
            self.df["raster"] = self.mapper.map(
                _clip_func, self.df["raster"], mask=mask, **kwargs
            )
        else:
            partial_func = functools.partial(_clip_func, mask=mask, **kwargs)
            self.df["raster"] = self.df["raster"].apply(partial_func)

        return self

    def write_tifs(self, folder: str, names: dict | None = None):
        """Writes all arrays as tif files in given folder and updates path attribute.

        This method should be run after the rasters have been clipped, merged or
        its array values have been recalculated.
        """
        names = {} if not names else names
        for i, raster in self.df["raster"].items():
            name = names.get(i, raster.name)
            path = str(Path(folder) / Path(name).stem) + ".tif"
            raster.write_tif(path)
            raster.path = path
            self.df.loc[i, "path"] = path
        self._rasters_have_changed = False

    def to_gdf(self, ignore_index: bool = False, mask=None):
        gdf_list = []
        for i, raster in enumerate(self.df["raster"]):
            row = self.df.iloc[[i]].drop("raster", axis=1)
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

    def merge_by_band(self, **kwargs):
        if self._rasters_have_changed or any(
            r._raster_has_changed or r._array_has_changed() for r in self.df["raster"]
        ):
            raise RasterHasChangedError("merge_by_band")

        band_df = pd.DataFrame({"band": self.df["band"].unique()})
        new_df: list[pd.DataFrame] = [
            pd.DataFrame(columns=_df_template),
            band_df,
        ]
        for i, band_idx in band_df["band"].items():
            band_group = self.df.loc[self.df["band"] == band_idx]

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

    def merge_tiles(self):
        for band in self.df["band"].unique():
            df = self.df[self.df["band"] == band]

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
        df = self.df.copy()
        df[["src_index", "cube"]] = df[["src_index", "cube"]].astype(str)
        src_indexes = (
            df.groupby("tile")["src_index"].apply(list).str.join("-").str.strip("-")
        )
        cubes = df.groupby("tile")["cube"].unique().str.join("-").str.strip("-")

        for tile in self.df["tile"]:
            df = self.df[self.df["tile"] == tile]

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

            self.df = pd.DataFrame(
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
            # self.df["bounds"] = self.bounds
            return self

    def explode(self, ignore_index: bool = False):
        pass

    def merge(self, by: str | list[str] | None = None, **kwargs):
        if self._rasters_have_changed or any(
            r._raster_has_changed or r._array_has_changed() for r in self.df["raster"]
        ):
            raise RasterHasChangedError("merge")

        if by is None:
            return self._merge_all(**kwargs)

        if isinstance(by, str):
            by = [by]
        if not is_list_like(by):
            raise TypeError("'by' should be string or list like.")

        unique = self.df[by].drop_duplicates().set_index(by)

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

        unique["raster"] = self.df.groupby(by, as_index=False, dropna=False).apply(
            lambda x: _get_merged_raster(x, **kwargs)
        )

        self.df = unique.reset_index()

        return self

        if isinstance(by, str):
            by = [by]
        if not is_list_like(by):
            raise TypeError("'by' should be string or list like.")

        unique = self.df[by].drop_duplicates().reset_index(drop=True)
        new_df = [
            pd.DataFrame(columns=_df_template),
            unique,
        ]

        for i in unique.index:
            group = self.df.loc[[i]]

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
            r._raster_has_changed or r._array_has_changed() for r in self.df["raster"]
        ):
            raise RasterHasChangedError("merge_by_band")

        band_df = pd.DataFrame({"band": self.df["band"].unique()})
        new_df: list[pd.DataFrame] = [
            pd.DataFrame(columns=_df_template),
            band_df,
        ]
        for i, band_idx in band_df["band"].items():
            band_group = self.df.loc[self.df["band"] == band_idx]

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
            r._raster_has_changed or r._array_has_changed() for r in self.df["raster"]
        ):
            raise RasterHasChangedError("merge_all")

        array, transform = merge.merge(list(self.df["path"]), **kwargs)
        meta = {
            "res": self._get_single_res(self.df),
            "transform": transform,
            "crs": self.crs,
        }
        new_df = pd.DataFrame(columns=_df_template, index=[0])
        new_df["raster"] = Raster(array=array, meta=meta)

        cube = self.copy()
        cube.df = new_df
        return cube

    def min(self):
        return min([x._min() for x in self.df["raster"]])

    def max(self):
        return max([x._max() for x in self.df["raster"]])

    def mean(self):
        return np.mean([x._mean() for x in self.df["raster"]])

    @property
    def raster(self):
        return self.df["raster"]

    @raster.setter
    def raster(self, new_value):
        self.df["raster"] = new_value
        return self.df["raster"]

    @property
    def arrays(self) -> list[np.ndarray]:
        return [r.array for r in self.df["raster"]]

    @property
    def band_indexes(self):
        return pd.Series(
            {i: r.indexes_as_tuple() for i, r in self.df["raster"].items()},
            name="indexes",
        )

    @property
    def index(self):
        return self.df.index

    @index.setter
    def index(self, new_value):
        self.df.index = new_value
        return self.index

    @property
    def shape(self):
        return pd.Series(
            {i: r.shape for i, r in self.df["raster"].items()}, name="shape"
        )

    @property
    def count(self):
        return pd.Series(
            {i: r.count for i, r in self.df["raster"].items()}, name="count"
        )

    @property
    def res(self) -> tuple[float, float]:
        return pd.Series({i: r.res for i, r in self.df["raster"].items()}, name="res")

    @property
    def crs(self):
        if hasattr(self, "_warped_crs"):
            return self._warped_crs
        return self._crs

    @property
    def unary_union(self) -> shapely.geometry.Polygon:
        return shapely.box(*self.total_bounds)

    @property
    def bounds(self) -> Series:
        return Series(
            {i: r.bounds for i, r in self.df["raster"].items()}, name="bounds"
        )

    @property
    def total_bounds(self) -> tuple[float, float, float, float]:
        minx = np.min([r.bounds[0] for r in self.df["raster"]])
        miny = np.min([r.bounds[1] for r in self.df["raster"]])
        maxx = np.min([r.bounds[2] for r in self.df["raster"]])
        maxy = np.min([r.bounds[3] for r in self.df["raster"]])
        return minx, miny, maxx, maxy

    @property
    def tiles(self) -> Series:
        self._set_tile_values()
        return self.df["tile"]

    def _set_tile_values(self):
        mapper = {bounds: i for i, bounds in enumerate(self.bounds.unique())}
        self.df["tile"] = self.bounds.map(mapper)

    def _set_band_indexes(self):
        self.df["src_index"] = (
            self.bounds.to_frame()
            .assign(src_index=1)
            .groupby("bounds")["src_index"]
            .transform("cumsum")
        )

    def _push_raster_col(self):
        col = self.df["raster"]
        self.df = self["df"].reindex(
            columns=[c for c in self["df"].columns if c != col] + [col]
        )
        return self

    def __hash__(self):
        return hash(self._hash)

    def __iter__(self):
        return iter(self.df["raster"])

    def __len__(self):
        return len(self.df["raster"])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(df=" "\n" f"{self.df.__repr__()}" "\n)"

    def __setattr__(self, __name: str, __value) -> None:
        return super().__setattr__(__name, __value)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def _delegate_raster_func(self, func, **kwargs):
        if self.use_mp:
            self.df["raster"] = self.mapper.map(func, self.df["raster"], **kwargs)
        else:
            self.df["raster"] = [func(r, **kwargs) for r in self.df["raster"]]

        return self

    def _delegate_array_func(self, func, **kwargs):
        if self.use_mp:
            arrays = self.mapper.map(func, self.arrays, **kwargs)
        else:
            arrays = [func(arr, **kwargs) for arr in self.arrays]

        for (i, raster), arr in zip(self.df["raster"].items(), arrays):
            raster.array = arr
            self.df.loc[i, "raster"] = raster

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
