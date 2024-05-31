import os
import abc
import functools
import itertools
import glob
import numbers
import random
import re
from collections.abc import Sequence, Iterable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import ClassVar
from typing import Callable

import dapla as dp
from affine import Affine
import joblib
import numpy as np
from rasterio import features
import pandas as pd
import pyproj
import rasterio
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from rtree.index import Index
from rtree.index import Property
from shapely import Geometry
from shapely import box, unary_union
from shapely.geometry import Point, shape
from shapely.geometry import Polygon, MultiPolygon
from IPython.display import display
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
from collections import defaultdict

try:
    from rioxarray.merge import merge_arrays
    from rioxarray.rioxarray import _generate_spatial_coords
except ImportError:
    pass
try:
    import xarray as xr
    from xarray import DataArray
except ImportError:

    class DataArray:
        """Placeholder."""


try:
    import torch
except ImportError:
    pass

try:
    from torchgeo.datasets.utils import disambiguate_timestamp
except ImportError:
    pass

try:
    from torchgeo.datasets.utils import BoundingBox
except ImportError:

    class BoundingBox:
        """Placeholder."""

        def __init__(self, *args, **kwargs) -> None:
            """Placeholder."""
            raise ImportError("missing optional dependency 'torchgeo'")


from ..geopandas_tools.bounds import to_bbox
from ..geopandas_tools.conversion import to_gdf
from ..geopandas_tools.conversion import to_shapely
from ..geopandas_tools.bounds import get_total_bounds
from ..geopandas_tools.general import get_common_crs
from ..helpers import get_all_files, get_numpy_func
from ..io._is_dapla import is_dapla
from ..io.opener import opener
from . import sentinel_config as config
from .raster import Raster
from .indices import ndvi

if is_dapla():
    ls_func = lambda *args, **kwargs: dp.FileClient.get_gcs_file_system().ls(
        *args, **kwargs
    )
    glob_func = lambda *args, **kwargs: dp.FileClient.get_gcs_file_system().glob(
        *args, **kwargs
    )
    open_func = lambda *args, **kwargs: dp.FileClient.get_gcs_file_system().open(
        *args, **kwargs
    )
else:
    ls_func = functools.partial(get_all_files, recursive=False)
    open_func = open
    glob_func = glob.glob

TORCHGEO_RETURN_TYPE = dict[str, torch.Tensor | pyproj.CRS | BoundingBox]
FILENAME_COL_SUFFIX = "_filename"
DEFAULT_FILENAME_REGEX = r".*\.(?:tif|tiff|jp2)$"


class ImageBase(abc.ABC):
    image_regexes: ClassVar[str | None] = None
    filename_regexes: ClassVar[str | tuple[str]] = (DEFAULT_FILENAME_REGEX,)
    date_format: ClassVar[str] = "%Y%m%dT%H%M%S"

    def __init__(self) -> None:

        if self.filename_regexes:
            if isinstance(self.filename_regexes, str):
                self.filename_regexes = (self.filename_regexes,)
            self.filename_patterns = [
                re.compile(regexes, flags=re.VERBOSE)
                for regexes in self.filename_regexes
            ]
        else:
            self.filename_patterns = None

        if self.image_regexes:
            if isinstance(self.image_regexes, str):
                self.image_regexes = (self.image_regexes,)
            self.image_patterns = [
                re.compile(regexes, flags=re.VERBOSE) for regexes in self.image_regexes
            ]
        else:
            self.image_patterns = None

    def _name_regex_searcher(
        self, group: str, patterns: tuple[re.Pattern]
    ) -> str | None:
        if not patterns or not any(pat.groups for pat in patterns):
            return None
        for pat in patterns:
            try:
                return re.match(pat, self.name).group(group)
            except (AttributeError, IndexError):
                pass
        raise ValueError(
            f"Couldn't find group '{group}' in name {self.name} with regex patterns {patterns}"
        )

    def _create_metadata_df(self, file_paths: list[str]) -> None:
        df = pd.DataFrame({"file_path": file_paths})
        for col in [
            f"band{FILENAME_COL_SUFFIX}",
            f"tile{FILENAME_COL_SUFFIX}",
            f"date{FILENAME_COL_SUFFIX}",
        ]:
            df[col] = None

        df["filename"] = df["file_path"].apply(lambda x: _fix_path(Path(x).name))
        df["image_path"] = df["file_path"].apply(
            lambda x: _fix_path(str(Path(x).parent))
        )

        if not len(df):
            self._df = df
            return

        if self.filename_patterns:
            df, match_cols_filename = _get_regexes_matches_for_df(
                df, "filename", self.filename_patterns, suffix=FILENAME_COL_SUFFIX
            )

            # display(df)
            # display(list(df["filename"]))
            # display(df[match_cols_filename])

            if not len(df):
                self._df = df
                return

            self._match_cols_filename = match_cols_filename
            grouped = (
                df.drop(columns=match_cols_filename)
                .drop_duplicates("image_path")
                .set_index("image_path")
            )
            for col in ["file_path", *match_cols_filename]:
                grouped[col] = df.groupby("image_path")[col].apply(tuple)

            grouped = grouped.reset_index()
        else:
            grouped = df.drop_duplicates("image_path")

        grouped["imagename"] = grouped["image_path"].apply(
            lambda x: _fix_path(Path(x).name)
        )

        if self.image_patterns and len(grouped):
            grouped, _ = _get_regexes_matches_for_df(
                grouped, "imagename", self.image_patterns, suffix=""
            )
            if not len(grouped):
                self._df = grouped
                return

        if "date" in grouped:
            self._df = grouped.sort_values("date")
        else:
            self._df = grouped

    def copy(self) -> "ImageBase":
        copied = deepcopy(self)
        for key, value in copied.__dict__.items():
            try:
                setattr(copied, key, value.copy())
            except TypeError:
                continue
            except AttributeError:
                setattr(copied, key, deepcopy(value))
        return copied


class Band(ImageBase):
    cmap: ClassVar[str | None] = None

    def __init__(
        self,
        data: str | np.ndarray,
        res: int | None,
        crs: Any | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        cmap: str | None = None,
        name: str | None = None,
        file_system: dp.gcs.GCSFileSystem | None = None,
        _mask: GeoDataFrame | GeoSeries | Geometry | tuple[float] | None = None,
    ) -> None:
        if isinstance(data, np.ndarray):
            self._values = data
            if bounds is None:
                raise ValueError("Must specify bounds when data is an array.")
            if crs is None:
                raise ValueError("Must specify crs when data is an array.")
            self._bounds = to_bbox(bounds)
            self._crs = crs
            self.transform = _get_transform_from_bounds(
                self.bounds, shape=self.values.shape
            )
        elif not isinstance(data, (str | Path | os.PathLike)):
            raise TypeError("'data' must be string, Path-like or numpy.ndarray.")
        else:
            self.path = str(data)
        self._res = res
        if cmap is not None:
            self.cmap = cmap
        self.file_system = file_system
        self._mask = _mask
        self._name = name

        if self.filename_regexes:
            if isinstance(self.filename_regexes, str):
                self.filename_regexes = [self.filename_regexes]
            self.filename_patterns = [
                re.compile(pat, flags=re.VERBOSE) for pat in self.filename_regexes
            ]
        else:
            self.filename_patterns = None

    def __lt__(self, other: "Band") -> bool:
        """Makes Bands sortable by band_id."""
        return self.band_id < other.band_id

    @property
    def values(self) -> np.ndarray:
        try:
            return self._values
        except AttributeError as e:
            raise AttributeError("array is not loaded.") from e

    @values.setter
    def values(self, new_val):
        if isinstance(new_val, np.ndarray):
            raise TypeError(f"{self.__class__.__name__} 'values' must be np.ndarray.")
        self._values = new_val

    @property
    def res(self) -> int:
        return self._res

    @property
    def tile(self) -> str:
        return self._name_regex_searcher("tile", self.filename_patterns)

    @property
    def date(self) -> str:
        return self._name_regex_searcher("date", self.filename_patterns)

    @property
    def band_id(self) -> str:
        return self._name_regex_searcher("band", self.filename_patterns)

    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name
        return Path(self.path).name

    @name.setter
    def name(self, value) -> None:
        self._name = value

    @property
    def stem(self) -> str:
        return Path(self.path).stem

    @property
    def crs(self) -> str | None:
        try:
            return self._crs
        except AttributeError:
            with opener(self.path, file_system=self.file_system) as file:
                with rasterio.open(file) as src:
                    self._bounds = to_bbox(src.bounds)
                    self._crs = src.crs
                    return self._crs

    @property
    def bounds(self) -> tuple[int, int, int, int] | None:
        try:
            return tuple(int(x) for x in self._bounds)
        except AttributeError:
            with opener(self.path, file_system=self.file_system) as file:
                with rasterio.open(file) as src:
                    self._bounds = to_bbox(src.bounds)
                    self._crs = src.crs
                    return self._bounds
        except TypeError:
            return None

    @property
    def height(self) -> int:
        i = 1 if len(self.values.shape) == 3 else 0
        return self.values.shape[i]

    @property
    def width(self) -> int:
        i = 2 if len(self.values.shape) == 3 else 1
        return self.values.shape[i]

    def get_n_largest(
        self, n: int, precision: float = 0.000001, column: str = "value"
    ) -> GeoDataFrame:
        copied = self.copy()
        value_must_be_at_least = np.sort(np.ravel(copied.values))[-n] - (precision or 0)
        copied._values = np.where(copied.values >= value_must_be_at_least, 1, 0)
        df = copied.to_gdf(column).loc[lambda x: x[column] == 1]
        df[column] = f"largest_{n}"
        return df

    def get_n_smallest(
        self, n: int, precision: float = 0.000001, column: str = "value"
    ) -> GeoDataFrame:
        copied = self.copy()
        value_must_be_at_least = np.sort(np.ravel(copied.values))[n] - (precision or 0)
        copied._values = np.where(copied.values <= value_must_be_at_least, 1, 0)
        df = copied.to_gdf(column).loc[lambda x: x[column] == 1]
        df[column] = f"smallest_{n}"
        return df

    def load(self, bounds=None, indexes=None, **kwargs) -> "Band":
        if not hasattr(self, "path"):
            raise ValueError("Can only load array from Band constructed from filepath.")
        bounds = to_bbox(bounds) if bounds is not None else self._mask

        with opener(self.path, file_system=self.file_system) as f:
            with rasterio.open(f) as src:
                self._res = src.res if not self.res else self.res
                # if bounds is None:
                #     out_shape = _get_shape_from_res(to_bbox(src.bounds), self.res, indexes)
                #     self.transform = src.transform
                #     arr = src.load(indexes=indexes, out_shape=out_shape, **kwargs)
                #     # if isinstance(indexes, int) and len(arr.shape) == 3:
                #     #     return arr[0]
                #     return arr
                # else:
                #     window = rasterio.windows.from_bounds(
                #         *bounds, transform=src.transform
                #     )
                #     out_shape = _get_shape_from_bounds(bounds, self.res)

                #     arr = src.read(
                #         indexes=indexes,
                #         out_shape=out_shape,
                #         window=window,
                #         boundless=boundless,
                #         **kwargs,
                #     )
                #     if isinstance(indexes, int):
                #         # arr = arr[0]
                #         height, width = arr.shape
                #     else:
                #         height, width = arr.shape[1:]

                #     self.transform = rasterio.transform.from_bounds(
                #         *bounds, width, height
                #     )
                #     if bounds is not None:
                #         self._bounds = bounds
                #     return arr

                if indexes is None and len(src.indexes) == 1:
                    indexes = 1

                if isinstance(indexes, int):
                    _indexes = (indexes,)
                else:
                    _indexes = indexes

                arr, transform = rasterio.merge.merge(
                    [src],
                    res=self.res,
                    indexes=_indexes,
                    bounds=bounds,
                    **kwargs,
                )
                self.transform = transform
                if bounds is not None:
                    self._bounds = bounds

                if isinstance(indexes, int):
                    arr = arr[0]

                self._values = arr
                return self

    def write(self, path: str | Path, **kwargs) -> None:
        if not hasattr(self, "_values"):
            raise ValueError(
                "Can only write image band from Band constructed from array."
            )

        profile = {
            # "driver": self.driver,
            # "compress": self.compress,
            # "dtype": self.dtype,
            "crs": self.crs,
            "transform": self.transform,
            # "nodata": self.nodata,
            # "count": self.count,
            # "height": self.height,
            # "width": self.width,
            # "indexes": self.indexes,
        } | kwargs

        with opener(path, "w", file_system=self.file_system) as f:
            with rasterio.open(f, **profile) as dst:
                # bounds = to_bbox(self._mask) if self._mask is not None else dst.bounds

                # res = dst.res if not self.res else self.res

                if len(self.values.shape) == 2:
                    return dst.write(self.values, indexes=1)

                for i in range(self.values.shape[0]):
                    dst.write(self.values[i], indexes=i + 1)

        self.path = path

    def to_gdf(self, column: str = "value") -> GeoDataFrame:
        """Create a GeoDataFrame from the image Band.

        Args:
            column: Name of resulting column that holds the raster values.

        Returns:
            A GeoDataFrame with a geometry column and array values.
        """
        if not hasattr(self, "_values"):
            raise ValueError(
                "Can only write image band from Band constructed from array."
            )

        return GeoDataFrame(
            pd.DataFrame(
                _array_to_geojson(self.values, self.transform),
                columns=[column, "geometry"],
            ),
            geometry="geometry",
            crs=self.crs,
        )

    def to_xarray(self) -> DataArray:
        """Convert the raster to  an xarray.DataArray."""
        name = self.name or self.__class__.__name__.lower()
        coords = _generate_spatial_coords(self.transform, self.width, self.height)
        if len(self.values.shape) == 2:
            dims = ["y", "x"]
        elif len(self.values.shape) == 3:
            dims = ["band", "y", "x"]
        else:
            raise ValueError("Array must be 2 or 3 dimensional.")
        return xr.DataArray(
            self.values,
            coords=coords,
            dims=dims,
            name=name,
            attrs={"crs": self.crs},
        )  # .transpose("y", "x")

    def __repr__(self) -> str:
        try:
            band_id = f"'{self.band_id}'"
        except (ValueError, AttributeError):
            band_id = None
        try:
            path = f"'{self.path}'"
        except AttributeError:
            path = None
        return (
            f"{self.__class__.__name__}(band_id={band_id}, res={self.res}, path={path})"
        )


class NDVIBand(Band):
    cmap: str = "Greens"


class Image(ImageBase):
    cloud_cover_regexes: ClassVar[tuple[str] | None] = None
    band_class: ClassVar[Band] = Band

    def __init__(
        self,
        data: str | Path | Sequence[Band],
        res: int | None = None,
        # crs: Any | None = None,
        file_system: dp.gcs.GCSFileSystem | None = None,
        df: pd.DataFrame | None = None,
        all_file_paths: list[str] | None = None,
        _mask: GeoDataFrame | GeoSeries | Geometry | tuple | None = None,
    ) -> None:
        super().__init__()

        self.res = res
        # self._crs = crs
        self.file_system = file_system
        self._mask = _mask

        if hasattr(data, "__iter__") and all(isinstance(x, Band) for x in data):
            self._bands = list(data)
            self._bounds = get_total_bounds(self._bands)
            self._crs = get_common_crs(self._bands)
            self._df = pd.DataFrame(
                {
                    "file_path": [None for _ in self._bands],
                    f"band{FILENAME_COL_SUFFIX}": [
                        band.band_id for band in self._bands
                    ],
                    "tile": [band.tile for band in self._bands],
                }
            )
            return

        if not isinstance(data, (str | Path | os.PathLike)):
            raise TypeError("'data' must be string, Path-like or a sequence of Band.")

        self.path = str(data)

        if df is None:
            if is_dapla():
                file_paths = list(sorted(set(glob_func(self.path + "/**"))))
            else:
                file_paths = list(
                    sorted(
                        set(
                            glob_func(self.path + "/**/**")
                            + glob_func(self.path + "/**/**/**")
                            + glob_func(self.path + "/**/**/**/**")
                            + glob_func(self.path + "/**/**/**/**/**")
                        )
                    )
                )
            self._create_metadata_df(file_paths)
        else:
            self._df = df

        self._df["image_path"] = self._df["image_path"].astype(str)

        cols_to_explode = [
            "file_path",
            *[x for x in self._df if FILENAME_COL_SUFFIX in x],
        ]
        try:
            self._df = self._df.explode(cols_to_explode, ignore_index=True)
        except ValueError:
            for col in cols_to_explode:
                self._df = self._df.explode(col)
            self._df = self._df.loc[lambda x: ~x["filename"].duplicated()].reset_index(
                drop=True
            )

        self._df = self._df.loc[
            lambda x: x["image_path"].str.contains(_fix_path(self.path))
        ]

        if self.filename_patterns and any(pat.groups for pat in self.filename_patterns):
            self._df = self._df.loc[
                lambda x: (x[f"band{FILENAME_COL_SUFFIX}"].notna())
            ].sort_values(f"band{FILENAME_COL_SUFFIX}")

        if self.cloud_cover_regexes:
            if all_file_paths is None:
                file_paths = ls_func(self.path)
            else:
                file_paths = [path for path in all_file_paths if self.name in path]
            self.cloud_cover_percentage = float(
                _get_regex_match_from_xml_in_local_dir(
                    file_paths, regexes=self.cloud_cover_regexes
                )
            )
        else:
            self.cloud_cover_percentage = None

        self._bands = [
            self.band_class(
                path,
                res=res,
                file_system=self.file_system,
                _mask=self._mask,
            )
            for path in (self.df["file_path"])
        ]
        if self.filename_patterns and any(pat.groups for pat in self.filename_patterns):
            self._bands = list(sorted(self._bands))

    def get_ndvi(self, red_band: str, nir_band: str) -> NDVIBand:

        red = self[red_band].load().values
        nir = self[nir_band].load().values

        arr: np.ndarray = ndvi(red, nir)

        return NDVIBand(
            arr,
            res=self.res,
            bounds=self.bounds,
            crs=self.crs,
            file_system=self.file_system,
        )

    @property
    def band_ids(self) -> list[str]:
        return list(self.df[f"band{FILENAME_COL_SUFFIX}"].unique())

    @property
    def file_paths(self) -> list[str]:
        return list(self.df["file_path"])

    @property
    def bands(self):
        return self._bands

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def stem(self) -> str:
        return Path(self.path).stem

    @property
    def level(self) -> str:
        return self._name_regex_searcher("level", self.image_patterns)

    @property
    def tile(self) -> str:
        return self._name_regex_searcher("tile", self.image_patterns)

    @property
    def date(self) -> str:
        return self._name_regex_searcher("date", self.image_patterns)

    @property
    def mint(self) -> float:
        return disambiguate_timestamp(self.date, self.date_format)[0]

    @property
    def maxt(self) -> float:
        return disambiguate_timestamp(self.date, self.date_format)[1]

    @property
    def df(self):
        return self._df

    def read(self, bounds=None, **kwargs) -> np.ndarray:
        """Return 3 dimensional numpy.ndarray of shape (n bands, width, height)."""
        return np.array(
            [
                (
                    band.load(bounds=bounds, **kwargs).values
                    if hasattr(band, "path")
                    else band.values
                )
                for band in self.bands
            ]
        )

    def write(self, array: np.ndarray, path: str | Path, **kwargs) -> None:
        for band in self.bands:
            return band.write(array, path, **kwargs)
            # if hasattr(band, "transform") and band.transform is not None:
        raise ValueError("No bands...")

    def sample(
        self, n: int = 1, size: int = 1000, mask: Any = None, **kwargs
    ) -> "Image":
        """Take a random spatial sample of the image."""
        if mask is not None:
            points = GeoSeries([self.unary_union]).clip(mask).sample_points(n)
        else:
            points = GeoSeries([self.unary_union]).sample_points(n)
        buffered = points.buffer(size / 2).clip(self.unary_union)
        boxes = to_gdf([box(*arr) for arr in buffered.bounds.values], crs=self.crs)
        return self.read(bbox=boxes, **kwargs)

    def get_path(self, band: str) -> str:
        simple_string_match = [path for path in self.file_paths if str(band) in path]
        if len(simple_string_match) == 1:
            return simple_string_match[0]

        regexes_matches = []
        for path in self.file_paths:
            for pat in self.filename_patterns:
                match_ = re.search(pat, Path(path).name)
                if match_ and str(band) == match_.group("band"):
                    regexes_matches.append(path)

        if len(regexes_matches) == 1:
            return regexes_matches[0]

        if len(regexes_matches) > 1:
            prefix = "Multiple"
        elif not regexes_matches:
            prefix = "No"

        raise KeyError(
            f"{prefix} matches for band {band} among paths {[Path(x).name for x in self.file_paths]}"
        )

    def __getitem__(self, band: str | Sequence[str]) -> "Band | Image":
        if isinstance(band, str):
            return self._get_band(band)
        copied = self.copy()
        try:
            copied._bands = [copied._get_band(x) for x in band]
        except TypeError as e:
            raise TypeError(
                f"{self.__class__.__name__} indices should be string or list of string. "
                f"Got {band}"
            ) from e
        return copied

    def __contains__(self, item: str | Sequence[str]) -> bool:
        if isinstance(item, str):
            return item in self.band_ids
        return all(x in self.band_ids for x in item)

    def __lt__(self, other: "Image") -> bool:
        """Makes Images sortable by date."""
        return self.date < other.date

    def __iter__(self):
        return iter(self.bands)

    def __len__(self) -> int:
        return len(self.bands)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bands={self.bands})"

    def get_cloud_band(self) -> Band:
        scl = self[self.cloud_band].load()
        scl._values = np.where(np.isin(scl.values, self.cloud_values), 1, 0)
        return scl

    def intersects(self, other: GeoDataFrame | GeoSeries | Geometry) -> bool:
        if hasattr(other, "crs") and not pyproj.CRS(self.crs).equals(
            pyproj.CRS(other.crs)
        ):
            raise ValueError(f"crs mismatch: {self.crs} and {other.crs}")
        return self.unary_union(to_shapely(other))

    @property
    def crs(self) -> str | None:
        try:
            return self._crs
        except AttributeError:
            if not len(self):
                return None
            with opener(self.file_paths[0], file_system=self.file_system) as file:
                with rasterio.open(file) as src:
                    self._bounds = to_bbox(src.bounds)
                    self._crs = src.crs
                    return self._crs

    @property
    def bounds(self) -> tuple[int, int, int, int] | None:
        try:
            return tuple(int(x) for x in self._bounds)
        except AttributeError:
            if not len(self):
                return None
            with opener(self.file_paths[0], file_system=self.file_system) as file:
                with rasterio.open(file) as src:
                    self._bounds = to_bbox(src.bounds)
                    self._crs = src.crs
                    return self._bounds
        except TypeError:
            return None

    # @property
    # def transform(self) -> Affine | None:
    #     """Get the Affine transform of the image."""
    #     try:
    #         return rasterio.transform.from_bounds(*self.bounds, self.width, self.height)
    #     except (ZeroDivisionError, TypeError):
    #         if not self.width or not self.height:
    #             return None

    # @property
    # def shape(self) -> tuple[int]:
    #     return self._shape

    # @property
    # def height(self) -> int:
    #     i = 1 if len(self.shape) == 3 else 0
    #     return self.shape[i]

    # @property
    # def width(self) -> int:
    #     i = 2 if len(self.shape) == 3 else 1
    #     return self.shape[i]

    # @transform.setter
    # def transform(self, value: Affine) -> None:
    #     self._bounds = rasterio.transform.array_bounds(self.height, self.width, value)

    @property
    def centroid(self) -> str:
        x = (self.bounds[0] + self.bounds[2]) / 2
        y = (self.bounds[1] + self.bounds[3]) / 2
        return Point(x, y)

    @property
    def unary_union(self) -> Polygon:
        return box(*self.bounds)

    @property
    def bbox(self) -> BoundingBox:
        bounds = GeoSeries([self.unary_union]).bounds
        return BoundingBox(
            minx=bounds.minx[0],
            miny=bounds.miny[0],
            maxx=bounds.maxx[0],
            maxy=bounds.maxy[0],
            mint=self.mint,
            maxt=self.maxt,
        )

    def _get_band(self, band: str) -> Band:
        if not isinstance(band, str):
            raise TypeError(f"band must be string. Got {type(band)}")

        bands = [x for x in self.bands if x.band_id == band]
        if len(bands) == 1:
            return bands[0]

        more_bands = [x for x in self.bands if x.path == band]
        if len(more_bands) == 1:
            return more_bands[0]

        if len(bands) > 1:
            prefix = "Multiple"
        elif not bands:
            prefix = "No"

        raise KeyError(
            f"{prefix} matches for band {band} among paths {[Path(band.path).name for band in self.bands]}"
        )


def src_load(src, indexes, out_shape, **kwargs):
    return src.load(indexes=indexes, out_shape=out_shape, **kwargs)


class ImageCollection(ImageBase):
    image_class: ClassVar[Image] = Image
    band_class: ClassVar[Band] = Band

    def __init__(
        self,
        data: str | Path | Sequence[Image],
        res: int,
        level: str | None,
        processes: int = 1,
        file_system: dp.gcs.GCSFileSystem | None = None,
        df: pd.DataFrame | None = None,
    ) -> None:
        super().__init__()

        self.level = level
        self.processes = processes
        self.file_system = file_system
        self.res = res
        self._mask = None
        self._band_ids = None

        if hasattr(data, "__iter__") and all(isinstance(x, Image) for x in data):
            self.path = None
            self._df = pd.concat(
                [x._df for x in data], ignore_index=True
            ).drop_duplicates()
            self.images = data
            return

        if not isinstance(data, (str | Path | os.PathLike)):
            raise TypeError("'data' must be string, Path-like or a sequence of Image.")

        self.path = str(data)

        if is_dapla():
            self._all_filepaths = list(sorted(set(glob_func(self.path + "/**"))))
        else:
            self._all_filepaths = list(
                sorted(
                    set(
                        glob_func(self.path + "/**/**")
                        + glob_func(self.path + "/**/**/**")
                        + glob_func(self.path + "/**/**/**/**")
                        + glob_func(self.path + "/**/**/**/**/**")
                    )
                )
            )

        if self.level:
            self._all_filepaths = [
                path for path in self._all_filepaths if self.level in path
            ]

        if df is not None:
            self._df = df
        else:
            self._create_metadata_df(self._all_filepaths)

    def groupby(self, by, **kwargs) -> list[tuple[Any], "ImageCollection"]:
        if isinstance(by, str):
            by = (by,)

        if "bounds" in by:
            self._df["bounds"] = [img.bounds for img in self]

        by = [col if col in self.df else f"{col}{FILENAME_COL_SUFFIX}" for col in by]

        df_long = self.df.explode(["file_path", *self._match_cols_filename])
        with joblib.Parallel(n_jobs=self.processes, backend="loky") as parallel:
            return list(
                sorted(
                    parallel(
                        joblib.delayed(_copy_and_add_df_parallel)(i, group, self)
                        for i, group in df_long.groupby(by, **kwargs)
                    )
                )
            )

    def _merge_with_numpy_func(
        self, method: str | Callable, bounds=None, indexes=None, **kwargs
    ) -> np.ndarray:
        arrs = []
        numpy_func = get_numpy_func(method) if not callable(method) else method
        for (_bounds,), collection in self.groupby("bounds"):
            arr = np.array(
                [
                    band.load(indexes=indexes).values
                    for img in collection
                    for band in img
                ]
            )
            arr = numpy_func(arr, axis=0)
            if len(arr.shape) == 2:
                height, width = arr.shape
            else:
                height, width = arr.shape[1:]

            transform = rasterio.transform.from_bounds(*_bounds, width, height)
            coords = _generate_spatial_coords(transform, width, height)

            arrs.append(
                xr.DataArray(
                    arr,
                    coords=coords,
                    dims=["y", "x"],
                    name=str(_bounds),
                    attrs={"crs": self.crs},
                )
            )

        if bounds is None:
            bounds = self.bounds

        merged = merge_arrays(arrs, bounds=bounds, res=self.res, **kwargs)

        return merged.to_numpy()

    def merge(self, method="median", bounds=None, indexes=None, **kwargs) -> Band:
        bounds = to_bbox(bounds) if bounds is not None else self._mask
        crs = self.crs

        if indexes is None:
            indexes = 1

        if isinstance(indexes, int):
            _indexes = (indexes,)
        else:
            _indexes = indexes

        if method == "mean":
            _method = "sum"
        else:
            _method = method

        if method not in list(rasterio.merge.MERGE_METHODS) + ["mean"]:
            arr = self._merge_with_numpy_func(
                method=method,
                bounds=bounds,
            )
        else:
            datasets = [_open_raster(path) for path in self.file_paths]
            arr, _ = rasterio.merge.merge(
                datasets,
                res=self.res,
                bounds=bounds,
                indexes=_indexes,
                method=_method,
                **kwargs,
            )

        if isinstance(indexes, int) and len(arr.shape) == 3 and arr.shape[0] == 1:
            arr = arr[0]

        if method == "mean":
            arr = arr / len(datasets)

        if bounds is None:
            bounds = self.bounds

        return self.band_class(
            arr, res=self.res, bounds=bounds, crs=crs, file_system=self.file_system
        )

    def merge_by_band(
        self, method="median", bounds=None, indexes=None, **kwargs
    ) -> Band:
        bounds = to_bbox(bounds) if bounds is not None else self._mask
        crs = self.crs

        if indexes is None:
            indexes = 1

        if isinstance(indexes, int):
            _indexes = (indexes,)
        else:
            _indexes = indexes

        if method == "mean":
            _method = "sum"
        else:
            _method = method

        arrs = []
        for (band_id,), band_collection in self.groupby("band"):
            if method not in list(rasterio.merge.MERGE_METHODS) + ["mean"]:
                arr = band_collection._merge_with_numpy_func(
                    method=method,
                    bounds=bounds,
                )
                arrs.append(arr)
                continue

            datasets = [_open_raster(path) for path in band_collection.file_paths]
            arr, _ = rasterio.merge.merge(
                datasets,
                res=self.res,
                bounds=bounds,
                indexes=_indexes,
                method=_method,
                **kwargs,
            )
            if isinstance(indexes, int):
                arr = arr[0]
            arrs.append(arr)

        arr = np.array(arrs)

        if method == "mean":
            arr = arr / len(datasets)

        if bounds is None:
            bounds = self.bounds

        return self.band_class(
            arr, res=self.res, bounds=bounds, crs=crs, file_system=self.file_system
        )

    def set_mask(
        self, mask: GeoDataFrame | GeoSeries | Geometry | tuple[float]
    ) -> "ImageCollection":
        """Set the mask to be used to clip the images to."""
        self._mask = to_shapely(mask)
        return self

    def filter(
        self,
        bands: str | list[str] | None = None,
        date_ranges: (
            tuple[str | None, str | None]
            | tuple[tuple[str | None, str | None], ...]
            | None
        ) = None,
        bbox: GeoDataFrame | GeoSeries | Geometry | tuple[float] | None = None,
        max_cloud_cover: int | None = None,
        copy: bool = True,
    ) -> "ImageCollection":
        copied = self.copy() if copy else self

        if isinstance(bbox, BoundingBox):
            date_ranges = (bbox.mint, bbox.maxt)

        if date_ranges:
            copied = copied._filter_dates(date_ranges, bbox, copy=False)

        if max_cloud_cover is not None:
            copied.images = [
                image
                for image in copied.images
                if image.cloud_cover_percentage < max_cloud_cover
            ]

        if bbox is not None:
            copied = copied._filter_bounds(bbox, copy=False)

        if bands is not None:
            if isinstance(bands, str):
                bands = [bands]
            bands = set(bands)
            copied._band_ids = bands
            copied.images = [img[bands] for img in copied.images if bands in img]

        return copied

    def _filter_dates(
        self,
        date_ranges: (
            tuple[str | None, str | None] | tuple[tuple[str | None, str | None], ...]
        ),
        bbox: BoundingBox | None = None,
        copy: bool = True,
    ) -> "ImageCollection":
        if not isinstance(date_ranges, (tuple, list)):
            raise TypeError(
                "date_ranges should be a 2-length tuple of strings or None, "
                "or a tuple of tuples for multiple date ranges"
            )
        if self.image_patterns is None:
            raise ValueError(
                "Cannot set date_ranges when the class's image_regexes attribute is None"
            )

        copied = self.copy() if copy else self

        copied.df = copied.df.loc[
            lambda x: x["image_path"].apply(
                lambda y: _date_is_within(
                    y, date_ranges, copied.image_patterns, copied.date_format
                )
            )
        ]

        return copied

    def _filter_bounds(
        self, other: GeoDataFrame | GeoSeries | Geometry | tuple, copy: bool = True
    ) -> "ImageCollection":
        copied = self.copy() if copy else self

        other = to_shapely(other)

        with joblib.Parallel(n_jobs=copied.processes, backend="threading") as parallel:
            intersects_list: list[bool] = parallel(
                joblib.delayed(_intesects)(image, other) for image in copied
            )
        copied.images = [
            image
            for image, intersects in zip(copied, intersects_list, strict=False)
            if intersects
        ]
        return copied

    def sample_tiles(self, n: int) -> "ImageCollection":
        copied = self.copy()
        sampled_tiles = set(self.df["tile"].drop_duplicates().sample(n))

        copied.images = [image for image in self if image.tile in sampled_tiles]
        return copied

    def sample_images(self, n: int) -> "ImageCollection":
        copied = self.copy()
        images = copied.images
        if n > len(images):
            raise ValueError(
                f"n ({n}) is higher than number of images in collection ({len(images)})"
            )
        sample = []
        for _ in range(n):
            random.shuffle(images)
            tile = images.pop()
            sample.append(tile)

        copied.images = images

    def __iter__(self):
        return iter(self.images)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(
        self, item: int | slice | Sequence[int] | BoundingBox | Sequence[BoundingBox]
    ) -> Image | TORCHGEO_RETURN_TYPE:
        if isinstance(item, int):
            return self.images[item]

        if isinstance(item, slice):
            copied = self.copy()
            copied.images = copied.images[item]
            return copied

        if not isinstance(item, BoundingBox) and not (
            isinstance(item, Iterable) and all(isinstance(x, BoundingBox) for x in item)
        ):
            try:
                copied = self.copy()
                copied.images = [copied.images[i] for i in item]
                return copied
            except Exception:
                if hasattr(item, "__iter__"):
                    endnote = f" of length {len(item)} with types {set(type(x) for x in item)}"
                raise TypeError(
                    "ImageCollection indices must be int or BoundingBox. "
                    f"Got {type(item)}{endnote}"
                )

        elif isinstance(item, BoundingBox):
            date_ranges: tuple[str] = (item.mint, item.maxt)
            data: torch.Tensor = numpy_to_torch(
                self.filter(bbox=item, date_ranges=date_ranges)
                .merge_by_band(bounds=item)
                .values
            )
        else:
            bboxes: list[Polygon] = [to_bbox(x) for x in item]
            date_ranges: list[list[str, str]] = [(x.mint, x.maxt) for x in item]
            data: torch.Tensor = torch.cat(
                [
                    numpy_to_torch(
                        self.filter(bbox=bbox, date_ranges=date_range)
                        .merge_by_band(bounds=bbox)
                        .values
                    )
                    for bbox, date_range in zip(bboxes, date_ranges, strict=True)
                ]
            )

        crs = get_common_crs(self.images)

        key = "image"  # if self.is_image else "mask"
        sample = {key: data, "crs": crs, "bbox": item}

        return sample

    def _copy_and_add_df(self, new_df: pd.DataFrame) -> "ImageCollection":
        copied = self.copy()
        grouped = (
            new_df.drop(columns=self._match_cols_filename)
            .drop_duplicates("image_path")
            .set_index("image_path")
        )
        for col in ["file_path", *self._match_cols_filename]:
            assert isinstance(new_df[col].iloc[0], str), new_df[col]
            grouped[col] = new_df.groupby("image_path")[col].apply(tuple)

        grouped = grouped.reset_index()

        copied.df = grouped
        return copied

    # def dates_as_float(self) -> list[tuple[float, float]]:
    #     return [disambiguate_timestamp(date, self.date_format) for date in self.dates]

    @property
    def mint(self) -> float:
        return min(img.mint for img in self)

    @property
    def maxt(self) -> float:
        return max(img.maxt for img in self)

    @property
    def band_ids(self) -> list[str]:
        return list(self.df[f"band{FILENAME_COL_SUFFIX}"].explode().unique())

    @property
    def file_paths(self) -> list[str]:
        return list(self.df["file_path"].explode())

    @property
    def dates(self) -> list[str]:
        return list(self.df["date"])

    def dates_as_int(self) -> list[int]:
        return [int(date[:8]) for date in self.dates]

    @property
    def image_paths(self) -> list[str]:
        return list(self.df["image_path"])

    @property
    def images(self) -> list["Image"]:
        try:
            return self._images
        except AttributeError:
            # only fetch images when they are needed
            self._images = _get_images(
                self.image_paths,
                all_file_paths=self._all_filepaths,
                df=self.df,
                res=self.res,
                processes=self.processes,
                image_class=self.image_class,
                _mask=self._mask,
            )
            if self.image_regexes:
                self._images = list(sorted(self._images))
            return self._images

    @images.setter
    def images(self, new_value: list["Image"]) -> list["Image"]:
        if self.filename_patterns and any(pat.groups for pat in self.filename_patterns):
            self._images = list(sorted(new_value))
        if not all(isinstance(x, Image) for x in self._images):
            raise TypeError("images should be a sequence of Image.")
        self._df = self._df.loc[
            lambda x: x["image_path"].isin({x.path for x in self._images})
        ]
        if self._band_ids is not None:
            are_matching = self._df[f"band{FILENAME_COL_SUFFIX}"].apply(
                lambda x: [
                    True if band_id in self._band_ids else False for band_id in x
                ]
            )
            if all(x is True for matches in are_matching for x in matches):
                return

            for col in ["file_path", *self._match_cols_filename]:
                # keeping only list elements that match the band_id in the boolean mask are_matching
                self._df[col] = [
                    tuple(
                        value
                        for value, matches in zip(row_list, match_row, strict=True)
                        if matches
                    )
                    for row_list, match_row in zip(
                        self._df[col], are_matching, strict=True
                    )
                ]

    @property
    def index(self) -> Index:
        try:
            if len(self) == len(self._index):
                return self._index
        except AttributeError:
            self._index = Index(interleaved=False, properties=Property(dimension=3))

            for i, img in enumerate(self.images):
                if img.date:
                    try:
                        mint, maxt = disambiguate_timestamp(img.date, self.date_format)
                    except (NameError, TypeError):
                        mint, maxt = 0, 1
                else:
                    mint, maxt = 0, 1
                # important: torchgeo has a different order of the bbox than shapely and geopandas
                minx, miny, maxx, maxy = img.bounds
                self._index.insert(i, (minx, maxx, miny, maxy, mint, maxt))
            return self._index

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, new_value: pd.DataFrame) -> None:
        if not isinstance(new_value, pd.DataFrame):
            raise TypeError("df should be a pandas.DataFrame.")
        self._df = new_value
        new_image_paths = set(self._df["image_path"])
        self._images = [image for image in self.images if image.path in new_image_paths]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)})"

    @property
    def centroid(self) -> Point:
        return self.unary_union.centroid

    @property
    def unary_union(self) -> Polygon | MultiPolygon:
        return unary_union([img.unary_union for img in self])

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return tuple(int(x) for x in get_total_bounds([img.bounds for img in self]))

    @property
    def crs(self) -> Any:
        return get_common_crs([img.crs for img in self])


def concat_image_collections(collections: Sequence[ImageCollection]) -> ImageCollection:
    resolutions = {x.res for x in collections}
    if len(resolutions) > 1:
        raise ValueError(f"resoultion mismatch. {resolutions}")
    images = list(itertools.chain.from_iterable([x.images for x in collections]))
    levels = {x.level for x in collections}
    level = list(levels)[0] if len(levels) == 1 else None
    first_collection = collections[0]

    out_collection = first_collection.__class__(
        images,
        res=list(resolutions)[0],
        level=level,
        processes=first_collection.processes,
        file_system=first_collection.file_system,
    )
    out_collection._all_filepaths = list(
        sorted(
            set(itertools.chain.from_iterable([x._all_filepaths for x in collections]))
        )
    )
    return out_collection


def _get_images(
    image_paths: list[str],
    *,
    res: int,
    all_file_paths: list[str],
    df: pd.DataFrame,
    processes: int,
    image_class: Image,
    _mask: GeoDataFrame | GeoSeries | Geometry | tuple[float] | None,
) -> list[Image]:
    with joblib.Parallel(n_jobs=processes, backend="threading") as parallel:
        return parallel(
            joblib.delayed(image_class)(
                path,
                df=df,
                res=res,
                all_file_paths=all_file_paths,
                _mask=_mask,
            )
            for path in image_paths
        )


def numpy_to_torch(array: np.ndarray) -> torch.Tensor:
    """Convert numpy array to a pytorch tensor."""
    # fix numpy dtypes which are not supported by pytorch tensors
    if array.dtype == np.uint16:
        array = array.astype(np.int32)
    elif array.dtype == np.uint32:
        array = array.astype(np.int64)

    return torch.tensor(array)


def _get_regex_match_from_xml_in_local_dir(
    paths: list[str], regexes: str | tuple[str]
) -> str | dict[str, str]:
    for i, path in enumerate(paths):
        if ".xml" not in path:
            continue
        with open_func(path, "rb") as file:
            filebytes: bytes = file.read()
            try:
                return _get_cloud_percentage(filebytes.decode("utf-8"), regexes)
            except _RegexError as e:
                if i == len(paths) - 1:
                    raise e


class _RegexError(ValueError):
    pass


def _get_cloud_percentage(xml_file: str, regexes: tuple[str]) -> str | dict[str, str]:
    for regexes in regexes:
        if isinstance(regexes, dict):
            out = {}
            for key, value in regexes.items():
                try:
                    out[key] = re.search(value, xml_file).group(1)
                except (TypeError, AttributeError):
                    continue
            if len(out) != len(regexes):
                raise _RegexError()
            return out
        try:
            return re.search(regexes, xml_file).group(1)
        except (TypeError, AttributeError):
            continue
    raise _RegexError()


def _fix_path(path: str) -> str:
    return (
        str(path).replace("\\", "/").replace(r"\"", "/").replace("//", "/").rstrip("/")
    )


def _get_regexes_matches_for_df(
    df, match_col: str, patterns: Sequence[re.Pattern], suffix: str = ""
) -> tuple[pd.DataFrame, list[str]]:
    if not len(df):
        return df, []
    assert df.index.is_unique
    matches: list[pd.DataFrame] = []
    for pat in patterns:
        if pat.groups:
            try:
                matches.append(df[match_col].str.extract(pat))
            except ValueError:
                continue
        matches.append(df[match_col].loc[df[match_col].str.match(pat)])

    matches = pd.concat(matches).groupby(level=0, dropna=True).first()

    if isinstance(matches, pd.Series):
        matches = pd.DataFrame({matches.name: matches.values}, index=matches.index)

    match_cols = [f"{col}{suffix}" for col in matches.columns]
    df[match_cols] = matches
    return df.loc[~df[match_cols].isna().all(axis=1)], match_cols


def _date_is_within(
    path,
    date_ranges: (
        tuple[str | None, str | None] | tuple[tuple[str | None, str | None], ...] | None
    ),
    image_patterns: Sequence[re.Pattern],
    date_format: str,
) -> bool:
    for pat in image_patterns:
        try:
            date = re.match(pat, Path(path).name).group("date")
            break
        except AttributeError:
            date = None

    if date is None:
        return False

    if date_ranges is None:
        return True

    if all(x is None or isinstance(x, (str, float)) for x in date_ranges):
        date_ranges = (date_ranges,)

    if all(isinstance(x, float) for date_range in date_ranges for x in date_range):
        date = disambiguate_timestamp(date, date_format)
    else:
        date = date[:8]

    for date_range in date_ranges:
        date_min, date_max = date_range

        if isinstance(date_min, float) and isinstance(date_max, float):
            if date[0] >= date_min + 0.0000001 and date[1] <= date_max - 0.0000001:
                return True
            continue

        try:
            date_min = date_min or "00000000"
            date_max = date_max or "99999999"
            assert isinstance(date_min, str)
            assert len(date_min) == 8
            assert isinstance(date_max, str)
            assert len(date_max) == 8
        except AssertionError:
            raise TypeError(
                "date_ranges should be a tuple of two 8-charactered strings (start and end date)."
                f"Got {date_range} of type {[type(x) for x in date_range]}"
            )
        if date >= date_min and date <= date_max:
            return True

    return False


def _get_shape_from_bounds(
    obj: GeoDataFrame | GeoSeries | Geometry | tuple, res: int
) -> tuple[int, int]:
    resx, resy = (res, res) if isinstance(res, numbers.Number) else res

    minx, miny, maxx, maxy = to_bbox(obj)
    diffx = maxx - minx
    diffy = maxy - miny
    width = int(diffx / resx)
    heigth = int(diffy / resy)
    return heigth, width


def _get_transform_from_bounds(
    obj: GeoDataFrame | GeoSeries | Geometry | tuple, shape: tuple[float, ...]
) -> Affine:
    minx, miny, maxx, maxy = to_bbox(obj)
    if len(shape) == 2:
        width, height = shape
    elif len(shape) == 3:
        _, width, height = shape
    else:
        raise ValueError
    return rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)


def _get_shape_from_res(
    bounds: tuple[float], res: int, indexes: int | tuple[int]
) -> tuple[int] | None:
    if res is None:
        return None
    if hasattr(res, "__iter__") and len(res) == 2:
        res = res[0]
    diffx = bounds[2] - bounds[0]
    diffy = bounds[3] - bounds[1]
    width = int(diffx / res)
    height = int(diffy / res)
    if not isinstance(indexes, int):
        return len(indexes), width, height
    return width, height


def _copy_dsdsmean(merged_data, new_data, merged_mask, new_mask, index, **kwargs):
    """Returns the mean of all pixel values."""
    print(index)
    mask = np.empty_like(merged_mask, dtype="bool")
    np.logical_or(merged_mask, new_mask, out=mask)
    np.logical_not(mask, out=mask)
    np.add(merged_data, new_data, out=merged_data, where=mask, casting="unsafe")
    np.logical_not(new_mask, out=mask)
    np.logical_and(merged_mask, mask, out=mask)
    np.copyto(merged_data, new_data, where=mask, casting="unsafe")


def _copy_sadmedian(merged_data, new_data, merged_mask, new_mask, index, **kwargs):
    """Returns the median of all pixel values."""
    mask = np.empty_like(merged_mask, dtype="bool")
    np.logical_or(merged_mask, new_mask, out=mask)
    np.logical_not(mask, out=mask)
    np.add(merged_data, new_data, out=merged_data, where=mask, casting="unsafe")
    np.logical_not(new_mask, out=mask)
    np.logical_and(merged_mask, mask, out=mask)
    np.copyto(merged_data, new_data, where=mask, casting="unsafe")


def _copy_mean(
    old_data, new_data, old_nodata, new_nodata, index=None, roff=None, coff=None
):
    old_data[:] = ((old_data * index + 1) + new_data) / (index + 1)


def _copy_median(
    old_data, new_data, old_nodata, new_nodata, index=None, roff=None, coff=None
):
    old_data[:] = ((old_data * index + 1) + new_data) / (index + 1)
    # print("index")
    # print(index)
    # print(old_data)
    # print(new_data)
    # print((old_data + new_data) / (index + 2))
    # old_data[:] = np.median(old_data, new_data)


def _array_to_geojson(array: np.ndarray, transform: Affine) -> list[tuple]:
    if np.ma.is_masked(array):
        array = array.data
    try:
        return [
            (value, shape(geom))
            for geom, value in features.shapes(array, transform=transform, mask=None)
        ]
    except ValueError:
        array = array.astype(np.float32)
        return [
            (value, shape(geom))
            for geom, value in features.shapes(array, transform=transform, mask=None)
        ]


def _intesects(x, other) -> bool:
    return box(*x.bounds).intersects(other)


def _copy_and_add_df_parallel(i, group, self):
    return (i, self._copy_and_add_df(group))


def _open_raster(path: str | Path) -> rasterio.io.DatasetReader:
    with opener(path) as file:
        return rasterio.open(file)


class Sentinel2Config:
    image_regexes: ClassVar[str] = (config.SENTINEL2_IMAGE_REGEX,)
    filename_regexes: ClassVar[str] = (
        config.SENTINEL2_FILENAME_REGEX,
        config.SENTINEL2_CLOUD_FILENAME_REGEX,
    )
    rbg_bands: ClassVar[list[str]] = ["B02", "B03", "B04"]
    ndvi_bands: ClassVar[list[str]] = ["B04", "B08"]
    cloud_band: ClassVar[str] = "SCL"
    cloud_values: ClassVar[tuple[int]] = (3, 8, 9, 10, 11)
    l2a_bands: ClassVar[dict[str, int]] = config.SENTINEL2_L2A_BANDS
    l1c_bands: ClassVar[dict[str, int]] = config.SENTINEL2_L1C_BANDS
    date_format: ClassVar[str] = "%Y%m%dT%H%M%S"


class Sentinel2Band(Sentinel2Config, Band):
    """Band with Sentinel2 specific name variables and regexes."""


class Sentinel2Image(Sentinel2Config, Image):
    cloud_cover_regexes: ClassVar[tuple[str]] = config.CLOUD_COVERAGE_REGEXES
    band_class: ClassVar[Sentinel2Band] = Sentinel2Band

    def get_ndvi(
        self,
        red_band: str = Sentinel2Config.ndvi_bands[0],
        nir_band: str = Sentinel2Config.ndvi_bands[1],
    ) -> NDVIBand:
        return super().get_ndvi(red_band=red_band, nir_band=nir_band)


class Sentinel2Collection(Sentinel2Config, ImageCollection):
    """ImageCollection with Sentinel2 specific name variables and regexes."""

    image_class: ClassVar[Sentinel2Image] = Sentinel2Image
    band_class: ClassVar[Sentinel2Band] = Sentinel2Band
