import datetime
import functools
import glob
import itertools
import math
import os
import random
import re
import time
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import ClassVar

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio
from affine import Affine
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from matplotlib.colors import LinearSegmentedColormap
from rasterio.enums import MergeAlg
from scipy import stats
from scipy.ndimage import binary_dilation
from scipy.ndimage import binary_erosion
from shapely import Geometry
from shapely import box
from shapely import unary_union
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon

try:
    import dapla as dp
    from dapla.gcs import GCSFileSystem
except ImportError:

    class GCSFileSystem:
        """Placeholder."""


try:
    from google.auth import exceptions
except ImportError:

    class exceptions:
        """Placeholder."""

        class RefreshError:
            """Placeholder."""


try:
    from gcsfs.core import GCSFile
except ImportError:

    class GCSFile:
        """Placeholder."""


try:
    from rioxarray.exceptions import NoDataInBounds
    from rioxarray.merge import merge_arrays
    from rioxarray.rioxarray import _generate_spatial_coords
except ImportError:
    pass
try:
    import xarray as xr
    from xarray import DataArray
    from xarray import Dataset
except ImportError:

    class DataArray:
        """Placeholder."""

    class Dataset:
        """Placeholder."""


from ..geopandas_tools.bounds import get_total_bounds
from ..geopandas_tools.conversion import to_bbox
from ..geopandas_tools.conversion import to_gdf
from ..geopandas_tools.conversion import to_shapely
from ..geopandas_tools.general import get_common_crs
from ..helpers import get_all_files
from ..helpers import get_numpy_func
from ..io._is_dapla import is_dapla
from ..io.opener import opener
from . import sentinel_config as config
from .base import _array_to_geojson
from .base import _gdf_to_arr
from .base import _get_shape_from_bounds
from .base import _get_transform_from_bounds
from .base import get_index_mapper
from .indices import ndvi
from .regex import _any_regex_matches
from .regex import _extract_regex_match_from_string
from .regex import _get_first_group_match
from .regex import _get_non_optional_groups
from .regex import _get_regexes_matches_for_df
from .regex import _RegexError
from .zonal import _aggregate
from .zonal import _make_geometry_iterrows
from .zonal import _no_overlap_df
from .zonal import _prepare_zonal
from .zonal import _zonal_post

if is_dapla():

    def _ls_func(*args, **kwargs) -> list[str]:
        return dp.FileClient.get_gcs_file_system().ls(*args, **kwargs)

    def _glob_func(*args, **kwargs) -> list[str]:
        return dp.FileClient.get_gcs_file_system().glob(*args, **kwargs)

    def _open_func(*args, **kwargs) -> GCSFile:
        return dp.FileClient.get_gcs_file_system().open(*args, **kwargs)

    def _read_parquet_func(*args, **kwargs) -> list[str]:
        return dp.read_pandas(*args, **kwargs)

else:
    _ls_func = functools.partial(get_all_files, recursive=False)
    _open_func = open
    _glob_func = glob.glob
    _read_parquet_func = pd.read_parquet

DATE_RANGES_TYPE = (
    tuple[str | pd.Timestamp | None, str | pd.Timestamp | None]
    | tuple[tuple[str | pd.Timestamp | None, str | pd.Timestamp | None], ...]
)

FILENAME_COL_SUFFIX = "_filename"

DEFAULT_FILENAME_REGEX = r"""
    .*?
    (?:_?(?P<date>\d{8}(?:T\d{6})?))?  # Optional underscore and date group
    .*?
    (?:_?(?P<band>B\d{1,2}A|B\d{1,2}))?  # Optional underscore and band group
    \.(?:tif|tiff|jp2)$  # End with .tif, .tiff, or .jp2
"""
DEFAULT_IMAGE_REGEX = r"""
    .*?
    (?:_?(?P<date>\d{8}(?:T\d{6})?))?  # Optional underscore and date group
"""

ALLOWED_INIT_KWARGS = [
    "image_class",
    "band_class",
    "image_regexes",
    "filename_regexes",
    "bounds_regexes",
    "all_bands",
    "crs",
    "masking",
    "_merged",
    "_add_metadata_attributes",
]


class ImageCollectionGroupBy:
    """Iterator and merger class returned from groupby.

    Can be iterated through like pandas.DataFrameGroupBy.
    Or use the methods merge_by_band or merge.
    """

    def __init__(
        self,
        data: Iterable[tuple[Any], "ImageCollection"],
        by: list[str],
        collection: "ImageCollection",
    ) -> None:
        """Initialiser.

        Args:
            data: Iterable of group values and ImageCollection groups.
            by: list of group attributes.
            collection: ImageCollection instance. Used to pass attributes.
        """
        self.data = list(data)
        self.by = by
        self.collection = collection

    def merge_by_band(
        self,
        bounds: tuple | Geometry | GeoDataFrame | GeoSeries | None = None,
        method: str | Callable = "mean",
        as_int: bool = True,
        indexes: int | tuple[int] | None = None,
        **kwargs,
    ) -> "ImageCollection":
        """Merge each group into separate Bands per band_id, returned as an ImageCollection."""
        images = self._run_func_for_collection_groups(
            _merge_by_band,
            method=method,
            bounds=bounds,
            as_int=as_int,
            indexes=indexes,
            **kwargs,
        )
        for img, (group_values, _) in zip(images, self.data, strict=True):
            for attr, group_value in zip(self.by, group_values, strict=True):
                try:
                    setattr(img, attr, group_value)
                except AttributeError:
                    setattr(img, f"_{attr}", group_value)

        collection = ImageCollection(
            images,
            # TODO band_class?
            level=self.collection.level,
            **self.collection._common_init_kwargs,
        )
        collection._merged = True
        return collection

    def merge(
        self,
        bounds: tuple | Geometry | GeoDataFrame | GeoSeries | None = None,
        method: str | Callable = "mean",
        as_int: bool = True,
        indexes: int | tuple[int] | None = None,
        **kwargs,
    ) -> "Image":
        """Merge each group into a single Band, returned as combined Image."""
        bands: list[Band] = self._run_func_for_collection_groups(
            _merge,
            method=method,
            bounds=bounds,
            as_int=as_int,
            indexes=indexes,
            **kwargs,
        )
        for band, (group_values, _) in zip(bands, self.data, strict=True):
            for attr, group_value in zip(self.by, group_values, strict=True):
                try:
                    setattr(band, attr, group_value)
                except AttributeError:
                    if hasattr(band, f"_{attr}"):
                        setattr(band, f"_{attr}", group_value)

        if "band_id" in self.by:
            for band in bands:
                assert band.band_id is not None

        image = Image(
            bands,
            # TODO band_class?
            **self.collection._common_init_kwargs,
        )
        image._merged = True
        return image

    def _run_func_for_collection_groups(self, func: Callable, **kwargs) -> list[Any]:
        if self.collection.processes == 1:
            return [func(group, **kwargs) for _, group in self]
        processes = min(self.collection.processes, len(self))

        if processes == 0:
            return []

        with joblib.Parallel(n_jobs=processes, backend="threading") as parallel:
            return parallel(joblib.delayed(func)(group, **kwargs) for _, group in self)

    def __iter__(self) -> Iterator[tuple[tuple[Any, ...], "ImageCollection"]]:
        """Iterate over the group values and the ImageCollection groups themselves."""
        return iter(self.data)

    def __len__(self) -> int:
        """Number of ImageCollection groups."""
        return len(self.data)

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}({len(self)})"


def standardize_band_id(x: str) -> str:
    return x.replace("B", "").replace("A", "").zfill(2)


class BandIdDict(dict):
    """Dict that tells the band initialiser to get the dict value of the band_id."""

    def __init__(self, data: dict | None = None, **kwargs) -> None:
        """Add dicts or kwargs."""
        self._standardized_keys = {}
        for key, value in ((data or {}) | kwargs).items():
            setattr(self, key, value)
            self._standardized_keys[standardize_band_id(key)] = value

    def __len__(self) -> int:
        """Number of items."""
        return len({key for key in self.__dict__ if key != "_standardized_keys"})

    def __getitem__(self, item: str) -> Any:
        """Get dict value from key."""
        try:
            return getattr(self, item)
        except AttributeError as e:
            try:
                return self._standardized_keys[standardize_band_id(item)]
            except KeyError:
                raise KeyError(item, self.__dict__) from e


@dataclass(frozen=True)
class BandMasking:
    """Frozen dict with forced keys."""

    band_id: str
    values: tuple[int]

    def __getitem__(self, item: str) -> Any:
        """Index into attributes to mimick dict."""
        return getattr(self, item)


class NoLevel:
    """Equivelant to None."""


class _ImageBase:
    image_regexes: ClassVar[str | None] = (DEFAULT_IMAGE_REGEX,)
    filename_regexes: ClassVar[str | tuple[str]] = (DEFAULT_FILENAME_REGEX,)
    metadata_attributes: ClassVar[dict | None] = None
    masking: ClassVar[BandMasking | None] = None

    def __init__(self, *, bbox=None, **kwargs) -> None:

        self._mask = None
        self._bounds = None
        self._merged = False
        self._from_array = False
        self._from_gdf = False
        self.metadata_attributes = self.metadata_attributes or {}
        self._path = None

        self._bbox = to_bbox(bbox) if bbox is not None else None

        if self.filename_regexes:
            if isinstance(self.filename_regexes, str):
                self.filename_regexes = (self.filename_regexes,)
            self.filename_patterns = [
                re.compile(regexes, flags=re.VERBOSE)
                for regexes in self.filename_regexes
            ]
        else:
            self.filename_patterns = ()

        if self.image_regexes:
            if isinstance(self.image_regexes, str):
                self.image_regexes = (self.image_regexes,)
            self.image_patterns = [
                re.compile(regexes, flags=re.VERBOSE) for regexes in self.image_regexes
            ]
        else:
            self.image_patterns = ()

        for key, value in kwargs.items():
            if key in ALLOWED_INIT_KWARGS and key in dir(self):
                setattr(self, key, value)
            else:
                raise ValueError(
                    f"{self.__class__.__name__} got an unexpected keyword argument '{key}'"
                )

    @property
    def _common_init_kwargs(self) -> dict:
        return {
            "file_system": self.file_system,
            "processes": self.processes,
            "res": self.res,
            "bbox": self._bbox,
            "nodata": self.nodata,
        }

    @property
    def path(self) -> str:
        try:
            return self._path
        except AttributeError as e:
            raise PathlessImageError(self) from e

    @property
    def res(self) -> int:
        """Pixel resolution."""
        return self._res

    @property
    def centroid(self) -> Point:
        """Centerpoint of the object."""
        return self.union_all().centroid

    def _name_regex_searcher(
        self, group: str, patterns: tuple[re.Pattern]
    ) -> str | None:
        if not patterns or not any(pat.groups for pat in patterns):
            return None
        for pat in patterns:
            try:
                return _get_first_group_match(pat, self.name)[group]
            except (TypeError, KeyError):
                pass
        if not any(group in _get_non_optional_groups(pat) for pat in patterns):
            return None
        raise ValueError(
            f"Couldn't find group '{group}' in name {self.name} with regex patterns {patterns}"
        )

    def _create_metadata_df(self, file_paths: list[str]) -> pd.DataFrame:
        """Create a dataframe with file paths and image paths that match regexes."""
        df = pd.DataFrame({"file_path": file_paths})

        df["file_path"] = df["file_path"].apply(_fix_path)
        df["filename"] = df["file_path"].apply(lambda x: Path(x).name)

        df["image_path"] = df["file_path"].apply(
            lambda x: _fix_path(str(Path(x).parent))
        )

        if not len(df):
            return df

        df = df[~df["file_path"].isin(df["image_path"])]

        if self.filename_patterns:
            df = _get_regexes_matches_for_df(df, "filename", self.filename_patterns)

            if not len(df):
                return df

            grouped = df.drop_duplicates("image_path").set_index("image_path")
            for col in ["file_path", "filename"]:
                if col in df:
                    grouped[col] = df.groupby("image_path")[col].apply(tuple)

            grouped = grouped.reset_index()
        else:
            df["file_path"] = df.groupby("image_path")["file_path"].apply(tuple)
            df["filename"] = df.groupby("image_path")["filename"].apply(tuple)
            grouped = df.drop_duplicates("image_path")

        grouped["imagename"] = grouped["image_path"].apply(
            lambda x: _fix_path(Path(x).name)
        )

        if self.image_patterns and len(grouped):
            grouped = _get_regexes_matches_for_df(
                grouped, "imagename", self.image_patterns
            )

        return grouped

    def copy(self) -> "_ImageBase":
        """Copy the instance and its attributes."""
        copied = deepcopy(self)
        for key, value in copied.__dict__.items():
            try:
                setattr(copied, key, value.copy())
            except AttributeError:
                setattr(copied, key, deepcopy(value))
            except TypeError:
                continue
        return copied

    def equals(self, other) -> bool:
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if value != getattr(other, key):
                print(key, value, getattr(other, key))
                return False
        return True


class _ImageBandBase(_ImageBase):
    """Common parent class of Image and Band."""

    def intersects(self, other: GeoDataFrame | GeoSeries | Geometry) -> bool:
        if hasattr(other, "crs") and not pyproj.CRS(self.crs).equals(
            pyproj.CRS(other.crs)
        ):
            raise ValueError(f"crs mismatch: {self.crs} and {other.crs}")
        return self.union_all().intersects(to_shapely(other))

    def union_all(self) -> Polygon:
        try:
            return box(*self.bounds)
        except TypeError:
            return Polygon()

    @property
    def mask_percentage(self) -> float:
        return self.mask.values.sum() / (self.mask.width * self.mask.height) * 100

    @property
    def year(self) -> str:
        if hasattr(self, "_year") and self._year:
            return self._year
        return self.date[:4]

    @property
    def month(self) -> str:
        if hasattr(self, "_month") and self._month:
            return self._month
        return "".join(self.date.split("-"))[4:6]

    @property
    def name(self) -> str | None:
        if hasattr(self, "_name") and self._name is not None:
            return self._name
        try:
            return Path(self.path).name
        except (ValueError, AttributeError):
            return None

    @name.setter
    def name(self, value) -> None:
        self._name = value

    @property
    def stem(self) -> str | None:
        try:
            return Path(self.path).stem
        except (AttributeError, ValueError):
            return None

    @property
    def level(self) -> str:
        return self._name_regex_searcher("level", self.image_patterns)

    def _add_metadata_attributes(self):

        missing_attributes = {}
        for key, value in self.metadata_attributes.items():
            if getattr(self, key) is None:
                missing_attributes[key] = value

        if not missing_attributes:
            return

        file_contents: list[str] = []
        for path in self._all_file_paths:
            if ".xml" not in path:
                continue
            with _open_func(path, "rb") as file:
                file_contents.append(file.read().decode("utf-8"))

        for key, value in missing_attributes.items():
            results = None
            for i, filetext in enumerate(file_contents):
                if isinstance(value, str) and value in dir(self):
                    method = getattr(self, value)
                    try:
                        results = method(filetext)
                    except _RegexError as e:
                        if i == len(self._all_file_paths) - 1:
                            raise e
                        continue
                    if results is not None:
                        break

                if callable(value):
                    try:
                        results = value(filetext)
                    except _RegexError as e:
                        if i == len(self._all_file_paths) - 1:
                            raise e
                        continue
                    if results is not None:
                        break

                try:
                    results = _extract_regex_match_from_string(filetext, value)
                except _RegexError as e:
                    if i == len(self._all_file_paths) - 1:
                        raise e

            if isinstance(results, BandIdDict) and isinstance(self, Band):
                results = results[self.band_id]

            setattr(self, key, results)


class Band(_ImageBandBase):
    """Band holding a single 2 dimensional array representing an image band."""

    cmap: ClassVar[str | None] = None

    @classmethod
    def from_gdf(
        cls,
        gdf: GeoDataFrame | GeoSeries,
        res: int,
        *,
        fill: int = 0,
        all_touched: bool = False,
        merge_alg: Callable = MergeAlg.replace,
        default_value: int = 1,
        dtype: Any | None = None,
        **kwargs,
    ) -> None:
        """Create Band from a GeoDataFrame."""
        arr: np.ndarray = _gdf_to_arr(
            gdf,
            res=res,
            fill=fill,
            all_touched=all_touched,
            merge_alg=merge_alg,
            default_value=default_value,
            dtype=dtype,
        )

        obj = cls(arr, res=res, crs=gdf.crs, bounds=gdf.total_bounds, **kwargs)
        obj._from_gdf = True
        return obj

    def __init__(
        self,
        data: str | np.ndarray,
        res: int | None,
        crs: Any | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        nodata: int | None = None,
        mask: "Band | None" = None,
        file_system: GCSFileSystem | None = None,
        processes: int = 1,
        name: str | None = None,
        band_id: str | None = None,
        cmap: str | None = None,
        all_file_paths: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Band initialiser."""
        super().__init__(**kwargs)

        if isinstance(data, (str | Path | os.PathLike)) and any(
            arg is not None for arg in [crs, bounds]
        ):
            raise ValueError("Can only specify 'bounds' and 'crs' if data is an array.")

        self._mask = mask
        self._values = None
        self.nodata = nodata
        self._crs = crs
        bounds = to_bbox(bounds) if bounds is not None else None
        self._bounds = bounds
        self._all_file_paths = all_file_paths

        self._image = None

        for key in self.metadata_attributes:
            setattr(self, key, None)

        if isinstance(data, np.ndarray):
            self.values = data
            if self._bounds is None:
                raise ValueError("Must specify bounds when data is an array.")
            self._crs = crs
            self.transform = _get_transform_from_bounds(
                self._bounds, shape=self.values.shape
            )
            self._from_array = True

        elif not isinstance(data, (str | Path | os.PathLike)):
            raise TypeError(
                "'data' must be string, Path-like or numpy.ndarray. "
                f"Got {type(data)}"
            )
        else:
            self._path = str(data)

        self._res = res
        if cmap is not None:
            self.cmap = cmap
        self.file_system = file_system
        self._name = name
        self._band_id = band_id
        self.processes = processes

        if (
            kwargs.get("_add_metadata_attributes", True)
            and self.metadata_attributes
            and self.path is not None
        ):
            if self._all_file_paths is None:
                self._all_file_paths = _get_all_file_paths(str(Path(self.path).parent))
            self._add_metadata_attributes()

    def __lt__(self, other: "Band") -> bool:
        """Makes Bands sortable by band_id."""
        return self.band_id < other.band_id

    # def __getattribute__(self, attr: str) -> Any:
    #     # try:
    #     #     value =
    #     # except AttributeError:
    #     #     value = None

    #     if (
    #         attr in (super().__getattribute__("metadata_attributes") or {})
    #         and super().__getattribute__(attr) is None
    #     ):
    #         if self._all_file_paths is None:
    #             self._all_file_paths = _get_all_file_paths(str(Path(self.path).parent))
    #         self._add_metadata_attributes()
    #     return super().__getattribute__(attr)

    @property
    def values(self) -> np.ndarray:
        """The numpy array, if loaded."""
        if self._values is None:
            raise ArrayNotLoadedError("array is not loaded.")
        return self._values

    @values.setter
    def values(self, new_val):
        if not isinstance(new_val, np.ndarray):
            raise TypeError(
                f"{self.__class__.__name__} 'values' must be np.ndarray. Got {type(new_val)}"
            )
        self._values = new_val

    @property
    def mask(self) -> "Band":
        """Mask Band."""
        return self._mask

    @mask.setter
    def mask(self, values: "Band") -> None:
        if values is not None and not isinstance(values, Band):
            raise TypeError(f"'mask' should be of type Band. Got {type(values)}")
        self._mask = values

    @property
    def band_id(self) -> str:
        """Band id."""
        if self._band_id is not None:
            return self._band_id
        return self._name_regex_searcher("band", self.filename_patterns)

    @property
    def height(self) -> int:
        """Pixel heigth of the image band."""
        return self.values.shape[-2]

    @property
    def width(self) -> int:
        """Pixel width of the image band."""
        return self.values.shape[-1]

    @property
    def tile(self) -> str:
        """Tile name from filename_regex."""
        if hasattr(self, "_tile") and self._tile:
            return self._tile
        return self._name_regex_searcher(
            "tile", self.filename_patterns + self.image_patterns
        )

    @property
    def date(self) -> str:
        """Tile name from filename_regex."""
        if hasattr(self, "_date") and self._date:
            return self._date

        return self._name_regex_searcher(
            "date", self.filename_patterns + self.image_patterns
        )

    @property
    def crs(self) -> str | None:
        """Coordinate reference system."""
        if self._crs is None:
            self._add_crs_and_bounds()
        return self._crs

    @property
    def bounds(self) -> tuple[int, int, int, int] | None:
        """Bounds as tuple (minx, miny, maxx, maxy)."""
        if self._bounds is None:
            self._add_crs_and_bounds()
        return self._bounds

    def _add_crs_and_bounds(self) -> None:
        with opener(self.path, file_system=self.file_system) as file:
            with rasterio.open(file) as src:
                self._bounds = to_bbox(src.bounds)
                self._crs = src.crs

    def get_n_largest(
        self, n: int, precision: float = 0.000001, column: str = "value"
    ) -> GeoDataFrame:
        """Get the largest values of the array as polygons in a GeoDataFrame."""
        copied = self.copy()
        value_must_be_at_least = np.sort(np.ravel(copied.values))[-n] - (precision or 0)
        copied._values = np.where(copied.values >= value_must_be_at_least, 1, 0)
        df = copied.to_gdf(column).loc[lambda x: x[column] == 1]
        df[column] = f"largest_{n}"
        return df

    def get_n_smallest(
        self, n: int, precision: float = 0.000001, column: str = "value"
    ) -> GeoDataFrame:
        """Get the lowest values of the array as polygons in a GeoDataFrame."""
        copied = self.copy()
        value_must_be_at_least = np.sort(np.ravel(copied.values))[n] - (precision or 0)
        copied._values = np.where(copied.values <= value_must_be_at_least, 1, 0)
        df = copied.to_gdf(column).loc[lambda x: x[column] == 1]
        df[column] = f"smallest_{n}"
        return df

    def load(
        self,
        bounds: tuple | Geometry | GeoDataFrame | GeoSeries | None = None,
        indexes: int | tuple[int] | None = None,
        masked: bool | None = None,
        **kwargs,
    ) -> "Band":
        """Load and potentially clip the array.

        The array is stored in the 'values' property.
        """
        if masked is None:
            masked = True if self.mask is None else False

        bounds_was_none = bounds is None

        bounds = _get_bounds(bounds, self._bbox)

        should_return_empty: bool = bounds is not None and bounds.area == 0
        if should_return_empty:
            self._values = np.array([])
            if self.mask is not None and not self.is_mask:
                self._mask = self._mask.load()
            self._bounds = None
            self.transform = None
            try:
                self._image._mask = self._mask
            except AttributeError:
                pass
            return self

        if self.has_array and bounds_was_none:
            return self

        # round down/up to integer to avoid precision trouble
        if bounds is not None:
            minx, miny, maxx, maxy = to_bbox(bounds)
            bounds = (int(minx), int(miny), math.ceil(maxx), math.ceil(maxy))

        if indexes is None:
            indexes = 1

        # as tuple to ensure we get 3d array
        _indexes: tuple[int] = (indexes,) if isinstance(indexes, int) else indexes

        # allow setting a fixed out_shape for the array, in order to make mask same shape as values
        out_shape = kwargs.pop("out_shape", None)

        if self.has_array:
            self.values = _clip_loaded_array(
                self.values, bounds, self.transform, self.crs, out_shape, **kwargs
            )
            self._bounds = bounds
            self.transform = _get_transform_from_bounds(self._bounds, self.values.shape)

        else:
            with opener(self.path, file_system=self.file_system) as f:
                with rasterio.open(f, nodata=self.nodata) as src:
                    self._res = int(src.res[0]) if not self.res else self.res

                    if self.nodata is None or np.isnan(self.nodata):
                        self.nodata = src.nodata
                    else:
                        dtype_min_value = _get_dtype_min(src.dtypes[0])
                        dtype_max_value = _get_dtype_max(src.dtypes[0])
                        if (
                            self.nodata > dtype_max_value
                            or self.nodata < dtype_min_value
                        ):
                            src._dtypes = tuple(
                                rasterio.dtypes.get_minimum_dtype(self.nodata)
                                for _ in range(len(_indexes))
                            )

                    if bounds is None:
                        if self._res != int(src.res[0]):
                            if out_shape is None:
                                out_shape = _get_shape_from_bounds(
                                    to_bbox(src.bounds), self.res, indexes
                                )
                            self.transform = _get_transform_from_bounds(
                                to_bbox(src.bounds), shape=out_shape
                            )
                        else:
                            self.transform = src.transform

                        self._values = src.read(
                            indexes=indexes,
                            out_shape=out_shape,
                            masked=masked,
                            **kwargs,
                        )
                    else:
                        window = rasterio.windows.from_bounds(
                            *bounds, transform=src.transform
                        )

                        if out_shape is None:
                            out_shape = _get_shape_from_bounds(
                                bounds, self.res, indexes
                            )

                        self._values = src.read(
                            indexes=indexes,
                            window=window,
                            boundless=False,
                            out_shape=out_shape,
                            masked=masked,
                            **kwargs,
                        )

                        assert out_shape == self._values.shape, (
                            out_shape,
                            self._values.shape,
                        )

                        self.transform = rasterio.transform.from_bounds(
                            *bounds, self.width, self.height
                        )
                        self._bounds = bounds

                    if self.nodata is not None and not np.isnan(self.nodata):
                        if isinstance(self.values, np.ma.core.MaskedArray):
                            self.values.data[self.values.data == src.nodata] = (
                                self.nodata
                            )
                        else:
                            self.values[self.values == src.nodata] = self.nodata

        if self.masking and self.is_mask:
            self.values = np.isin(self.values, self.masking["values"])

        elif self.mask is not None and not isinstance(
            self.values, np.ma.core.MaskedArray
        ):
            self.mask = self.mask.copy().load(
                bounds=bounds, indexes=indexes, out_shape=out_shape, **kwargs
            )
            mask_arr = self.mask.values

            self._values = np.ma.array(
                self._values, mask=mask_arr, fill_value=self.nodata
            )

        try:
            self._image._mask = self._mask
        except AttributeError:
            pass

        return self

    @property
    def is_mask(self) -> bool:
        """True if the band_id is equal to the masking band_id."""
        return self.band_id == self.masking["band_id"]

    @property
    def has_array(self) -> bool:
        """Whether the array is loaded."""
        try:
            if not isinstance(self.values, np.ndarray):
                raise ValueError()
            return True
        except ValueError:  # also catches ArrayNotLoadedError
            return False

    def write(
        self, path: str | Path, driver: str = "GTiff", compress: str = "LZW", **kwargs
    ) -> None:
        """Write the array as an image file."""
        if not hasattr(self, "_values"):
            raise ValueError(
                "Can only write image band from Band constructed from array."
            )

        if self.crs is None:
            raise ValueError("Cannot write None crs to image.")

        profile = {
            "driver": driver,
            "compress": compress,
            "dtype": rasterio.dtypes.get_minimum_dtype(self.values),
            "crs": self.crs,
            "transform": self.transform,
            "nodata": self.nodata,
            "count": 1 if len(self.values.shape) == 2 else self.values.shape[0],
            "height": self.height,
            "width": self.width,
        } | kwargs

        with opener(path, "wb", file_system=self.file_system) as f:
            with rasterio.open(f, "w", **profile) as dst:

                if dst.nodata is None:
                    dst.nodata = _get_dtype_min(dst.dtypes[0])

                # if (
                #     isinstance(self.values, np.ma.core.MaskedArray)
                #     # and dst.nodata is not None
                # ):
                #     self.values.data[np.isnan(self.values.data)] = dst.nodata
                #     self.values.data[self.values.mask] = dst.nodata

                if len(self.values.shape) == 2:
                    dst.write(self.values, indexes=1)
                else:
                    for i in range(self.values.shape[0]):
                        dst.write(self.values[i], indexes=i + 1)

                if isinstance(self.values, np.ma.core.MaskedArray):
                    dst.write_mask(self.values.mask)

        self._path = str(path)

    def apply(self, func: Callable, **kwargs) -> "Band":
        """Apply a function to the array."""
        self.values = func(self.values, **kwargs)
        return self

    def normalize(self) -> "Band":
        """Normalize array values between 0 and 1."""
        arr = self.values
        self.values = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        return self

    def sample(self, size: int = 1000, mask: Any = None, **kwargs) -> "Image":
        """Take a random spatial sample area of the Band."""
        copied = self.copy()
        if mask is not None:
            point = GeoSeries([copied.union_all()]).clip(mask).sample_points(1)
        else:
            point = GeoSeries([copied.union_all()]).sample_points(1)
        buffered = point.buffer(size / 2).clip(copied.union_all())
        copied = copied.load(bounds=buffered.total_bounds, **kwargs)
        return copied

    def buffer(self, distance: int, copy: bool = True) -> "Band":
        """Buffer array points with the value 1 in a binary array.

        Args:
            distance: Number of array cells to buffer by.
            copy: Whether to copy the Band.

        Returns:
            Band with buffered values.
        """
        copied = self.copy() if copy else self
        copied.values = array_buffer(copied.values, distance)
        return copied

    def gradient(self, degrees: bool = False, copy: bool = True) -> "Band":
        """Get the slope of an elevation band.

        Calculates the absolute slope between the grid cells
        based on the image resolution.

        For multi-band images, the calculation is done for each band.

        Args:
            band: band instance.
            degrees: If False (default), the returned values will be in ratios,
                where a value of 1 means 1 meter up per 1 meter forward. If True,
                the values will be in degrees from 0 to 90.
            copy: Whether to copy or overwrite the original Raster.
                Defaults to True.

        Returns:
            The class instance with new array values, or a copy if copy is True.

        Examples:
        ---------
        Making an array where the gradient to the center is always 10.

        >>> import sgis as sg
        >>> import numpy as np
        >>> arr = np.array(
        ...         [
        ...             [100, 100, 100, 100, 100],
        ...             [100, 110, 110, 110, 100],
        ...             [100, 110, 120, 110, 100],
        ...             [100, 110, 110, 110, 100],
        ...             [100, 100, 100, 100, 100],
        ...         ]
        ...     )

        Now let's create a Raster from this array with a resolution of 10.

        >>> band = sg.Band(arr, crs=None, bounds=(0, 0, 50, 50), res=10)

        The gradient will be 1 (1 meter up for every meter forward).
        The calculation is by default done in place to save memory.

        >>> band.gradient()
        >>> band.values
        array([[0., 1., 1., 1., 0.],
            [1., 1., 1., 1., 1.],
            [1., 1., 0., 1., 1.],
            [1., 1., 1., 1., 1.],
            [0., 1., 1., 1., 0.]])
        """
        copied = self.copy() if copy else self
        copied._values = _get_gradient(copied, degrees=degrees, copy=copy)
        return copied

    def zonal(
        self,
        polygons: GeoDataFrame,
        aggfunc: str | Callable | list[Callable | str],
        array_func: Callable | None = None,
        dropna: bool = True,
    ) -> GeoDataFrame:
        """Calculate zonal statistics in polygons.

        Args:
            polygons: A GeoDataFrame of polygon geometries.
            aggfunc: Function(s) of which to aggregate the values
                within each polygon.
            array_func: Optional calculation of the raster
                array before calculating the zonal statistics.
            dropna: If True (default), polygons with all missing
                values will be removed.

        Returns:
            A GeoDataFrame with aggregated values per polygon.
        """
        idx_mapper, idx_name = get_index_mapper(polygons)
        polygons, aggfunc, func_names = _prepare_zonal(polygons, aggfunc)
        poly_iter = _make_geometry_iterrows(polygons)

        kwargs = {
            "band": self,
            "aggfunc": aggfunc,
            "array_func": array_func,
            "func_names": func_names,
        }

        if self.processes == 1:
            aggregated = [_zonal_one_pair(i, poly, **kwargs) for i, poly in poly_iter]
        else:
            with joblib.Parallel(n_jobs=self.processes, backend="loky") as parallel:
                aggregated = parallel(
                    joblib.delayed(_zonal_one_pair)(i, poly, **kwargs)
                    for i, poly in poly_iter
                )

        return _zonal_post(
            aggregated,
            polygons=polygons,
            idx_mapper=idx_mapper,
            idx_name=idx_name,
            dropna=dropna,
        )

    def to_gdf(self, column: str = "value") -> GeoDataFrame:
        """Create a GeoDataFrame from the image Band.

        Args:
            column: Name of resulting column that holds the raster values.

        Returns:
            A GeoDataFrame with a geometry column and array values.
        """
        if not hasattr(self, "_values"):
            raise ValueError("Array is not loaded.")

        if self.values.shape[0] == 0:
            return GeoDataFrame({"geometry": []}, crs=self.crs)

        return GeoDataFrame(
            pd.DataFrame(
                _array_to_geojson(
                    self.values, self.transform, processes=self.processes
                ),
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
        )

    def __repr__(self) -> str:
        """String representation."""
        try:
            band_id = f"'{self.band_id}'" if self.band_id else None
        except (ValueError, AttributeError):
            band_id = None
        try:
            path = f"'{self.path}'"
        except (ValueError, AttributeError):
            path = None
        return (
            f"{self.__class__.__name__}(band_id={band_id}, res={self.res}, path={path})"
        )


class NDVIBand(Band):
    """Band for NDVI values."""

    cmap: str = "Greens"

    # @staticmethod
    # def get_cmap(arr: np.ndarray):
    #     return get_cmap(arr)


def median_as_int_and_minimum_dtype(arr: np.ndarray) -> np.ndarray:
    arr = np.median(arr, axis=0).astype(int)
    min_dtype = rasterio.dtypes.get_minimum_dtype(arr)
    return arr.astype(min_dtype)


class Image(_ImageBandBase):
    """Image consisting of one or more Bands."""

    band_class: ClassVar[Band] = Band

    def __init__(
        self,
        data: str | Path | Sequence[Band],
        res: int | None = None,
        file_system: GCSFileSystem | None = None,
        processes: int = 1,
        df: pd.DataFrame | None = None,
        nodata: int | None = None,
        all_file_paths: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Image initialiser."""
        super().__init__(**kwargs)

        self.nodata = nodata
        self._res = res
        self._crs = None
        self.file_system = file_system
        self.processes = processes

        if hasattr(data, "__iter__") and all(isinstance(x, Band) for x in data):
            self._bands = list(data)
            if res is None:
                res = list({band.res for band in self.bands})
                if len(res) == 1:
                    self._res = res[0]
                else:
                    raise ValueError(f"Different resolutions for the bands: {res}")
            else:
                self._res = res
            return

        if not isinstance(data, (str | Path | os.PathLike)):
            raise TypeError("'data' must be string, Path-like or a sequence of Band.")

        self._bands = None
        self._path = _fix_path(data)  # str(data).rstrip("/").rstrip(r"\"")

        if all_file_paths is None and self.path:
            self._all_file_paths = _get_all_file_paths(self.path)
        elif self.path:
            self._all_file_paths = [
                x for x in all_file_paths if self.path in _fix_path(x)
            ]
        else:
            self._all_file_paths = None

        if df is None:
            # file_paths = _get_all_file_paths(self.path)

            if not self._all_file_paths:
                self._all_file_paths = [self.path]
            df = self._create_metadata_df(self._all_file_paths)

        df["image_path"] = df["image_path"].astype(str)

        cols_to_explode = [
            "file_path",
            "filename",
            *[x for x in df if FILENAME_COL_SUFFIX in x],
        ]
        try:
            df = df.explode(cols_to_explode, ignore_index=True)
        except ValueError:
            for col in cols_to_explode:
                df = df.explode(col)
            df = df.loc[lambda x: ~x["filename"].duplicated()].reset_index(drop=True)

        df = df.loc[lambda x: x["image_path"] == _fix_path(self.path)]

        self._df = df

        for key in self.metadata_attributes:
            setattr(self, key, None)

        if self.metadata_attributes:
            self._add_metadata_attributes()

    @property
    def values(self) -> np.ndarray:
        """3 dimensional numpy array."""
        values = [band.values for band in self]
        if self.mask is not None:
            mask = [band.mask.values for band in self]
            return np.ma.array(values, mask=mask, fill_value=self.nodata)
        return np.array(values)

    def ndvi(self, red_band: str, nir_band: str, copy: bool = True) -> NDVIBand:
        """Calculate the NDVI for the Image."""
        copied = self.copy() if copy else self
        red = copied[red_band].load()
        nir = copied[nir_band].load()

        arr: np.ndarray | np.ma.core.MaskedArray = ndvi(red.values, nir.values)

        return NDVIBand(
            arr,
            bounds=red.bounds,
            crs=red.crs,
            mask=red.mask,
            **red._common_init_kwargs,
        )

    def get_brightness(
        self,
        bounds: tuple | Geometry | GeoDataFrame | GeoSeries | None = None,
        rbg_bands: list[str] | None = None,
    ) -> Band:
        """Get a Band with a brightness score of the Image's RBG bands."""
        if rbg_bands is None:
            try:
                r, b, g = self.rbg_bands
            except AttributeError as err:
                raise AttributeError(
                    "Must specify rbg_bands when there is no class variable 'rbd_bands'"
                ) from err
        else:
            r, b, g = rbg_bands

        red = self[r].load(bounds=bounds)
        blue = self[b].load(bounds=bounds)
        green = self[g].load(bounds=bounds)

        brightness = (
            0.299 * red.values + 0.587 * green.values + 0.114 * blue.values
        ).astype(int)

        return Band(
            brightness,
            bounds=red.bounds,
            crs=self.crs,
            mask=self.mask,
            **self._common_init_kwargs,
        )

    def to_xarray(self) -> DataArray:
        """Convert the raster to  an xarray.DataArray."""
        name = self.name or self.__class__.__name__.lower()
        coords = _generate_spatial_coords(
            self[0].transform, self[0].width, self[0].height
        )
        dims = ["band", "y", "x"]
        return xr.DataArray(
            self.values,
            coords=coords,
            dims=dims,
            name=name,
            attrs={"crs": self.crs},
        )

    @property
    def mask(self) -> Band | None:
        """Mask Band."""
        if self._mask is not None:
            # if not self._mask.has_array:
            #     try:
            #         self._mask.values = self[0]._mask.values
            #     except Exception:
            #         pass
            return self._mask
        if self.masking is None:
            return None

        mask_band_id = self.masking["band_id"]
        mask_paths = [path for path in self._df["file_path"] if mask_band_id in path]
        if len(mask_paths) > 1:
            raise ValueError(
                f"Multiple file_paths match mask band_id {mask_band_id} for {self.path}"
            )
        elif not mask_paths:
            raise ValueError(
                f"No file_paths match mask band_id {mask_band_id} for {self.path}"
            )
        self._mask = self.band_class(
            mask_paths[0],
            _add_metadata_attributes=False,
            **self._common_init_kwargs,
        )

        return self._mask

    @mask.setter
    def mask(self, values: Band) -> None:
        if values is None:
            self._mask = None
            for band in self:
                band.mask = None
            return
        if not isinstance(values, Band):
            raise TypeError(f"mask must be Band. Got {type(values)}")
        self._mask = values
        mask_arr = self._mask.values
        for band in self:
            band._mask = self._mask
            try:
                band.values = np.ma.array(
                    band.values, mask=mask_arr, fill_value=band.nodata
                )
            except ArrayNotLoadedError:
                pass

    @property
    def band_ids(self) -> list[str]:
        """The Band ids."""
        return [band.band_id for band in self]

    @property
    def file_paths(self) -> list[str]:
        """The Band file paths."""
        return [band.path for band in self]

    @property
    def bands(self) -> list[Band]:
        """The Image Bands."""
        if self._bands is not None:
            return self._bands

        self._bands = [
            self.band_class(
                path,
                mask=self.mask,
                _add_metadata_attributes=False,
                **self._common_init_kwargs,
            )
            for path in (self._df["file_path"])
        ]

        if self.masking:
            mask_band_id = self.masking["band_id"]
            self._bands = [
                band for band in self._bands if mask_band_id not in band.path
            ]

        if (
            self.filename_patterns
            and any(_get_non_optional_groups(pat) for pat in self.filename_patterns)
            or self.image_patterns
            and any(_get_non_optional_groups(pat) for pat in self.image_patterns)
        ):
            self._bands = [band for band in self._bands if band.band_id is not None]

        if self.filename_patterns:
            self._bands = [
                band
                for band in self._bands
                if any(re.search(pat, band.name) for pat in self.filename_patterns)
            ]

        if self.image_patterns:
            self._bands = [
                band
                for band in self._bands
                if any(
                    re.search(pat, Path(band.path).parent.name)
                    for pat in self.image_patterns
                )
            ]

        if self._should_be_sorted:
            self._bands = list(sorted(self._bands))

        for key in self.metadata_attributes:
            for band in self:
                value = getattr(self, key)
                if value is None:
                    continue
                if isinstance(value, BandIdDict):
                    try:
                        value = value[band.band_id]
                    except KeyError:
                        continue
                setattr(band, key, value)

        for band in self:
            band._image = self

        return self._bands

    @property
    def _should_be_sorted(self) -> bool:
        sort_groups = ["band", "band_id"]
        return self.filename_patterns and any(
            group in _get_non_optional_groups(pat)
            for group in sort_groups
            for pat in self.filename_patterns
        )

    @property
    def tile(self) -> str:
        """Tile name from filename_regex."""
        if hasattr(self, "_tile") and self._tile:
            return self._tile
        return self._name_regex_searcher(
            "tile", self.image_patterns + self.filename_patterns
        )

    @property
    def date(self) -> str:
        """Tile name from filename_regex."""
        if hasattr(self, "_date") and self._date:
            return self._date

        return self._name_regex_searcher(
            "date", self.image_patterns + self.filename_patterns
        )

    @property
    def crs(self) -> str | None:
        """Coordinate reference system of the Image."""
        if self._crs is not None:
            return self._crs
        if not len(self):
            return None
        self._crs = get_common_crs(self)
        return self._crs

    @property
    def bounds(self) -> tuple[int, int, int, int] | None:
        """Bounds of the Image (minx, miny, maxx, maxy)."""
        try:
            return get_total_bounds([band.bounds for band in self])
        except exceptions.RefreshError:
            bounds = []
            for band in self:
                time.sleep(0.1)
                bounds.append(band.bounds)
            return get_total_bounds(bounds)

    def to_gdf(self, column: str = "value") -> GeoDataFrame:
        """Convert the array to a GeoDataFrame of grid polygons and values."""
        return pd.concat(
            [band.to_gdf(column=column) for band in self], ignore_index=True
        )

    def sample(
        self, n: int = 1, size: int = 1000, mask: Any = None, **kwargs
    ) -> "Image":
        """Take a random spatial sample of the image."""
        copied = self.copy()
        if mask is not None:
            points = GeoSeries([self.union_all()]).clip(mask).sample_points(n)
        else:
            points = GeoSeries([self.union_all()]).sample_points(n)
        buffered = points.buffer(size / 2).clip(self.union_all())
        boxes = to_gdf([box(*arr) for arr in buffered.bounds.values], crs=self.crs)
        copied._bands = [band.load(bounds=boxes, **kwargs) for band in copied]
        copied._bounds = get_total_bounds([band.bounds for band in copied])
        return copied

    def __getitem__(
        self, band: str | int | Sequence[str] | Sequence[int]
    ) -> "Band | Image":
        """Get bands by band_id or integer index or a sequence of such.

        Returns a Band if a string or int is passed,
        returns an Image if a sequence of strings or integers is passed.
        """
        if isinstance(band, str):
            return self._get_band(band)
        if isinstance(band, int):
            return self.bands[band]  # .copy()

        copied = self.copy()
        try:
            copied._bands = [copied._get_band(x) for x in band]
        except TypeError:
            try:
                copied._bands = [copied.bands[i] for i in band]
            except TypeError as e:
                raise TypeError(
                    f"{self.__class__.__name__} indices should be string, int "
                    f"or sequence of string or int. Got {band}."
                ) from e
        return copied

    def __contains__(self, item: str | Sequence[str]) -> bool:
        """Check if the Image contains a band_id (str) or all band_ids in a sequence."""
        if isinstance(item, str):
            return item in self.band_ids
        return all(x in self.band_ids for x in item)

    def __lt__(self, other: "Image") -> bool:
        """Makes Images sortable by date."""
        try:
            return self.date < other.date
        except Exception as e:
            print(self.path)
            print(self.date)
            print(other.path)
            print(other.date)
            raise e

    def __iter__(self) -> Iterator[Band]:
        """Iterate over the Bands."""
        return iter(self.bands)

    def __len__(self) -> int:
        """Number of bands in the Image."""
        return len(self.bands)

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(bands={self.bands})"

    def _get_band(self, band: str) -> Band:
        if not isinstance(band, str):
            raise TypeError(f"band must be string. Got {type(band)}")

        bands = [x for x in self.bands if x.band_id == band]
        if len(bands) == 1:
            return bands[0]
        if len(bands) > 1:
            raise ValueError(f"Multiple matches for band_id {band} for {self}")

        bands = [x for x in self.bands if x.band_id == band.replace("B0", "B")]
        if len(bands) == 1:
            return bands[0]

        bands = [x for x in self.bands if x.band_id.replace("B0", "B") == band]
        if len(bands) == 1:
            return bands[0]

        try:
            more_bands = [x for x in self.bands if x.path == band]
        except PathlessImageError:
            more_bands = bands

        if len(more_bands) == 1:
            return more_bands[0]

        if len(bands) > 1:
            prefix = "Multiple"
        elif not bands:
            prefix = "No"

        raise KeyError(
            f"{prefix} matches for band {band} among paths {[Path(band.path).name for band in self.bands]}"
        )


class ImageCollection(_ImageBase):
    """Collection of Images.

    Loops though Images.
    """

    image_class: ClassVar[Image] = Image
    band_class: ClassVar[Band] = Band
    _metadata_attribute_collection_type: ClassVar[type] = pd.Series

    def __init__(
        self,
        data: str | Path | Sequence[Image] | Sequence[str | Path],
        res: int,
        level: str | None = NoLevel,
        processes: int = 1,
        file_system: GCSFileSystem | None = None,
        metadata: str | dict | pd.DataFrame | None = None,
        nodata: int | None = None,
        **kwargs,
    ) -> None:
        """Initialiser."""
        super().__init__(**kwargs)

        self.nodata = nodata
        self.level = level if not isinstance(level, NoLevel) else None
        self.processes = processes
        self.file_system = file_system
        self._res = res
        self._band_ids = None
        self._crs = None  # crs

        if metadata is not None:
            if isinstance(metadata, (str | Path | os.PathLike)):
                self.metadata = _read_parquet_func(metadata)
            else:
                self.metadata = metadata
        else:
            self.metadata = metadata

        self._df = None
        self._all_file_paths = None
        self._images = None

        if hasattr(data, "__iter__") and not isinstance(data, str):
            self._path = None
            if all(isinstance(x, Image) for x in data):
                self.images = [x.copy() for x in data]
                return
            elif all(isinstance(x, (str | Path | os.PathLike)) for x in data):
                self._all_file_paths = list(
                    itertools.chain.from_iterable(
                        _get_all_file_paths(str(path)) for path in data
                    )
                )
                self._df = self._create_metadata_df([str(x) for x in data])
                return

        if not isinstance(data, (str | Path | os.PathLike)):
            raise TypeError("'data' must be string, Path-like or a sequence of Image.")

        self._path = str(data)

        self._all_file_paths = _get_all_file_paths(self.path)

        if self.level:
            self._all_file_paths = [
                path for path in self._all_file_paths if self.level in path
            ]

        self._df = self._create_metadata_df(self._all_file_paths)

    @property
    def values(self) -> np.ndarray:
        """4 dimensional numpy array."""
        if isinstance(self[0].values, np.ma.core.MaskedArray):
            return np.ma.array([img.values for img in self])
        return np.array([img.values for img in self])

    @property
    def mask(self) -> np.ndarray:
        """4 dimensional numpy array."""
        return np.array([img.mask.values for img in self])

    def groupby(self, by: str | list[str], **kwargs) -> ImageCollectionGroupBy:
        """Group the Collection by Image or Band attribute(s)."""
        df = pd.DataFrame(
            [(i, img) for i, img in enumerate(self) for _ in img],
            columns=["_image_idx", "_image_instance"],
        )

        if isinstance(by, str):
            by = [by]

        for attr in by:
            if attr == "bounds":
                # need integers to check equality when grouping
                df[attr] = [
                    tuple(int(x) for x in band.bounds) for img in self for band in img
                ]
                continue

            try:
                df[attr] = [getattr(band, attr) for img in self for band in img]
            except AttributeError:
                df[attr] = [getattr(img, attr) for img in self for _ in img]

        with joblib.Parallel(n_jobs=self.processes, backend="loky") as parallel:
            return ImageCollectionGroupBy(
                sorted(
                    parallel(
                        joblib.delayed(_copy_and_add_df_parallel)(i, group, self)
                        for i, group in df.groupby(by, **kwargs)
                    )
                ),
                by=by,
                collection=self,
            )

    def explode(self, copy: bool = True) -> "ImageCollection":
        """Make all Images single-banded."""
        copied = self.copy() if copy else self
        copied.images = [
            self.image_class(
                [band],
                masking=self.masking,
                band_class=self.band_class,
                **self._common_init_kwargs,
                df=self._df,
                all_file_paths=self._all_file_paths,
            )
            for img in self
            for band in img
        ]
        for img in copied:
            assert len(img) == 1
            try:
                img._path = img[0].path
            except PathlessImageError:
                pass
        return copied

    def apply(self, func: Callable, **kwargs) -> "ImageCollection":
        """Apply a function to all bands in each image of the collection."""
        for img in self:
            img._bands = [func(band, **kwargs) for band in img]
        return self

    def get_unique_band_ids(self) -> list[str]:
        """Get a list of unique band_ids across all images."""
        return list({band.band_id for img in self for band in img})

    def filter(
        self,
        bands: str | list[str] | None = None,
        date_ranges: DATE_RANGES_TYPE = None,
        bbox: GeoDataFrame | GeoSeries | Geometry | tuple[float] | None = None,
        intersects: GeoDataFrame | GeoSeries | Geometry | tuple[float] | None = None,
        max_cloud_coverage: int | None = None,
        copy: bool = True,
    ) -> "ImageCollection":
        """Filter images and bands in the collection."""
        copied = self.copy() if copy else self

        if date_ranges:
            copied = copied._filter_dates(date_ranges)

        if max_cloud_coverage is not None:
            copied.images = [
                image
                for image in copied.images
                if image.cloud_coverage_percentage < max_cloud_coverage
            ]

        if bbox is not None:
            copied = copied._filter_bounds(bbox)
            copied._set_bbox(bbox)

        if intersects is not None:
            copied = copied._filter_bounds(intersects)

        if bands is not None:
            if isinstance(bands, str):
                bands = [bands]
            bands = set(bands)
            copied._band_ids = bands
            copied.images = [img[bands] for img in copied.images if bands in img]

        return copied

    def merge(
        self,
        bounds: tuple | Geometry | GeoDataFrame | GeoSeries | None = None,
        method: str | Callable = "mean",
        as_int: bool = True,
        indexes: int | tuple[int] | None = None,
        **kwargs,
    ) -> Band:
        """Merge all areas and all bands to a single Band."""
        bounds = _get_bounds(bounds, self._bbox)
        if bounds is not None:
            bounds = to_bbox(bounds)

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

        if self.masking or method not in list(rasterio.merge.MERGE_METHODS) + ["mean"]:
            arr = self._merge_with_numpy_func(
                method=method,
                bounds=bounds,
                as_int=as_int,
                **kwargs,
            )
        else:
            datasets = [_open_raster(path) for path in self.file_paths]
            arr, _ = rasterio.merge.merge(
                datasets,
                res=self.res,
                bounds=(bounds if bounds is not None else self.bounds),
                indexes=_indexes,
                method=_method,
                nodata=self.nodata,
                **kwargs,
            )

        if isinstance(indexes, int) and len(arr.shape) == 3 and arr.shape[0] == 1:
            arr = arr[0]

        if method == "mean":
            if as_int:
                arr = arr // len(datasets)
            else:
                arr = arr / len(datasets)

        if bounds is None:
            bounds = self.bounds

        # return self.band_class(
        band = Band(
            arr,
            bounds=bounds,
            crs=crs,
            mask=self.mask,
            **self._common_init_kwargs,
        )

        band._merged = True
        return band

    def merge_by_band(
        self,
        bounds: tuple | Geometry | GeoDataFrame | GeoSeries | None = None,
        method: str = "mean",
        as_int: bool = True,
        indexes: int | tuple[int] | None = None,
        **kwargs,
    ) -> Image:
        """Merge all areas to a single tile, one band per band_id."""
        bounds = _get_bounds(bounds, self._bbox)
        if bounds is not None:
            bounds = to_bbox(bounds)
        bounds = self.bounds if bounds is None else bounds
        out_bounds = bounds
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
        bands: list[Band] = []
        for (band_id,), band_collection in self.groupby("band_id"):
            if self.masking or method not in list(rasterio.merge.MERGE_METHODS) + [
                "mean"
            ]:
                arr = band_collection._merge_with_numpy_func(
                    method=method,
                    bounds=bounds,
                    as_int=as_int,
                    **kwargs,
                )
            else:
                datasets = [_open_raster(path) for path in band_collection.file_paths]
                arr, _ = rasterio.merge.merge(
                    datasets,
                    res=self.res,
                    bounds=(bounds if bounds is not None else self.bounds),
                    indexes=_indexes,
                    method=_method,
                    nodata=self.nodata,
                    **kwargs,
                )
                if isinstance(indexes, int):
                    arr = arr[0]
                if method == "mean":
                    if as_int:
                        arr = arr // len(datasets)
                    else:
                        arr = arr / len(datasets)

            arrs.append(arr)
            bands.append(
                self.band_class(
                    arr,
                    bounds=out_bounds,
                    crs=crs,
                    band_id=band_id,
                    _add_metadata_attributes=False,
                    **self._common_init_kwargs,
                )
            )

        # return self.image_class( # TODO
        image = Image(
            bands,
            band_class=self.band_class,
            **self._common_init_kwargs,
        )

        image._merged = True
        return image

    def _merge_with_numpy_func(
        self,
        method: str | Callable,
        bounds: tuple | Geometry | GeoDataFrame | GeoSeries | None = None,
        as_int: bool = True,
        indexes: int | tuple[int] | None = None,
        **kwargs,
    ) -> np.ndarray:
        arrs = []
        kwargs["indexes"] = indexes
        bounds = to_shapely(bounds) if bounds is not None else None
        numpy_func = get_numpy_func(method) if not callable(method) else method
        for (_bounds,), collection in self.groupby("bounds"):
            _bounds = (
                to_shapely(_bounds).intersection(bounds)
                if bounds is not None
                else to_shapely(_bounds)
            )
            if not _bounds.area:
                continue

            _bounds = to_bbox(_bounds)
            arr = np.array(
                [
                    (
                        band.load(
                            bounds=(_bounds if _bounds is not None else None),
                            **kwargs,
                        )
                    ).values
                    for img in collection
                    for band in img
                ]
            )
            arr = numpy_func(arr, axis=0)
            if as_int:
                arr = arr.astype(int)
                min_dtype = rasterio.dtypes.get_minimum_dtype(arr)
                arr = arr.astype(min_dtype)

            if len(arr.shape) == 2:
                height, width = arr.shape
            elif len(arr.shape) == 3:
                height, width = arr.shape[1:]
            else:
                raise ValueError(arr.shape)

            transform = rasterio.transform.from_bounds(*_bounds, width, height)
            coords = _generate_spatial_coords(transform, width, height)

            arrs.append(
                xr.DataArray(
                    arr,
                    coords=coords,
                    dims=["y", "x"],
                    attrs={"crs": self.crs},
                )
            )

        merged = merge_arrays(
            arrs,
            res=self.res,
            nodata=self.nodata,
        )

        return merged.to_numpy()

    def sort_images(self, ascending: bool = True) -> "ImageCollection":
        """Sort Images by date."""
        self._images = (
            list(sorted([img for img in self if img.date is not None]))
            + sorted(
                [img for img in self if img.date is None and img.path is not None],
                key=lambda x: x.path,
            )
            + [img for img in self if img.date is None and img.path is None]
        )
        if not ascending:
            self._images = list(reversed(self.images))
        return self

    def load(
        self,
        bounds: tuple | Geometry | GeoDataFrame | GeoSeries | None = None,
        indexes: int | tuple[int] | None = None,
        **kwargs,
    ) -> "ImageCollection":
        """Load all image Bands with threading."""
        if (
            bounds is None
            and indexes is None
            and all(band.has_array for img in self for band in img)
        ):
            return self
        with joblib.Parallel(n_jobs=self.processes, backend="threading") as parallel:
            parallel(
                joblib.delayed(_load_band)(
                    band, bounds=bounds, indexes=indexes, **kwargs
                )
                for img in self
                for band in img
            )

        return self

    def _set_bbox(
        self, bbox: GeoDataFrame | GeoSeries | Geometry | tuple[float]
    ) -> "ImageCollection":
        """Set the mask to be used to clip the images to."""
        self._bbox = to_bbox(bbox)
        # only update images when already instansiated
        if self._images is not None:
            for img in self._images:
                img._bbox = self._bbox
                if img.bands is None:
                    continue
                for band in img:
                    band._bbox = self._bbox
                    bounds = box(*band._bbox).intersection(box(*band.bounds))
                    band._bounds = to_bbox(bounds) if not bounds.is_empty else None

        return self

    def _filter_dates(
        self,
        date_ranges: DATE_RANGES_TYPE = None,
    ) -> "ImageCollection":
        if not isinstance(date_ranges, (tuple, list)):
            raise TypeError(
                "date_ranges should be a 2-length tuple of strings or None, "
                "or a tuple of tuples for multiple date ranges"
            )
        if not self.image_patterns:
            raise ValueError(
                "Cannot set date_ranges when the class's image_regexes attribute is None"
            )

        self.images = [img for img in self if _date_is_within(img.date, date_ranges)]
        return self

    def _filter_bounds(
        self, other: GeoDataFrame | GeoSeries | Geometry | tuple
    ) -> "ImageCollection":
        if self._images is None:
            return self

        other = to_shapely(other)

        # intersects_list = GeoSeries([img.union_all() for img in self]).intersects(other)
        with joblib.Parallel(n_jobs=self.processes, backend="loky") as parallel:
            intersects_list: list[bool] = parallel(
                joblib.delayed(_intesects)(image, other) for image in self
            )

        self.images = [
            image
            for image, intersects in zip(self, intersects_list, strict=False)
            if intersects
        ]
        return self

    def to_xarray(self, **kwargs) -> DataArray:
        """Convert the raster to  an xarray.DataArray."""
        # arrs = []
        # for img in self:
        #     for band in img:
        #         arr = band.load(**kwargs).values
        #         arrs.append(arr)

        # n_images = len(self)
        # n_bands = len(img)
        # height, width = arr.shape

        # arr_4d = np.array(arrs).reshape(n_images, n_bands, height, width)

        try:
            name = Path(self.path).stem
        except TypeError:
            name = self.__class__.__name__.lower()

        first_band = self[0][0]
        coords = _generate_spatial_coords(
            first_band.transform, first_band.width, first_band.height
        )
        dims = ["image", "band", "y", "x"]
        return xr.DataArray(
            self.values,
            coords=coords,
            dims=dims,
            name=name,
            attrs={"crs": self.crs},
        )

    def to_gdfs(self, column: str = "value") -> dict[str, GeoDataFrame]:
        """Convert each band in each Image to a GeoDataFrame."""
        out = {}
        i = 0
        for img in self:
            for band in img:
                i += 1
                try:
                    name = band.name
                except AttributeError:
                    name = f"{self.__class__.__name__}({i})"

                # band.load()

                if name not in out:
                    out[name] = band.to_gdf(column=column)
        return out

    def sample(self, n: int = 1, size: int = 500) -> "ImageCollection":
        """Sample one or more areas of a given size and set this as mask for the images."""
        unioned = self.union_all()
        buffered_in = unioned.buffer(-size / 2)
        if not buffered_in.is_empty:
            bbox = to_gdf(buffered_in)
        else:
            bbox = to_gdf(unioned)

        copied = self.copy()
        sampled_images = []
        while len(sampled_images) < n:
            mask = to_bbox(bbox.sample_points(1).buffer(size))
            images = copied.filter(bbox=mask).images
            random.shuffle(images)
            try:
                images = images[:n]
            except IndexError:
                pass
            sampled_images += images
        copied._images = sampled_images[:n]
        if copied._should_be_sorted:
            copied._images = list(sorted(copied._images))

        return copied

    def sample_tiles(self, n: int) -> "ImageCollection":
        """Sample one or more tiles in a copy of the ImageCollection."""
        copied = self.copy()
        sampled_tiles = list({img.tile for img in self})
        random.shuffle(sampled_tiles)
        sampled_tiles = sampled_tiles[:n]

        copied.images = [image for image in self if image.tile in sampled_tiles]
        return copied

    def sample_images(self, n: int) -> "ImageCollection":
        """Sample one or more images in a copy of the ImageCollection."""
        copied = self.copy()
        images = copied.images
        if n > len(images):
            raise ValueError(
                f"n ({n}) is higher than number of images in collection ({len(images)})"
            )
        sample = []
        for _ in range(n):
            random.shuffle(images)
            img = images.pop()
            sample.append(img)

        copied.images = sample

        return copied

    def __or__(self, collection: "ImageCollection") -> "ImageCollection":
        """Concatenate the collection with another collection."""
        return concat_image_collections([self, collection])

    def __iter__(self) -> Iterator[Image]:
        """Iterate over the images."""
        return iter(self.images)

    def __len__(self) -> int:
        """Number of images."""
        return len(self.images)

    def __getattr__(self, attr: str) -> Any:
        """Make iterable of metadata_attribute."""
        if attr in (self.metadata_attributes or {}):
            return self._metadata_attribute_collection_type(
                [getattr(img, attr) for img in self]
            )
        return super().__getattribute__(attr)

    def __getitem__(self, item: int | slice | Sequence[int | bool]) -> Image:
        """Select one Image by integer index, or multiple Images by slice, list of int."""
        if isinstance(item, int):
            return self.images[item]

        if isinstance(item, slice):
            copied = self.copy()
            copied.images = copied.images[item]
            return copied

        if isinstance(item, ImageCollection):

            def _get_from_single_element_list(lst: list[Any]) -> Any:
                if len(lst) != 1:
                    raise ValueError(lst)
                return next(iter(lst))

            copied = self.copy()
            copied._images = [
                _get_from_single_element_list(
                    [img2 for img2 in copied if img2.stem in img.path]
                )
                for img in item
            ]
            return copied

        copied = self.copy()
        if callable(item):
            item = [item(img) for img in copied]

        # check for base bool and numpy bool
        if all("bool" in str(type(x)) for x in item):
            copied.images = [img for x, img in zip(item, copied, strict=True) if x]

        else:
            copied.images = [copied.images[i] for i in item]
        return copied

    @property
    def dates(self) -> list[str]:
        """List of image dates."""
        return [img.date for img in self]

    @property
    def image_paths(self) -> list[str]:
        """List of image paths."""
        return [img.path for img in self]

    @property
    def images(self) -> list["Image"]:
        """List of images in the Collection."""
        if self._images is not None:
            return self._images
        # only fetch images when they are needed
        self._images = _get_images(
            list(self._df["image_path"]),
            all_file_paths=self._all_file_paths,
            df=self._df,
            image_class=self.image_class,
            band_class=self.band_class,
            masking=self.masking,
            **self._common_init_kwargs,
        )
        if self.masking is not None:
            images = []
            for image in self._images:
                try:
                    if not isinstance(image.mask, Band):
                        raise ValueError()
                    images.append(image)
                except ValueError:
                    continue
            self._images = images
            for image in self._images:
                image._bands = [band for band in image if band.band_id is not None]

        if self.metadata is not None:
            attributes_to_add = ["crs", "bounds"] + list(self.metadata_attributes)
            for img in self:
                for band in img:
                    for key in attributes_to_add:
                        try:
                            value = self.metadata[band.path][key]
                        except KeyError:
                            try:
                                value = self.metadata[key][band.path]
                            except KeyError:
                                continue
                        try:
                            setattr(band, key, value)
                        except Exception:
                            setattr(band, f"_{key}", value)

        self._images = [img for img in self if len(img)]

        if self._should_be_sorted:
            self._images = list(sorted(self._images))

        return self._images

    @property
    def _should_be_sorted(self) -> bool:
        """True if the ImageCollection has regexes that should make it sortable by date."""
        sort_group = "date"
        return (
            self.filename_patterns
            and any(
                sort_group in pat.groupindex
                and sort_group in _get_non_optional_groups(pat)
                for pat in self.filename_patterns
            )
            or self.image_patterns
            and any(
                sort_group in pat.groupindex
                and sort_group in _get_non_optional_groups(pat)
                for pat in self.image_patterns
            )
            or all(img.date is not None for img in self)
        )

    @images.setter
    def images(self, new_value: list["Image"]) -> list["Image"]:
        self._images = list(new_value)
        if not all(isinstance(x, Image) for x in self._images):
            raise TypeError("images should be a sequence of Image.")

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}({len(self)}, path='{self.path}')"

    def union_all(self) -> Polygon | MultiPolygon:
        """(Multi)Polygon representing the union of all image bounds."""
        return unary_union([img.union_all() for img in self])

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Total bounds for all Images combined."""
        return get_total_bounds([img.bounds for img in self])

    @property
    def crs(self) -> Any:
        """Common coordinate reference system of the Images."""
        if self._crs is not None:
            return self._crs
        self._crs = get_common_crs([img.crs for img in self])
        return self._crs

    def plot_pixels(
        self,
        by: str | list[str] | None = None,
        x_var: str = "date",
        y_label: str = "value",
        p: float = 0.95,
        ylim: tuple[float, float] | None = None,
        figsize: tuple[int] = (20, 8),
        rounding: int = 3,
    ) -> None:
        """Plot each individual pixel in a dotplot for all dates.

        Args:
            by: Band attributes to groupby. Defaults to "bounds" and "band_id"
                if all bands have no-None band_ids, otherwise defaults to "bounds".
            x_var: Attribute to use on the x-axis. Defaults to "date"
                if the ImageCollection is sortable by date, otherwise a range index.
                Can be set to "days_since_start".
            y_label: Label to use on the y-axis.
            p: p-value for the confidence interval.
            ylim: Limits of the y-axis.
            figsize: Figure size as tuple (width, height).
            rounding: rounding of title n

        """
        if by is None and all(band.band_id is not None for img in self for band in img):
            by = ["bounds", "band_id"]
        elif by is None:
            by = ["bounds"]

        alpha = 1 - p

        # for img in self:
        #     for band in img:
        #         band.load()

        for group_values, subcollection in self.groupby(by):
            print("group_values:", *group_values)

            if "date" in x_var and subcollection._should_be_sorted:
                subcollection._images = list(sorted(subcollection._images))

            y = np.array([band.values for img in subcollection for band in img])
            if "date" in x_var and subcollection._should_be_sorted:
                x = np.array(
                    [
                        datetime.datetime.strptime(band.date[:8], "%Y%m%d").date()
                        for img in subcollection
                        for band in img
                    ]
                )
                x = (
                    pd.to_datetime(
                        [band.date[:8] for img in subcollection for band in img]
                    )
                    - pd.Timestamp(np.min(x))
                ).days
            else:
                x = np.arange(0, len(y))

            mask = np.array(
                [
                    (
                        band.values.mask
                        if hasattr(band.values, "mask")
                        else np.full(band.values.shape, False)
                    )
                    for img in subcollection
                    for band in img
                ]
            )

            if x_var == "days_since_start":
                x = x - np.min(x)

            for i in range(y.shape[1]):
                for j in range(y.shape[2]):
                    this_y = y[:, i, j]

                    this_mask = mask[:, i, j]
                    this_x = x[~this_mask]
                    this_y = this_y[~this_mask]

                    if ylim:
                        condition = (this_y >= ylim[0]) & (this_y <= ylim[1])
                        this_y = this_y[condition]
                        this_x = this_x[condition]

                    coef, intercept = np.linalg.lstsq(
                        np.vstack([this_x, np.ones(this_x.shape[0])]).T,
                        this_y,
                        rcond=None,
                    )[0]
                    predicted = np.array([intercept + coef * x for x in this_x])

                    predicted_start = predicted[0]
                    predicted_end = predicted[-1]
                    predicted_change = predicted_end - predicted_start

                    # Degrees of freedom
                    dof = len(this_x) - 2

                    # 95% confidence interval
                    t_val = stats.t.ppf(1 - alpha / 2, dof)

                    # Mean squared error of the residuals
                    mse = np.sum((this_y - predicted) ** 2) / dof

                    # Calculate the standard error of predictions
                    pred_stderr = np.sqrt(
                        mse
                        * (
                            1 / len(this_x)
                            + (this_x - np.mean(this_x)) ** 2
                            / np.sum((this_x - np.mean(this_x)) ** 2)
                        )
                    )

                    # Calculate the confidence interval for predictions
                    ci_lower = predicted - t_val * pred_stderr
                    ci_upper = predicted + t_val * pred_stderr

                    fig = plt.figure(figsize=figsize)
                    ax = fig.add_subplot(1, 1, 1)

                    ax.scatter(this_x, this_y, color="#2c93db")
                    ax.plot(this_x, predicted, color="#e0436b")
                    ax.fill_between(
                        this_x,
                        ci_lower,
                        ci_upper,
                        color="#e0436b",
                        alpha=0.2,
                        label=f"{int(alpha*100)}% CI",
                    )
                    plt.title(
                        f"coef: {round(coef, int(np.log(1 / abs(coef))))}, "
                        f"pred change: {round(predicted_change, rounding)}, "
                        f"pred start: {round(predicted_start, rounding)}, "
                        f"pred end: {round(predicted_end, rounding)}"
                    )
                    plt.xlabel(x_var)
                    plt.ylabel(y_label)
                    plt.show()


def _get_all_regex_matches(xml_file: str, regexes: tuple[str]) -> tuple[str]:
    for regex in regexes:
        try:
            return re.search(regex, xml_file)
        except (TypeError, AttributeError):
            continue
    raise ValueError(
        f"Could not find processing_baseline info from {regexes} in {xml_file}"
    )


class Sentinel2Config:
    """Holder of Sentinel 2 regexes, band_ids etc."""

    image_regexes: ClassVar[str] = (config.SENTINEL2_IMAGE_REGEX,)
    filename_regexes: ClassVar[str] = (
        config.SENTINEL2_FILENAME_REGEX,
        config.SENTINEL2_CLOUD_FILENAME_REGEX,
    )
    metadata_attributes: ClassVar[
        dict[str, Callable | functools.partial | tuple[str]]
    ] = {
        "processing_baseline": functools.partial(
            _extract_regex_match_from_string,
            regexes=(r"<PROCESSING_BASELINE>(.*?)</PROCESSING_BASELINE>",),
        ),
        "cloud_coverage_percentage": "_get_cloud_coverage_percentage",
        "is_refined": functools.partial(
            _any_regex_matches, regexes=(r'<Image_Refining flag="REFINED">',)
        ),
        "boa_add_offset": "_get_boa_add_offset_dict",
    }
    all_bands: ClassVar[list[str]] = list(config.SENTINEL2_BANDS)
    rbg_bands: ClassVar[list[str]] = config.SENTINEL2_RBG_BANDS
    ndvi_bands: ClassVar[list[str]] = config.SENTINEL2_NDVI_BANDS
    l2a_bands: ClassVar[dict[str, int]] = config.SENTINEL2_L2A_BANDS
    l1c_bands: ClassVar[dict[str, int]] = config.SENTINEL2_L1C_BANDS
    masking: ClassVar[BandMasking] = BandMasking(
        band_id="SCL", values=(3, 8, 9, 10, 11)
    )

    def _get_cloud_coverage_percentage(self, xml_file: str) -> float:
        return float(
            _extract_regex_match_from_string(
                xml_file,
                (
                    r"<Cloud_Coverage_Assessment>([\d.]+)</Cloud_Coverage_Assessment>",
                    r"<CLOUDY_PIXEL_OVER_LAND_PERCENTAGE>([\d.]+)</CLOUDY_PIXEL_OVER_LAND_PERCENTAGE>",
                ),
            )
        )

    def _get_boa_add_offset_dict(self, xml_file: str) -> BandIdDict:
        pat = re.compile(
            r"""
    <BOA_ADD_OFFSET\s*
    band_id="(?P<band_id>\d+)"\s*
    >\s*(?P<value>-?\d+)\s*
    </BOA_ADD_OFFSET>
    """,
            flags=re.VERBOSE,
        )

        try:
            matches = [x.groupdict() for x in re.finditer(pat, xml_file)]
        except (TypeError, AttributeError, KeyError) as e:
            raise _RegexError(f"Could not find boa_add_offset info from {pat}") from e
        if not matches:
            raise _RegexError(f"Could not find boa_add_offset info from {pat}")
        return BandIdDict(
            pd.DataFrame(matches).set_index("band_id")["value"].astype(int).to_dict()
        )


class Sentinel2CloudlessConfig(Sentinel2Config):
    """Holder of regexes, band_ids etc. for Sentinel 2 cloudless mosaic."""

    image_regexes: ClassVar[str] = (config.SENTINEL2_MOSAIC_IMAGE_REGEX,)
    filename_regexes: ClassVar[str] = (config.SENTINEL2_MOSAIC_FILENAME_REGEX,)
    masking: ClassVar[None] = None
    all_bands: ClassVar[list[str]] = [
        x.replace("B0", "B") for x in Sentinel2Config.all_bands
    ]
    rbg_bands: ClassVar[list[str]] = [
        x.replace("B0", "B") for x in Sentinel2Config.rbg_bands
    ]
    ndvi_bands: ClassVar[list[str]] = [
        x.replace("B0", "B") for x in Sentinel2Config.ndvi_bands
    ]


class Sentinel2Band(Sentinel2Config, Band):
    """Band with Sentinel2 specific name variables and regexes."""


class Sentinel2Image(Sentinel2Config, Image):
    """Image with Sentinel2 specific name variables and regexes."""

    band_class: ClassVar[Sentinel2Band] = Sentinel2Band

    def ndvi(
        self,
        red_band: str = Sentinel2Config.ndvi_bands[0],
        nir_band: str = Sentinel2Config.ndvi_bands[1],
        copy: bool = True,
    ) -> NDVIBand:
        """Calculate the NDVI for the Image."""
        return super().ndvi(red_band=red_band, nir_band=nir_band, copy=copy)


class Sentinel2Collection(Sentinel2Config, ImageCollection):
    """ImageCollection with Sentinel2 specific name variables and path regexes."""

    image_class: ClassVar[Sentinel2Image] = Sentinel2Image
    band_class: ClassVar[Sentinel2Band] = Sentinel2Band

    def __init__(self, data: str | Path | Sequence[Image], **kwargs) -> None:
        """ImageCollection with Sentinel2 specific name variables and path regexes."""
        level = kwargs.get("level", NoLevel)
        if isinstance(level, type) and isinstance(level(), NoLevel):
            raise ValueError("Must specify level for Sentinel2Collection.")
        super().__init__(data=data, **kwargs)


class Sentinel2CloudlessBand(Sentinel2CloudlessConfig, Band):
    """Band for cloudless mosaic with Sentinel2 specific name variables and regexes."""


class Sentinel2CloudlessImage(Sentinel2CloudlessConfig, Sentinel2Image):
    """Image for cloudless mosaic with Sentinel2 specific name variables and regexes."""

    band_class: ClassVar[Sentinel2CloudlessBand] = Sentinel2CloudlessBand

    ndvi = Sentinel2Image.ndvi


class Sentinel2CloudlessCollection(Sentinel2CloudlessConfig, ImageCollection):
    """ImageCollection with Sentinel2 specific name variables and regexes."""

    image_class: ClassVar[Sentinel2CloudlessImage] = Sentinel2CloudlessImage
    band_class: ClassVar[Sentinel2Band] = Sentinel2CloudlessBand


def concat_image_collections(collections: Sequence[ImageCollection]) -> ImageCollection:
    """Union multiple ImageCollections together.

    Same as using the union operator |.
    """
    resolutions = {x.res for x in collections}
    if len(resolutions) > 1:
        raise ValueError(f"resoultion mismatch. {resolutions}")
    images = list(itertools.chain.from_iterable([x.images for x in collections]))
    levels = {x.level for x in collections}
    level = next(iter(levels)) if len(levels) == 1 else None
    first_collection = collections[0]

    out_collection = first_collection.__class__(
        images,
        level=level,
        band_class=first_collection.band_class,
        image_class=first_collection.image_class,
        **first_collection._common_init_kwargs,
    )
    out_collection._all_file_paths = list(
        sorted(
            set(itertools.chain.from_iterable([x._all_file_paths for x in collections]))
        )
    )
    return out_collection


def _get_gradient(band: Band, degrees: bool = False, copy: bool = True) -> Band:
    copied = band.copy() if copy else band
    if len(copied.values.shape) == 3:
        return np.array(
            [_slope_2d(arr, copied.res, degrees=degrees) for arr in copied.values]
        )
    elif len(copied.values.shape) == 2:
        return _slope_2d(copied.values, copied.res, degrees=degrees)
    else:
        raise ValueError("array must be 2 or 3 dimensional")


def to_xarray(
    array: np.ndarray, transform: Affine, crs: Any, name: str | None = None
) -> DataArray:
    """Convert the raster to  an xarray.DataArray."""
    if len(array.shape) == 2:
        height, width = array.shape
        dims = ["y", "x"]
    elif len(array.shape) == 3:
        height, width = array.shape[1:]
        dims = ["band", "y", "x"]
    else:
        raise ValueError(f"Array should be 2 or 3 dimensional. Got shape {array.shape}")

    coords = _generate_spatial_coords(transform, width, height)
    return xr.DataArray(
        array,
        coords=coords,
        dims=dims,
        name=name,
        attrs={"crs": crs},
    )


def _slope_2d(array: np.ndarray, res: int, degrees: int) -> np.ndarray:
    gradient_x, gradient_y = np.gradient(array, res, res)

    gradient = abs(gradient_x) + abs(gradient_y)

    if not degrees:
        return gradient

    radians = np.arctan(gradient)
    degrees = np.degrees(radians)

    assert np.max(degrees) <= 90

    return degrees


def _clip_loaded_array(
    arr: np.ndarray,
    bounds: tuple[int, int, int, int],
    transform: Affine,
    crs: Any,
    out_shape: tuple[int, int],
    **kwargs,
) -> np.ndarray:
    # xarray needs a numpy array of polygons
    bounds_arr: np.ndarray = GeoSeries([to_shapely(bounds)]).values
    try:

        while out_shape != arr.shape:
            arr = (
                to_xarray(
                    arr,
                    transform=transform,
                    crs=crs,
                )
                .rio.clip(bounds_arr, crs=crs, **kwargs)
                .to_numpy()
            )
            # bounds_arr = bounds_arr.buffer(0.0000001)
        return arr

    except NoDataInBounds:
        return np.array([])


def _fix_path(path: str) -> str:
    return (
        str(path).replace("\\", "/").replace(r"\"", "/").replace("//", "/").rstrip("/")
    )


def _get_all_file_paths(path: str) -> list[str]:
    if is_dapla():
        return list(sorted(set(_glob_func(path + "/**"))))
    else:
        return list(
            sorted(
                set(
                    _glob_func(path + "/**")
                    + _glob_func(path + "/**/**")
                    + _glob_func(path + "/**/**/**")
                    + _glob_func(path + "/**/**/**/**")
                    + _glob_func(path + "/**/**/**/**/**")
                )
            )
        )


def _get_images(
    image_paths: list[str],
    *,
    all_file_paths: list[str],
    df: pd.DataFrame,
    processes: int,
    image_class: type,
    band_class: type,
    bbox: GeoDataFrame | GeoSeries | Geometry | tuple[float] | None,
    masking: BandMasking | None,
    **kwargs,
) -> list[Image]:

    with joblib.Parallel(n_jobs=processes, backend="loky") as parallel:
        images = parallel(
            joblib.delayed(image_class)(
                path,
                df=df,
                all_file_paths=all_file_paths,
                masking=masking,
                band_class=band_class,
                **kwargs,
            )
            for path in image_paths
        )
    if bbox is not None:
        intersects_list = GeoSeries([img.union_all() for img in images]).intersects(
            to_shapely(bbox)
        )
        return [
            img
            for img, intersects in zip(images, intersects_list, strict=False)
            if intersects
        ]
    return images


class ArrayNotLoadedError(ValueError):
    """Arrays are not loaded."""


class PathlessImageError(ValueError):
    """'path' attribute is needed but instance has no path."""

    def __init__(self, instance: _ImageBase) -> None:
        """Initialise error class."""
        self.instance = instance

    def __str__(self) -> str:
        """String representation."""
        if self.instance._merged:
            what = "that have been merged"
        elif self.instance._from_array:
            what = "from arrays"
        elif self.instance._from_gdf:
            what = "from GeoDataFrames"
        else:
            raise ValueError(self.instance)

        return (
            f"{self.instance.__class__.__name__} instances {what} "
            "have no 'path' until they are written to file."
        )


def _date_is_within(
    date: str | None,
    date_ranges: DATE_RANGES_TYPE,
) -> bool:
    if date_ranges is None:
        return True

    if date is None:
        return False

    date = pd.Timestamp(date)

    if all(x is None or isinstance(x, str) for x in date_ranges):
        date_ranges = (date_ranges,)

    for date_range in date_ranges:
        date_min, date_max = date_range

        if date_min is not None:
            date_min = pd.Timestamp(date_min)
        if date_max is not None:
            date_max = pd.Timestamp(date_max)

        if (date_min is None or date >= date_min) and (
            date_max is None or date <= date_max
        ):
            return True

    return False


def _get_dtype_min(dtype: str | type) -> int | float:
    try:
        return np.iinfo(dtype).min
    except ValueError:
        return np.finfo(dtype).min


def _get_dtype_max(dtype: str | type) -> int | float:
    try:
        return np.iinfo(dtype).max
    except ValueError:
        return np.finfo(dtype).max


def _intesects(x, other) -> bool:
    return box(*x.bounds).intersects(other)


def _copy_and_add_df_parallel(
    i: tuple[Any, ...], group: pd.DataFrame, self: ImageCollection
) -> tuple[tuple[Any], ImageCollection]:
    copied = self.copy()
    copied.images = [
        img.copy() for img in group.drop_duplicates("_image_idx")["_image_instance"]
    ]
    if "band_id" in group:
        band_ids = set(group["band_id"].values)
        for img in copied.images:
            img._bands = [band for band in img if band.band_id in band_ids]

    return (i, copied)


def _get_bounds(bounds, bbox) -> None | Polygon:
    if bounds is None and bbox is None:
        return None
    elif bounds is not None and bbox is None:
        return to_shapely(bounds)  # .intersection(self.union_all())
    elif bounds is None and bbox is not None:
        return to_shapely(bbox)  # .intersection(self.union_all())
    else:
        return to_shapely(bounds).intersection(to_shapely(bbox))


def _get_single_value(values: tuple):
    if len(set(values)) == 1:
        return next(iter(values))
    else:
        return None


def _open_raster(path: str | Path) -> rasterio.io.DatasetReader:
    with opener(path) as file:
        return rasterio.open(file)


def _load_band(band: Band, **kwargs) -> None:
    band.load(**kwargs)


def _merge_by_band(collection: ImageCollection, **kwargs) -> Image:
    return collection.merge_by_band(**kwargs)


def _merge(collection: ImageCollection, **kwargs) -> Band:
    return collection.merge(**kwargs)


def _zonal_one_pair(i: int, poly: Polygon, band: Band, aggfunc, array_func, func_names):
    clipped = band.copy().load(bounds=poly)
    if not np.size(clipped.values):
        return _no_overlap_df(func_names, i, date=band.date)
    return _aggregate(clipped.values, array_func, aggfunc, func_names, band.date, i)


def array_buffer(arr: np.ndarray, distance: int) -> np.ndarray:
    """Buffer array points with the value 1 in a binary array.

    Args:
        arr: The array.
        distance: Number of array cells to buffer by.

    Returns:
        Array with buffered values.
    """
    if not np.all(np.isin(arr, (1, 0, True, False))):
        raise ValueError("Array must be all 0s and 1s or boolean.")

    dtype = arr.dtype

    structure = np.ones((2 * abs(distance) + 1, 2 * abs(distance) + 1))

    arr = np.where(arr, 1, 0)

    if distance > 0:
        return binary_dilation(arr, structure=structure).astype(dtype)
    elif distance < 0:

        return binary_erosion(arr, structure=structure).astype(dtype)


def get_cmap(arr: np.ndarray) -> LinearSegmentedColormap:

    # blue = [[i / 10 + 0.1, i / 10 + 0.1, 1 - (i / 10) + 0.1] for i in range(11)][1:]
    blue = [
        [0.1, 0.1, 1.0],
        [0.2, 0.2, 0.9],
        [0.3, 0.3, 0.8],
        [0.4, 0.4, 0.7],
        [0.6, 0.6, 0.6],
        [0.6, 0.6, 0.6],
        [0.7, 0.7, 0.7],
        [0.8, 0.8, 0.8],
    ]
    # gray = list(reversed([[i / 10 - 0.1, i / 10, i / 10 - 0.1] for i in range(11)][1:]))
    gray = [
        [0.6, 0.6, 0.6],
        [0.6, 0.6, 0.6],
        [0.6, 0.6, 0.6],
        [0.6, 0.6, 0.6],
        [0.6, 0.6, 0.6],
        [0.4, 0.7, 0.4],
        [0.3, 0.7, 0.3],
        [0.2, 0.8, 0.2],
    ]
    # gray = [[0.6, 0.6, 0.6] for i in range(10)]
    # green = [[0.2 + i/20, i / 10 - 0.1, + i/20] for i in range(11)][1:]
    green = [
        [0.25, 0.0, 0.05],
        [0.3, 0.1, 0.1],
        [0.35, 0.2, 0.15],
        [0.4, 0.3, 0.2],
        [0.45, 0.4, 0.25],
        [0.5, 0.5, 0.3],
        [0.55, 0.6, 0.35],
        [0.7, 0.9, 0.5],
    ]
    green = [
        [0.6, 0.6, 0.6],
        [0.4, 0.7, 0.4],
        [0.3, 0.8, 0.3],
        [0.25, 0.4, 0.25],
        [0.2, 0.5, 0.2],
        [0.10, 0.7, 0.10],
        [0, 0.9, 0],
    ]

    def get_start(arr):
        min_value = np.min(arr)
        if min_value < -0.75:
            return 0
        if min_value < -0.5:
            return 1
        if min_value < -0.25:
            return 2
        if min_value < 0:
            return 3
        if min_value < 0.25:
            return 4
        if min_value < 0.5:
            return 5
        if min_value < 0.75:
            return 6
        return 7

    def get_stop(arr):
        max_value = np.max(arr)
        if max_value <= 0.05:
            return 0
        if max_value < 0.175:
            return 1
        if max_value < 0.25:
            return 2
        if max_value < 0.375:
            return 3
        if max_value < 0.5:
            return 4
        if max_value < 0.75:
            return 5
        return 6

    cmap_name = "blue_gray_green"

    start = get_start(arr)
    stop = get_stop(arr)
    blue = blue[start]
    gray = gray[start]
    # green = green[start]
    green = green[stop]

    # green[0] = np.arange(0, 1, 0.1)[::-1][stop]
    # green[1] = np.arange(0, 1, 0.1)[stop]
    # green[2] = np.arange(0, 1, 0.1)[::-1][stop]

    print(green)
    print(start, stop)
    print("blue gray green")
    print(blue)
    print(gray)
    print(green)

    # Define the segments of the colormap
    cdict = {
        "red": [
            (0.0, blue[0], blue[0]),
            (0.3, gray[0], gray[0]),
            (0.7, gray[0], gray[0]),
            (1.0, green[0], green[0]),
        ],
        "green": [
            (0.0, blue[1], blue[1]),
            (0.3, gray[1], gray[1]),
            (0.7, gray[1], gray[1]),
            (1.0, green[1], green[1]),
        ],
        "blue": [
            (0.0, blue[2], blue[2]),
            (0.3, gray[2], gray[2]),
            (0.7, gray[2], gray[2]),
            (1.0, green[2], green[2]),
        ],
    }

    return LinearSegmentedColormap(cmap_name, segmentdata=cdict, N=50)
