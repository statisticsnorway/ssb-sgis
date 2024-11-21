import datetime
import functools
import glob
import itertools
import os
import random
import re
import time
from abc import abstractmethod
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
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
from pandas.api.types import is_dict_like
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
except ImportError:
    pass


try:
    from google.auth import exceptions
except ImportError:

    class exceptions:
        """Placeholder."""

        class RefreshError(Exception):
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
    from xarray import DataArray
    from xarray import Dataset
    from xarray import combine_by_coords
except ImportError:

    class DataArray:
        """Placeholder."""

    class Dataset:
        """Placeholder."""

    def combine_by_coords(*args, **kwargs) -> None:
        raise ImportError("xarray")


from ..geopandas_tools.bounds import get_total_bounds
from ..geopandas_tools.conversion import to_bbox
from ..geopandas_tools.conversion import to_gdf
from ..geopandas_tools.conversion import to_geoseries
from ..geopandas_tools.conversion import to_shapely
from ..geopandas_tools.general import get_common_crs
from ..helpers import _fix_path
from ..helpers import get_all_files
from ..helpers import get_numpy_func
from ..helpers import is_method
from ..helpers import is_property
from ..io._is_dapla import is_dapla
from ..io.opener import opener
from . import sentinel_config as config
from .base import _array_to_geojson
from .base import _gdf_to_arr
from .base import _get_res_from_bounds
from .base import _get_shape_from_bounds
from .base import _get_transform_from_bounds
from .base import _res_as_tuple
from .base import get_index_mapper
from .indices import ndvi
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
    "all_bands",
    "crs",
    "masking",
    "_merged",
    "date",
]

_LOAD_COUNTER: int = 0


def _get_child_paths_threaded(data: Sequence[str]) -> set[str]:
    with ThreadPoolExecutor() as executor:
        all_paths: Iterator[set[str]] = executor.map(_ls_func, data)
    return set(itertools.chain.from_iterable(all_paths))


@dataclass
class PixelwiseResults:
    """Container of results from pixelwise operation to be converted."""

    row_indices: np.ndarray
    col_indices: np.ndarray
    results: list[Any]
    res: int | tuple[int, int]
    bounds: tuple[float, float, float, float]
    shape: tuple[int, int]
    crs: Any
    nodata: int | float | None

    def to_tuple(self) -> tuple[int, int, Any]:
        """Return 3-length tuple of row indices, column indices and pixelwise results."""
        return self.row_indices, self.col_indices, self.results

    def to_dict(self) -> dict[tuple[int, int], Any]:
        """Return dictionary with row and column indices as keys and pixelwise results as values."""
        return {
            (int(row), int(col)): value
            for row, col, value in zip(
                self.row_indices, self.col_indices, self.results, strict=True
            )
        }

    def to_geopandas(self, column: str = "value") -> GeoDataFrame:
        """Return GeoDataFrame with pixel geometries and values from the pixelwise operation."""
        minx, miny = self.bounds[:2]
        resx, resy = _res_as_tuple(self.res)

        minxs = np.full(self.row_indices.shape, minx) + (minx * self.row_indices * resx)
        minys = np.full(self.col_indices.shape, miny) + (miny * self.col_indices * resy)
        maxxs = minxs + resx
        maxys = minys + resy

        return GeoDataFrame(
            {
                column: self.results,
                "geometry": [
                    box(minx, miny, maxx, maxy)
                    for minx, miny, maxx, maxy in zip(
                        minxs, minys, maxxs, maxys, strict=True
                    )
                ],
            },
            index=[self.row_indices, self.col_indices],
            crs=self.crs,
        )

    def to_numpy(self) -> np.ndarray | tuple[np.ndarray, ...]:
        """Reshape pixelwise results to 2d numpy arrays in the shape of the full arrays of the image bands."""
        try:
            n_out_arrays = len(next(iter(self.results)))
        except TypeError:
            n_out_arrays = 1

        out_arrays = [
            np.full(self.shape, self.nodata).astype(np.float64)
            for _ in range(n_out_arrays)
        ]

        for row, col, these_results in zip(
            self.row_indices, self.col_indices, self.results, strict=True
        ):
            if these_results is None:
                continue
            for i, arr in enumerate(out_arrays):
                try:
                    arr[row, col] = these_results[i]
                except TypeError:
                    arr[row, col] = these_results

        for i, array in enumerate(out_arrays):
            all_are_integers = np.all(np.mod(array, 1) == 0)
            if all_are_integers:
                out_arrays[i] = array.astype(int)

        if len(out_arrays) == 1:
            return out_arrays[0]

        return tuple(out_arrays)


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
            collection: Ungrouped ImageCollection. Used to pass attributes to outputs.
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
        return f"{self.__class__.__name__}({len(self)}, by={self.by})"


@dataclass(frozen=True)
class BandMasking:
    """Frozen dict with forced keys."""

    band_id: str
    values: Sequence[int] | dict[int, Any]

    def __getitem__(self, item: str) -> Any:
        """Index into attributes to mimick dict."""
        return getattr(self, item)


class None_:
    """Default None for args that are not allowed to be None."""

    def __new__(cls) -> None:
        """Always returns None."""
        return None


class _ImageBase:
    image_regexes: ClassVar[str | None] = (DEFAULT_IMAGE_REGEX,)
    filename_regexes: ClassVar[str | tuple[str]] = (DEFAULT_FILENAME_REGEX,)
    metadata_attributes: ClassVar[dict | None] = None
    masking: ClassVar[BandMasking | None] = None

    def __init__(self, *, metadata=None, bbox=None, **kwargs) -> None:

        self._bounds = None
        self._path = None
        self._bbox = to_bbox(bbox) if bbox is not None else None

        self.metadata_attributes = self.metadata_attributes or {}

        if metadata is not None:
            self.metadata = self._metadata_to_nested_dict(metadata)
        else:
            self.metadata = {}

        self.image_patterns = self._compile_regexes("image_regexes")
        self.filename_patterns = self._compile_regexes("filename_regexes")

        for key, value in kwargs.items():
            error_obj = ValueError(
                f"{self.__class__.__name__} got an unexpected keyword argument '{key}'"
            )
            if key in ALLOWED_INIT_KWARGS and key in dir(self):
                self._safe_setattr(key, value, error_obj)
            else:
                raise error_obj

        # attributes for debugging
        self._metadata_from_xml = False
        self._merged = False
        self._from_array = False
        self._from_geopandas = False

    def _safe_setattr(
        self, key: str, value: Any, error_obj: Exception | None = None
    ) -> None:
        if is_property(self, key):
            setattr(self, f"_{key}", value)
        elif is_method(self, key):
            if error_obj is None:
                raise AttributeError(f"Cannot set method '{key}'.")
            raise error_obj
        else:
            setattr(self, key, value)

    def _compile_regexes(self, regex_attr: str) -> tuple[re.Pattern]:
        regexes: tuple[str] | str = getattr(self, regex_attr)
        if not regexes:
            return ()
        if isinstance(regexes, str):
            regexes = (regexes,)
        return tuple(re.compile(regexes, flags=re.VERBOSE) for regexes in regexes)

    @staticmethod
    def _metadata_to_nested_dict(
        metadata: str | Path | os.PathLike | dict | pd.DataFrame | None,
    ) -> dict[str, dict[str, Any]]:
        """Construct metadata dict from dictlike, DataFrame or file path.

        Extract metadata value:
        >>> self.metadata[self.path]['cloud_cover_percentage']
        """
        if isinstance(metadata, (str | Path | os.PathLike)):
            metadata = _read_parquet_func(metadata)

        if isinstance(metadata, pd.DataFrame):

            def is_scalar(x) -> bool:
                """Check if scalar because 'truth value of Series is ambigous'."""
                return not hasattr(x, "__len__") or len(x) <= 1

            def na_to_none(x) -> None:
                """Convert to None rowwise because pandas doesn't always."""
                return x if not (is_scalar(x) and pd.isna(x)) else None

            # to nested dict because pandas indexing gives rare KeyError with long strings
            return {
                _fix_path(path): {
                    attr: na_to_none(value) for attr, value in row.items()
                }
                for path, row in metadata.iterrows()
            }
        elif is_dict_like(metadata):
            return {_fix_path(path): value for path, value in metadata.items()}

        # try to allow custom types with dict-like indexing
        return metadata

    @property
    def _common_init_kwargs(self) -> dict:
        return {
            "processes": self.processes,
            "res": self.res,
            "bbox": self._bbox,
            "nodata": self.nodata,
            "metadata": self.metadata,
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
        # if self._res is None:
        #     if self.has_array:
        #         self._res = _get_res_from_bounds(self.bounds, self.values.shape)
        #     else:
        #         with opener(self.path) as file:
        #             with rasterio.open(file) as src:
        #                 self._res = src.res
        return self._res

    @abstractmethod
    def union_all(self) -> Polygon | MultiPolygon:
        pass

    def assign(self, **kwargs) -> "_ImageBase":
        for key, value in kwargs.items():
            self._safe_setattr(key, value)
        return self

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
        if isinstance(self, Band):
            for pat in patterns:
                try:
                    return _get_first_group_match(
                        pat, str(Path(self.path).parent.name)
                    )[group]
                except (TypeError, KeyError):
                    pass
        if not any(group in _get_non_optional_groups(pat) for pat in patterns):
            return None
        band_text = (
            f" or {Path(self.path).parent.name!s}" if isinstance(self, Band) else ""
        )
        raise ValueError(
            f"Couldn't find group '{group}' in name {self.name}{band_text} with regex patterns {patterns}"
        )

    def _create_metadata_df(self, file_paths: Sequence[str]) -> pd.DataFrame:
        """Create a dataframe with file paths and image paths that match regexes.

        Used in __init__ to select relevant paths fast.
        """
        df = pd.DataFrame({"file_path": list(file_paths)})

        df["file_name"] = df["file_path"].apply(lambda x: Path(x).name)

        df["image_path"] = df["file_path"].apply(
            lambda x: _fix_path(str(Path(x).parent))
        )

        if not len(df):
            return df

        df = df[~df["file_path"].isin(df["image_path"])]

        if self.filename_patterns:
            df = _get_regexes_matches_for_df(df, "file_name", self.filename_patterns)

            if not len(df):
                return df

            grouped = df.drop_duplicates("image_path").set_index("image_path")
            for col in ["file_path", "file_name"]:
                if col in df:
                    grouped[col] = df.groupby("image_path")[col].apply(tuple)

            grouped = grouped.reset_index()
        else:
            df["file_path"] = df.groupby("image_path")["file_path"].apply(tuple)
            df["file_name"] = df.groupby("image_path")["file_name"].apply(tuple)
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

    def intersects(
        self, geometry: GeoDataFrame | GeoSeries | Geometry | tuple | _ImageBase
    ) -> bool:
        if hasattr(geometry, "crs") and not pyproj.CRS(self.crs).equals(
            pyproj.CRS(geometry.crs)
        ):
            raise ValueError(f"crs mismatch: {self.crs} and {geometry.crs}")
        return self.union_all().intersects(to_shapely(geometry))

    def union_all(self) -> Polygon:
        try:
            return box(*self.bounds)
        except TypeError:
            return Polygon()

    @property
    def centroid(self) -> Point:
        """Centerpoint of the object."""
        return self.union_all().centroid

    @property
    def year(self) -> str:
        if hasattr(self, "_year") and self._year:
            return self._year
        return str(self.date)[:4]

    @property
    def month(self) -> str:
        if hasattr(self, "_month") and self._month:
            return self._month
        return str(self.date).replace("-", "").replace("/", "")[4:6]

    @property
    def name(self) -> str | None:
        if hasattr(self, "_name") and self._name is not None:
            return self._name
        try:
            return Path(self.path).name
        except (ValueError, AttributeError, TypeError):
            return None

    @name.setter
    def name(self, value) -> None:
        self._name = value

    @property
    def stem(self) -> str | None:
        try:
            return Path(self.path).stem
        except (AttributeError, ValueError, TypeError):
            return None

    @property
    def level(self) -> str:
        return self._name_regex_searcher("level", self.image_patterns)

    def _get_metadata_attributes(self, metadata_attributes: dict) -> dict:
        """Search through xml files for missing metadata attributes."""
        self._metadata_from_xml = True

        missing_metadata_attributes = {
            attr: constructor_func
            for attr, constructor_func in metadata_attributes.items()
            if not hasattr(self, attr) or getattr(self, attr) is None
        }

        nonmissing_metadata_attributes = {
            attr: getattr(self, attr)
            for attr in metadata_attributes
            if attr not in missing_metadata_attributes
        }

        if not missing_metadata_attributes:
            return nonmissing_metadata_attributes

        # read all xml content once
        file_contents: dict[str, str] = {}
        for path in self._all_file_paths:
            if ".xml" not in path:
                continue
            with _open_func(path, "rb") as file:
                file_contents[path] = file.read().decode("utf-8")

        def is_last_xml(i: int) -> bool:
            return i == len(file_contents) - 1

        for attr, value in missing_metadata_attributes.items():
            results = None
            for i, file_content in enumerate(file_contents.values()):
                if isinstance(value, str) and value in dir(self):
                    # method or a hardcoded value
                    value: Callable | Any = getattr(self, value)

                if callable(value):
                    try:
                        results = value(file_content)
                    except _RegexError as e:
                        if is_last_xml(i):
                            raise e.__class__(self.path, list(file_contents), e) from e
                        continue
                    if results is not None:
                        break
                elif (
                    isinstance(value, str)
                    or hasattr(value, "__iter__")
                    and all(isinstance(x, str | re.Pattern) for x in value)
                ):
                    try:
                        results = _extract_regex_match_from_string(file_content, value)
                    except _RegexError as e:
                        if is_last_xml(i):
                            raise e
                elif value is not None:
                    results = value
                    break

            missing_metadata_attributes[attr] = results

        return missing_metadata_attributes | nonmissing_metadata_attributes

    def _to_xarray(self, array: np.ndarray, transform: Affine) -> DataArray:
        """Convert the raster to  an xarray.DataArray."""
        attrs = {"crs": self.crs}
        for attr in set(self.metadata_attributes).union({"date"}):
            try:
                attrs[attr] = getattr(self, attr)
            except Exception:
                pass

        if len(array.shape) == 2:
            height, width = array.shape
            dims = ["y", "x"]
        elif len(array.shape) == 3:
            height, width = array.shape[1:]
            dims = ["band", "y", "x"]
        elif not any(dim for dim in array.shape):
            DataArray(
                name=self.name or self.__class__.__name__,
                attrs=attrs,
            )
        else:
            raise ValueError(
                f"Array should be 2 or 3 dimensional. Got shape {array.shape}"
            )

        coords = _generate_spatial_coords(transform, width, height)

        return DataArray(
            array,
            coords=coords,
            dims=dims,
            name=self.name or self.__class__.__name__,
            attrs=attrs,
        )


class Band(_ImageBandBase):
    """Band holding a single 2 dimensional array representing an image band."""

    cmap: ClassVar[str | None] = None

    @classmethod
    def from_geopandas(
        cls,
        gdf: GeoDataFrame | GeoSeries,
        *,
        res: int | None = None,
        out_shape: tuple[int, int] | None = None,
        bounds: Any | None = None,
        fill: int = 0,
        all_touched: bool = False,
        merge_alg: Callable = MergeAlg.replace,
        default_value: int = 1,
        dtype: Any | None = None,
        **kwargs,
    ) -> None:
        """Create Band from a GeoDataFrame."""
        if bounds is not None:
            bounds = to_bbox(bounds)

        if out_shape == (0,):
            arr = np.array([])
        else:
            arr = _gdf_to_arr(
                gdf,
                res=res,
                bounds=bounds,
                fill=fill,
                all_touched=all_touched,
                merge_alg=merge_alg,
                default_value=default_value,
                dtype=dtype,
                out_shape=out_shape,
            )
        if bounds is None:
            bounds = gdf.total_bounds

        obj = cls(arr, crs=gdf.crs, bounds=bounds, **kwargs)
        obj._from_geopandas = True
        return obj

    def __init__(
        self,
        data: str | np.ndarray | None = None,
        res: int | None_ = None_,
        crs: Any | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        nodata: int | None = None,
        mask: "Band | None" = None,
        processes: int = 1,
        name: str | None = None,
        band_id: str | None = None,
        cmap: str | None = None,
        all_file_paths: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Band initialiser."""
        if data is None:
            # allowing 'path' to replace 'data' as argument
            # to make the print repr. valid as initialiser
            if "path" not in kwargs:
                raise TypeError("Must specify either 'data' or 'path'.")
            data = kwargs.pop("path")

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

        if isinstance(data, np.ndarray):
            if self._bounds is None:
                raise ValueError("Must specify bounds when data is an array.")
            if not (res is None or (callable(res) and res() is None)):
                # if not (res is None or (callable(res) and res() is None)) and _res_as_tuple(
                #     res
                # ) != _get_res_from_bounds(self._bounds, data.shape):
                raise ValueError(
                    f"Cannot specify 'res' when data is an array. {res} and {_get_res_from_bounds(self._bounds, data.shape)}"
                )
            self._crs = crs
            self.transform = _get_transform_from_bounds(self._bounds, shape=data.shape)
            self._from_array = True
            self.values = data

            self._res = _get_res_from_bounds(self._bounds, self.values.shape)

        elif not isinstance(data, (str | Path | os.PathLike)):
            raise TypeError(
                "'data' must be string, Path-like or numpy.ndarray. "
                f"Got {type(data)}"
            )
        else:
            self._path = _fix_path(str(data))
            self._res = res if not (callable(res) and res() is None) else None

        if cmap is not None:
            self.cmap = cmap
        self._name = name
        self._band_id = band_id
        self.processes = processes

        if self._all_file_paths:
            self._all_file_paths = {_fix_path(path) for path in self._all_file_paths}
            parent = _fix_path(Path(self.path).parent)
            self._all_file_paths = {
                path for path in self._all_file_paths if parent in path
            }

        if self.metadata:
            if self.path is not None:
                self.metadata = {
                    key: value
                    for key, value in self.metadata.items()
                    if key == self.path
                }
            this_metadata = self.metadata[self.path]
            for key, value in this_metadata.items():
                if key in dir(self):
                    setattr(self, f"_{key}", value)
                else:
                    setattr(self, key, value)

        elif self.metadata_attributes and self.path is not None:
            if self._all_file_paths is None:
                self._all_file_paths = _get_all_file_paths(str(Path(self.path).parent))
            for key, value in self._get_metadata_attributes(
                self.metadata_attributes
            ).items():
                setattr(self, key, value)

    def __lt__(self, other: "Band") -> bool:
        """Makes Bands sortable by band_id."""
        return self.band_id < other.band_id

    def value_counts(self) -> pd.Series:
        """Value count of each value of the band's array."""
        try:
            values = self.values.data[self.values.mask == False]
        except AttributeError:
            values = self.values
        unique_values, counts = np.unique(values, return_counts=True)
        return pd.Series(counts, index=unique_values).sort_values(ascending=False)

    @property
    def values(self) -> np.ndarray:
        """The numpy array, if loaded."""
        if self._values is None:
            raise _ArrayNotLoadedError("array is not loaded.")
        return self._values

    @values.setter
    def values(self, new_val):
        if isinstance(new_val, np.ndarray):
            self._values = new_val
        else:
            self._values = self._to_numpy(new_val)

    @property
    def band_id(self) -> str:
        """Band id."""
        if self._band_id is not None:
            return self._band_id
        return self._name_regex_searcher("band", self.filename_patterns)

    @property
    def height(self) -> int:
        """Pixel heigth of the image band."""
        try:
            return self.values.shape[-2]
        except IndexError:
            return 0

    @property
    def width(self) -> int:
        """Pixel width of the image band."""
        try:
            return self.values.shape[-1]
        except IndexError:
            return 0

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
    def crs(self) -> pyproj.CRS | None:
        """Coordinate reference system."""
        if self._crs is None:
            self._add_crs_and_bounds()
        return pyproj.CRS(self._crs)

    @property
    def bounds(self) -> tuple[int, int, int, int] | None:
        """Bounds as tuple (minx, miny, maxx, maxy)."""
        if self._bounds is None:
            self._add_crs_and_bounds()
        return self._bounds

    def _add_crs_and_bounds(self) -> None:
        with opener(self.path) as file:
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
        df = copied.to_geopandas(column).loc[lambda x: x[column] == 1]
        df[column] = f"largest_{n}"
        return df

    def get_n_smallest(
        self, n: int, precision: float = 0.000001, column: str = "value"
    ) -> GeoDataFrame:
        """Get the lowest values of the array as polygons in a GeoDataFrame."""
        copied = self.copy()
        value_must_be_at_least = np.sort(np.ravel(copied.values))[n] - (precision or 0)
        copied._values = np.where(copied.values <= value_must_be_at_least, 1, 0)
        df = copied.to_geopandas(column).loc[lambda x: x[column] == 1]
        df[column] = f"smallest_{n}"
        return df

    def clip(
        self,
        mask: GeoDataFrame | GeoSeries | Polygon | MultiPolygon,
    ) -> "Band":
        """Clip band values to geometry mask while preserving bounds."""
        if not self.height or not self.width:
            return self

        fill: int = self.nodata or 0

        mask_array: np.ndarray = Band.from_geopandas(
            gdf=to_gdf(mask)[["geometry"]],
            default_value=1,
            fill=fill,
            out_shape=self.values.shape,
            bounds=mask,
        ).values

        is_not_polygon = mask_array == fill

        if isinstance(self.values, np.ma.core.MaskedArray):
            self._values.mask |= is_not_polygon
        else:
            self._values = np.ma.array(
                self.values, mask=is_not_polygon, fill_value=self.nodata
            )

        return self

    def load(
        self,
        bounds: tuple | Geometry | GeoDataFrame | GeoSeries | None = None,
        indexes: int | tuple[int] | None = None,
        masked: bool = True,
        file_system=None,
        **kwargs,
    ) -> "Band":
        """Load and potentially clip the array.

        The array is stored in the 'values' property.
        """
        global _LOAD_COUNTER
        _LOAD_COUNTER += 1

        _masking = kwargs.pop("_masking", self.masking)

        bounds_was_none = bounds is None

        bounds = _get_bounds(bounds, self._bbox, self.union_all())

        should_return_empty: bool = bounds is not None and bounds.area == 0
        if should_return_empty:
            self._values = np.array([])
            self._bounds = None
            self.transform = None
            # activate setter
            self.values = self._values

            return self

        if self.has_array and bounds_was_none:
            return self

        if bounds is not None:
            minx, miny, maxx, maxy = to_bbox(bounds)
            # bounds = (int(minx), int(miny), math.ceil(maxx), math.ceil(maxy))
            bounds = minx, miny, maxx, maxy

        if indexes is None:
            indexes = 1

        # as tuple to ensure we get 3d array
        _indexes: tuple[int] = (indexes,) if isinstance(indexes, int) else indexes

        # allow setting a fixed out_shape for the array, in order to make mask same shape as values
        out_shape = kwargs.pop("out_shape", None)

        if self.has_array and [int(x) for x in bounds] != [int(x) for x in self.bounds]:
            raise ValueError(
                "Cannot re-load array with different bounds. "
                "Use .copy() to read with different bounds. "
                "Or .clip(mask) to clip.",
                self,
                self.values.shape,
                [int(x) for x in bounds],
                [int(x) for x in self.bounds],
            )

        with opener(self.path, file_system=file_system) as f:
            with rasterio.open(f, nodata=self.nodata) as src:
                self._res = src.res if not self.res else self.res
                if self.nodata is None or np.isnan(self.nodata):
                    self.nodata = src.nodata
                else:
                    dtype_min_value = _get_dtype_min(src.dtypes[0])
                    dtype_max_value = _get_dtype_max(src.dtypes[0])
                    if self.nodata > dtype_max_value or self.nodata < dtype_min_value:
                        src._dtypes = tuple(
                            rasterio.dtypes.get_minimum_dtype(self.nodata)
                            for _ in range(len(_indexes))
                        )

                if bounds is None:
                    if self._res != src.res:
                        if out_shape is None:
                            out_shape = _get_shape_from_bounds(
                                to_bbox(src.bounds), self.res, indexes
                            )
                        self.transform = _get_transform_from_bounds(
                            to_bbox(src.bounds), shape=out_shape
                        )
                    else:
                        self.transform = src.transform

                    values = src.read(
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
                        out_shape = _get_shape_from_bounds(bounds, self.res, indexes)

                    values = src.read(
                        indexes=indexes,
                        window=window,
                        boundless=False,
                        out_shape=out_shape,
                        masked=masked,
                        **kwargs,
                    )

                    assert out_shape == values.shape, (
                        out_shape,
                        values.shape,
                    )

                    width, height = values.shape[-2:]

                    if width and height:
                        self.transform = rasterio.transform.from_bounds(
                            *bounds, width, height
                        )

                if self.nodata is not None and not np.isnan(self.nodata):
                    if isinstance(values, np.ma.core.MaskedArray):
                        values.data[values.data == src.nodata] = self.nodata
                    else:
                        values[values == src.nodata] = self.nodata

        if _masking and not isinstance(values, np.ma.core.MaskedArray):
            mask_arr = _read_mask_array(self, bounds=bounds)
            values = np.ma.array(values, mask=mask_arr, fill_value=self.nodata)
        elif _masking:
            mask_arr = _read_mask_array(self, bounds=bounds)
            values.mask |= mask_arr

        if bounds is not None:
            self._bounds = to_bbox(bounds)

        self._values = values
        # trigger the setter
        self.values = values

        return self

    @property
    def has_array(self) -> bool:
        """Whether the array is loaded."""
        try:
            if not isinstance(self.values, (np.ndarray | DataArray)):
                raise ValueError()
            return True
        except ValueError:  # also catches _ArrayNotLoadedError
            return False

    def write(
        self,
        path: str | Path,
        driver: str = "GTiff",
        compress: str = "LZW",
        file_system=None,
        **kwargs,
    ) -> None:
        """Write the array as an image file."""
        if not hasattr(self, "_values"):
            raise ValueError(
                "Can only write image band from Band constructed from array."
            )

        if self.crs is None:
            raise ValueError("Cannot write None crs to image.")

        if self.nodata:
            # TODO take out .data if masked?
            values_with_nodata = np.concatenate(
                [self.values.flatten(), np.array([self.nodata])]
            )
        else:
            values_with_nodata = self.values
        profile = {
            "driver": driver,
            "compress": compress,
            "dtype": rasterio.dtypes.get_minimum_dtype(values_with_nodata),
            "crs": self.crs,
            "transform": self.transform,
            "nodata": self.nodata,
            "count": 1 if len(self.values.shape) == 2 else self.values.shape[0],
            "height": self.height,
            "width": self.width,
        } | kwargs

        with opener(path, "wb", file_system=file_system) as f:
            with rasterio.open(f, "w", **profile) as dst:

                if dst.nodata is None:
                    dst.nodata = _get_dtype_min(dst.dtypes[0])

                if (
                    isinstance(self.values, np.ma.core.MaskedArray)
                    and dst.nodata is not None
                ):
                    self.values.data[np.isnan(self.values.data)] = dst.nodata
                    self.values.data[self.values.mask] = dst.nodata

                if len(self.values.shape) == 2:
                    dst.write(self.values, indexes=1)
                else:
                    for i in range(self.values.shape[0]):
                        dst.write(self.values[i], indexes=i + 1)

                if isinstance(self.values, np.ma.core.MaskedArray):
                    dst.write_mask(self.values.mask)

        self._path = _fix_path(str(path))

    def apply(self, func: Callable, **kwargs) -> "Band":
        """Apply a function to the Band."""
        results = func(self, **kwargs)
        if isinstance(results, Band):
            return results
        self.values = results
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

        >>> band.gradient(copy=False)
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

    def to_geopandas(self, column: str = "value", dropna: bool = True) -> GeoDataFrame:
        """Create a GeoDataFrame from the image Band.

        Args:
            column: Name of resulting column that holds the raster values.
            dropna: Whether to remove values that are NA or equal to the nodata
                value.

        Returns:
            A GeoDataFrame with a geometry column and array values.
        """
        if not hasattr(self, "_values"):
            raise ValueError("Array is not loaded.")

        if isinstance(self.values, np.ma.core.MaskedArray):
            self.values.data[self.values.mask] = self.nodata or 0
        if self.values.shape[0] == 0:
            df = GeoDataFrame({"geometry": []}, crs=self.crs)
        else:
            df = GeoDataFrame(
                pd.DataFrame(
                    _array_to_geojson(
                        self.values, self.transform, processes=self.processes
                    ),
                    columns=[column, "geometry"],
                ),
                geometry="geometry",
                crs=self.crs,
            )

        if dropna:
            return df[(df[column] != self.nodata) & (df[column].notna())]
        return df

    def to_xarray(self) -> DataArray:
        """Convert the raster to an xarray.DataArray."""
        return self._to_xarray(
            self.values,
            transform=self.transform,
            # name=self.name or self.__class__.__name__.lower(),
        )

    def to_numpy(self) -> np.ndarray | np.ma.core.MaskedArray:
        """Convert the raster to a numpy.ndarray."""
        return self._to_numpy(self.values).copy()

    def _to_numpy(
        self, arr: np.ndarray | DataArray, masked: bool = True
    ) -> np.ndarray | np.ma.core.MaskedArray:
        if not isinstance(arr, np.ndarray):
            mask_arr = None
            if masked:
                try:
                    mask_arr = arr.isnull().values
                except AttributeError:
                    pass
            try:
                arr = arr.to_numpy()
            except AttributeError:
                arr = arr.values
            if mask_arr is not None:
                arr = np.ma.array(arr, mask=mask_arr, fill_value=self.nodata)

        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        if (
            masked
            and not isinstance(arr, np.ma.core.MaskedArray)
            and mask_arr is not None
        ):
            arr = np.ma.array(arr, mask=mask_arr, fill_value=self.nodata)

        return arr

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


def median_as_int_and_minimum_dtype(arr: np.ndarray) -> np.ndarray:
    arr = np.median(arr, axis=0).astype(int)
    min_dtype = rasterio.dtypes.get_minimum_dtype(arr)
    return arr.astype(min_dtype)


class Image(_ImageBandBase):
    """Image consisting of one or more Bands."""

    band_class: ClassVar[Band] = Band

    def __init__(
        self,
        data: str | Path | Sequence[Band] | None = None,
        res: int | None_ = None_,
        mask: "Band | None" = None,
        processes: int = 1,
        df: pd.DataFrame | None = None,
        nodata: int | None = None,
        all_file_paths: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Image initialiser."""
        if data is None:
            # allowing 'bands' to replace 'data' as argument
            # to make the print repr. valid as initialiser
            if "bands" not in kwargs:
                raise TypeError("Must specify either 'data' or 'bands'.")
            data = kwargs.pop("bands")

        super().__init__(**kwargs)

        self.nodata = nodata
        self.processes = processes
        self._crs = None
        self._bands = None
        self._mask = mask

        if isinstance(data, Band):
            data = [data]

        if hasattr(data, "__iter__") and all(isinstance(x, Band) for x in data):
            self._construct_image_from_bands(data, res)
            return
        elif not isinstance(data, (str | Path | os.PathLike)):
            raise TypeError(
                f"'data' must be string, Path-like or a sequence of Band. Got {data}"
            )

        self._res = res if not (callable(res) and res() is None) else None
        self._path = _fix_path(data)

        if all_file_paths is None and self.path:
            self._all_file_paths = _get_all_file_paths(self.path)
        elif self.path:
            name = Path(self.path).name
            all_file_paths = {_fix_path(x) for x in all_file_paths if name in x}
            self._all_file_paths = {x for x in all_file_paths if self.path in x}
        else:
            self._all_file_paths = None

        if df is None:
            if not self._all_file_paths:
                self._all_file_paths = [self.path]
            df = self._create_metadata_df(self._all_file_paths)

        df["image_path"] = df["image_path"].astype(str)

        cols_to_explode = ["file_path", "file_name"]
        try:
            df = df.explode(cols_to_explode, ignore_index=True)
        except ValueError:
            for col in cols_to_explode:
                df = df.explode(col)
            df = df.loc[lambda x: ~x["file_name"].duplicated()].reset_index(drop=True)

        df = df.loc[lambda x: x["image_path"] == self.path]

        self._df = df

        if self.path is not None and self.metadata:
            self.metadata = {
                key: value for key, value in self.metadata.items() if self.path in key
            }

        if self.metadata:
            try:
                metadata = self.metadata[self.path]
            except KeyError:
                metadata = {}
            for key, value in metadata.items():
                if key in dir(self):
                    setattr(self, f"_{key}", value)
                else:
                    setattr(self, key, value)

        elif self.metadata_attributes and self.path is not None:
            for key, value in self._get_metadata_attributes(
                self.metadata_attributes
            ).items():
                setattr(self, key, value)

    def clip(
        self, mask: GeoDataFrame | GeoSeries | Polygon | MultiPolygon, copy: bool = True
    ) -> "Image":
        """Clip band values to geometry mask while preserving bounds."""
        copied = self.copy() if copy else self

        fill: int = self.nodata or 0

        mask_array: np.ndarray = Band.from_geopandas(
            gdf=to_gdf(mask)[["geometry"]],
            default_value=1,
            fill=fill,
            out_shape=next(iter(self)).values.shape,
            bounds=self.bounds,
        ).values

        is_not_polygon = mask_array == fill

        for band in copied:
            if isinstance(band.values, np.ma.core.MaskedArray):
                band._values.mask |= is_not_polygon
            else:
                band._values = np.ma.array(
                    band.values, mask=is_not_polygon, fill_value=band.nodata
                )

        return copied

    def load(
        self,
        bounds: tuple | Geometry | GeoDataFrame | GeoSeries | None = None,
        indexes: int | tuple[int] | None = None,
        file_system=None,
        **kwargs,
    ) -> "ImageCollection":
        """Load all image Bands with threading."""
        if bounds is None and indexes is None and all(band.has_array for band in self):
            return self

        if self.masking:
            mask_array: np.ndarray = _read_mask_array(
                self,
                bounds=bounds,
                indexes=indexes,
                file_system=file_system,
                **kwargs,
            )

        with joblib.Parallel(n_jobs=self.processes, backend="threading") as parallel:
            parallel(
                joblib.delayed(_load_band)(
                    band,
                    bounds=bounds,
                    indexes=indexes,
                    file_system=file_system,
                    _masking=None,
                    **kwargs,
                )
                for band in self
            )

        if self.masking:
            for band in self:
                if isinstance(band.values, np.ma.core.MaskedArray):
                    band.values.mask |= mask_array
                else:
                    band.values = np.ma.array(
                        band.values, mask=mask_array, fill_value=self.nodata
                    )

        return self

    def _construct_image_from_bands(
        self, data: Sequence[Band], res: int | None
    ) -> None:
        self._bands = list(data)
        if res is None:
            res = {band.res for band in self.bands}
            if len(res) == 1:
                self._res = next(iter(res))
            else:
                raise ValueError(f"Different resolutions for the bands: {res}")
        else:
            self._res = res
        for key in self.metadata_attributes:
            band_values = {getattr(band, key) for band in self if hasattr(band, key)}
            band_values = {x for x in band_values if x is not None}
            if len(band_values) > 1:
                raise ValueError(f"Different {key} values in bands: {band_values}")
            elif len(band_values):
                try:
                    setattr(self, key, next(iter(band_values)))
                except AttributeError:
                    setattr(self, f"_{key}", next(iter(band_values)))

    def copy(self) -> "Image":
        """Copy the instance and its attributes."""
        copied = super().copy()
        for band in copied:
            band._mask = copied._mask
        return copied

    def apply(self, func: Callable, **kwargs) -> "Image":
        """Apply a function to each band of the Image."""
        with joblib.Parallel(n_jobs=self.processes, backend="loky") as parallel:
            parallel(joblib.delayed(_band_apply)(band, func, **kwargs) for band in self)

        return self

    def ndvi(
        self, red_band: str, nir_band: str, padding: int = 0, copy: bool = True
    ) -> NDVIBand:
        """Calculate the NDVI for the Image."""
        copied = self.copy() if copy else self
        red = copied[red_band].load()
        nir = copied[nir_band].load()

        arr: np.ndarray | np.ma.core.MaskedArray = ndvi(
            red.values, nir.values, padding=padding
        )

        return NDVIBand(
            arr,
            bounds=red.bounds,
            crs=red.crs,
            **{k: v for k, v in red._common_init_kwargs.items() if k != "res"},
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
            **{k: v for k, v in self._common_init_kwargs.items() if k != "res"},
        )

    def to_xarray(self) -> DataArray:
        """Convert the raster to  an xarray.DataArray."""
        return self._to_xarray(
            np.array([band.values for band in self]),
            transform=self[0].transform,
        )

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

        if self.masking:
            mask_band_id = self.masking["band_id"]
            paths = [path for path in self._df["file_path"] if mask_band_id not in path]
        else:
            paths = self._df["file_path"]

        self._bands = [
            self.band_class(
                path,
                all_file_paths=self._all_file_paths,
                **self._common_init_kwargs,
            )
            for path in paths
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

        return self._bands

    @property
    def _should_be_sorted(self) -> bool:
        sort_groups = ["band", "band_id"]
        return (
            self.filename_patterns
            and any(
                group in _get_non_optional_groups(pat)
                for group in sort_groups
                for pat in self.filename_patterns
            )
            or all(band.band_id is not None for band in self)
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

    def to_geopandas(self, column: str = "value") -> GeoDataFrame:
        """Convert the array to a GeoDataFrame of grid polygons and values."""
        return pd.concat(
            [band.to_geopandas(column=column) for band in self], ignore_index=True
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
            return self.bands[band]

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
            print("", self.path, self.date, other.path, other.date, sep="\n")
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
        res: int | None_ = None_,
        level: str | None_ | None = None_,
        processes: int = 1,
        metadata: str | dict | pd.DataFrame | None = None,
        nodata: int | None = None,
        **kwargs,
    ) -> None:
        """Initialiser."""
        if data is not None and kwargs.get("root"):
            root = _fix_path(kwargs.pop("root"))
            data = [f"{root}/{name}" for name in data]
            _from_root = True
        else:
            _from_root = False

        super().__init__(metadata=metadata, **kwargs)

        if callable(level) and level() is None:
            level = None

        self.nodata = nodata
        self.level = level
        self.processes = processes
        self._res = res if not (callable(res) and res() is None) else None
        self._crs = None

        self._df = None
        self._all_file_paths = None
        self._images = None

        if hasattr(data, "__iter__") and not isinstance(data, str):
            self._path = None
            if all(isinstance(x, Image) for x in data):
                self.images = [x.copy() for x in data]
                return
            elif all(isinstance(x, (str | Path | os.PathLike)) for x in data):
                # adding band paths (asuming 'data' is a sequence of image paths)
                try:
                    self._all_file_paths = _get_child_paths_threaded(data) | {
                        _fix_path(x) for x in data
                    }
                except FileNotFoundError as e:
                    if _from_root:
                        raise TypeError(
                            "When passing 'root', 'data' must be a sequence of image file names that have 'root' as parent path."
                        ) from e
                    raise e
                if self.level:
                    self._all_file_paths = [
                        path for path in self._all_file_paths if self.level in path
                    ]
                self._df = self._create_metadata_df(self._all_file_paths)
                return

        if not isinstance(data, (str | Path | os.PathLike)):
            raise TypeError("'data' must be string, Path-like or a sequence of Image.")

        self._path = _fix_path(str(data))

        self._all_file_paths = _get_all_file_paths(self.path)

        if self.level:
            self._all_file_paths = [
                path for path in self._all_file_paths if self.level in path
            ]

        self._df = self._create_metadata_df(self._all_file_paths)

    def groupby(
        self, by: str | list[str], copy: bool = True, **kwargs
    ) -> ImageCollectionGroupBy:
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
                        joblib.delayed(_copy_and_add_df_parallel)(
                            group_values, group_df, self, copy
                        )
                        for group_values, group_df in df.groupby(by, **kwargs)
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
                img._path = _fix_path(img[0].path)
            except PathlessImageError:
                pass
        return copied

    def apply(self, func: Callable, **kwargs) -> "ImageCollection":
        """Apply a function to all bands in each image of the collection."""
        with joblib.Parallel(n_jobs=self.processes, backend="loky") as parallel:
            parallel(
                joblib.delayed(_band_apply)(band, func, **kwargs)
                for img in self
                for band in img
            )

        return self

    def pixelwise(
        self,
        func: Callable,
        kwargs: dict | None = None,
        index_aligned_kwargs: dict | None = None,
        masked: bool = True,
        processes: int | None = None,
    ) -> np.ndarray | tuple[np.ndarray] | None:
        """Run a function for each pixel.

        The function should take a 1d array as first argument. This will be
        the pixel values for all bands in all images in the collection.
        """
        values = np.array([band.values for img in self for band in img])

        if (
            masked
            and self.nodata is not None
            and hasattr(next(iter(next(iter(self)))).values, "mask")
        ):
            mask_array = np.array(
                [
                    (band.values.mask) | (band.values.data == self.nodata)
                    for img in self
                    for band in img
                ]
            )
        elif masked and self.nodata is not None:
            mask_array = np.array(
                [band.values == self.nodata for img in self for band in img]
            )
        elif masked:
            mask_array = np.array([band.values.mask for img in self for band in img])
        else:
            mask_array = None

        nonmissing_row_indices, nonmissing_col_indices, results = pixelwise(
            func=func,
            values=values,
            mask_array=mask_array,
            index_aligned_kwargs=index_aligned_kwargs,
            kwargs=kwargs,
            processes=processes or self.processes,
        )

        return PixelwiseResults(
            nonmissing_row_indices,
            nonmissing_col_indices,
            results,
            shape=values.shape[1:],
            res=self.res,
            bounds=self.bounds,
            crs=self.crs,
            nodata=self.nodata or np.nan,
        )

    def get_unique_band_ids(self) -> list[str]:
        """Get a list of unique band_ids across all images."""
        return list({band.band_id for img in self for band in img})

    def filter(
        self,
        bands: str | list[str] | None = None,
        date_ranges: DATE_RANGES_TYPE = None,
        bbox: GeoDataFrame | GeoSeries | Geometry | tuple[float] | None = None,
        intersects: GeoDataFrame | GeoSeries | Geometry | tuple[float] | None = None,
        max_cloud_cover: int | None = None,
        copy: bool = True,
    ) -> "ImageCollection":
        """Filter images and bands in the collection."""
        copied = self.copy() if copy else self

        if date_ranges:
            copied = copied._filter_dates(date_ranges)

        if max_cloud_cover is not None:
            copied.images = [
                image
                for image in copied.images
                if image.cloud_cover_percentage < max_cloud_cover
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
        bounds = _get_bounds(bounds, self._bbox, self.union_all())
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
            **{k: v for k, v in self._common_init_kwargs.items() if k != "res"},
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
        bounds = _get_bounds(bounds, self._bbox, self.union_all())
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
                    **{k: v for k, v in self._common_init_kwargs.items() if k != "res"},
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
                        # band.load(
                        #     bounds=(_bounds if _bounds is not None else None),
                        #     **kwargs,
                        # )
                        # if not band.has_array
                        # else
                        band
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
                DataArray(
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
        """Sort Images by date, then file path if date attribute is missing."""
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
        file_system=None,
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
            if self.masking:
                masks: list[np.ndarray] = parallel(
                    joblib.delayed(_read_mask_array)(
                        img,
                        bounds=bounds,
                        indexes=indexes,
                        file_system=file_system,
                        **kwargs,
                    )
                    for img in self
                )

            parallel(
                joblib.delayed(_load_band)(
                    band,
                    bounds=bounds,
                    indexes=indexes,
                    file_system=file_system,
                    _masking=None,
                    **kwargs,
                )
                for img in self
                for band in img
            )

        if self.masking:
            for img, mask_array in zip(self, masks, strict=True):
                for band in img:
                    if isinstance(band.values, np.ma.core.MaskedArray):
                        band.values.mask |= mask_array
                    else:
                        band.values = np.ma.array(
                            band.values, mask=mask_array, fill_value=self.nodata
                        )

        return self

    def clip(
        self,
        mask: Geometry | GeoDataFrame | GeoSeries,
        dropna: bool = True,
        copy: bool = True,
    ) -> "ImageCollection":
        """Clip all image Bands while preserving bounds."""
        copied = self.copy() if copy else self

        copied._images = [img for img in copied if img.union_all()]

        fill: int = self.nodata or 0

        common_band_from_geopandas_kwargs = dict(
            gdf=to_gdf(mask)[["geometry"]],
            default_value=1,
            fill=fill,
        )

        for img in copied:
            img._rounded_bounds = tuple(int(x) for x in img.bounds)

        for bounds in {img._rounded_bounds for img in copied}:
            shapes = {
                band.values.shape
                for img in copied
                for band in img
                if img._rounded_bounds == bounds
            }
            if len(shapes) != 1:
                raise ValueError(f"Different shapes: {shapes}. For bounds {bounds}")

            mask_array: np.ndarray = Band.from_geopandas(
                **common_band_from_geopandas_kwargs,
                out_shape=next(iter(shapes)),
                bounds=bounds,
            ).values

            is_not_polygon = mask_array == fill

            for img in copied:
                if img._rounded_bounds != bounds:
                    continue

                for band in img:
                    if isinstance(band.values, np.ma.core.MaskedArray):
                        band._values.mask |= is_not_polygon
                    else:
                        band._values = np.ma.array(
                            band.values, mask=is_not_polygon, fill_value=band.nodata
                        )

        for img in copied:
            del img._rounded_bounds

        if dropna:
            copied.images = [
                img for img in copied if any(np.sum(band.values) for band in img)
            ]

        return copied

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

        if self.processes == 1:
            intersects_list: pd.Series = GeoSeries(
                [img.union_all() for img in self]
            ).intersects(other)
        else:
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

    def to_xarray(
        self,
        **kwargs,
    ) -> Dataset:
        """Convert the raster to  an xarray.Dataset.

        Images are converted to 2d arrays for each unique bounds.
        The spatial dimensions will be labeled "x" and "y". The third
        dimension defaults to "date" if all images have date attributes.
        Otherwise defaults to the image name.
        """
        if any(not band.has_array for img in self for band in img):
            raise ValueError("Arrays must be loaded.")

        # if by is None:
        if all(img.date for img in self):
            by = ["date"]
        elif not pd.Index([img.name for img in self]).is_unique:
            raise ValueError("Images must have unique names.")
        else:
            by = ["name"]
        # elif isinstance(by, str):
        # by = [by]

        xarrs: dict[str, DataArray] = {}
        for (bounds, band_id), collection in self.groupby(["bounds", "band_id"]):
            name = f"{band_id}_{'-'.join(str(int(x)) for x in bounds)}"
            first_band = collection[0][0]
            coords = _generate_spatial_coords(
                first_band.transform, first_band.width, first_band.height
            )
            values = np.array([band.to_numpy() for img in collection for band in img])
            assert len(values) == len(collection)

            # coords["band_id"] = [
            #     band.band_id or i for i, band in enumerate(collection[0])
            # ]
            for attr in by:
                coords[attr] = [getattr(img, attr) for img in collection]
            # coords["band"] = band_id  #

            dims = [*by, "y", "x"]
            # dims = ["band", "y", "x"]
            # dims = {}
            # for attr in by:
            #     dims[attr] = [getattr(img, attr) for img in collection]

            xarrs[name] = DataArray(
                values,
                coords=coords,
                dims=dims,
                # name=name,
                name=band_id,
                attrs={
                    "crs": collection.crs,
                    "band_id": band_id,
                },  # , "bounds": bounds},
                **kwargs,
            )

        return combine_by_coords(list(xarrs.values()))
        # return Dataset(xarrs)

    def to_geopandas(self, column: str = "value") -> dict[str, GeoDataFrame]:
        """Convert each band in each Image to a GeoDataFrame."""
        out = {}
        i = 0
        for img in self:
            for band in img:
                i += 1
                try:
                    name = band.name
                except AttributeError:
                    name = None

                if name is None:
                    name = f"{self.__class__.__name__}({i})"

                if name not in out:
                    out[name] = band.to_geopandas(column=column)
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

    def __iter__(self) -> Iterator[Image]:
        """Iterate over the images."""
        return iter(self.images)

    def __len__(self) -> int:
        """Number of images."""
        return len(self.images)

    def __getattr__(self, attr: str) -> Any:
        """Make iterable of metadata attribute."""
        if attr in (self.metadata_attributes or {}):
            return self._metadata_attribute_collection_type(
                [getattr(img, attr) for img in self]
            )
        return super().__getattribute__(attr)

    def __getitem__(
        self, item: int | slice | Sequence[int | bool]
    ) -> "Image | ImageCollection":
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
    def date(self) -> Any:
        """List of image dates."""
        return self._metadata_attribute_collection_type([img.date for img in self])

    @property
    def image_paths(self) -> Any:
        """List of image paths."""
        return self._metadata_attribute_collection_type([img.path for img in self])

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
            or all(getattr(img, sort_group) is not None for img in self)
        )

    @images.setter
    def images(self, new_value: list["Image"]) -> list["Image"]:
        new_value = list(new_value)
        if not new_value:
            self._images = new_value
            return
        if all(isinstance(x, Band) for x in new_value):
            if len(new_value) != len(self):
                raise ValueError("'images' must have same length as number of images.")
            new_images = []
            for i, img in enumerate(self):
                img._bands = [new_value[i]]
                new_images.append(img)
            self._images = new_images
            return
        if not all(isinstance(x, Image) for x in new_value):
            raise TypeError("images should be a sequence of Image.")
        self._images = new_value

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

        for group_values, subcollection in self.groupby(by):
            print("subcollection group values:", group_values)

            if "date" in x_var and subcollection._should_be_sorted:
                subcollection._images = list(sorted(subcollection._images))

            if "date" in x_var and subcollection._should_be_sorted:
                x = np.array(
                    [
                        datetime.datetime.strptime(band.date[:8], "%Y%m%d").date()
                        for img in subcollection
                        for band in img
                    ]
                )
                first_date = pd.Timestamp(x[0])
                x = (
                    pd.to_datetime(
                        [band.date[:8] for img in subcollection for band in img]
                    )
                    - pd.Timestamp(np.min(x))
                ).days
            else:
                x = np.arange(0, sum(1 for img in subcollection for band in img))

            subcollection.pixelwise(
                _plot_pixels_1d,
                kwargs=dict(
                    alpha=alpha,
                    x_var=x_var,
                    y_label=y_label,
                    rounding=rounding,
                    first_date=first_date,
                    figsize=figsize,
                ),
                index_aligned_kwargs=dict(x=x),
            )

    def __repr__(self) -> str:
        """String representation."""
        root = ""
        if self.path is not None:
            data = f"'{self.path}'"
        elif all(img.path is not None for img in self):
            data = [img.path for img in self]
            parents = {str(Path(path).parent) for path in data}
            if len(parents) == 1:
                data = [Path(path).name for path in data]
                root = f" root='{next(iter(parents))}',"
        else:
            data = [img for img in self]
        return f"{self.__class__.__name__}({data},{root} res={self.res}, level='{self.level}')"


class Sentinel2Config:
    """Holder of Sentinel 2 regexes, band_ids etc."""

    image_regexes: ClassVar[str] = (config.SENTINEL2_IMAGE_REGEX,)
    filename_regexes: ClassVar[str] = (config.SENTINEL2_FILENAME_REGEX,)
    metadata_attributes: ClassVar[
        dict[str, Callable | functools.partial | tuple[str]]
    ] = {
        "processing_baseline": functools.partial(
            _extract_regex_match_from_string,
            regexes=(r"<PROCESSING_BASELINE>(.*?)</PROCESSING_BASELINE>",),
        ),
        "cloud_cover_percentage": "_get_cloud_cover_percentage",
        "is_refined": "_get_image_refining_flag",
        "boa_quantification_value": "_get_boa_quantification_value",
    }
    l1c_bands: ClassVar[set[str]] = {
        "B01": 60,
        "B02": 10,
        "B03": 10,
        "B04": 10,
        "B05": 20,
        "B06": 20,
        "B07": 20,
        "B08": 10,
        "B8A": 20,
        "B09": 60,
        "B10": 60,
        "B11": 20,
        "B12": 20,
    }
    l2a_bands: ClassVar[set[str]] = {
        key: res for key, res in l1c_bands.items() if key != "B10"
    }
    all_bands: ClassVar[set[str]] = l1c_bands
    rbg_bands: ClassVar[tuple[str]] = ("B04", "B02", "B03")
    ndvi_bands: ClassVar[tuple[str]] = ("B04", "B08")
    masking: ClassVar[BandMasking] = BandMasking(
        band_id="SCL",
        values={
            2: "Topographic casted shadows",
            3: "Cloud shadows",
            8: "Cloud medium probability",
            9: "Cloud high probability",
            10: "Thin cirrus",
            11: "Snow or ice",
        },
    )

    def _get_image_refining_flag(self, xml_file: str) -> bool:
        match_ = re.search(
            r'Image_Refining flag="(?:REFINED|NOT_REFINED)"',
            xml_file,
        )
        if match_ is None:
            return None

        if "NOT_REFINED" in match_.group(0):
            return False
        elif "REFINED" in match_.group(0):
            return True
        else:
            raise _RegexError(xml_file)

    def _get_boa_quantification_value(self, xml_file: str) -> int:
        return int(
            _extract_regex_match_from_string(
                xml_file,
                (
                    r'<BOA_QUANTIFICATION_VALUE unit="none">-?(\d+)</BOA_QUANTIFICATION_VALUE>',
                ),
            )
        )

    def _get_cloud_cover_percentage(self, xml_file: str) -> float:
        return float(
            _extract_regex_match_from_string(
                xml_file,
                (
                    r"<Cloud_Coverage_Assessment>([\d.]+)</Cloud_Coverage_Assessment>",
                    r"<CLOUDY_PIXEL_OVER_LAND_PERCENTAGE>([\d.]+)</CLOUDY_PIXEL_OVER_LAND_PERCENTAGE>",
                ),
            )
        )


class Sentinel2CloudlessConfig(Sentinel2Config):
    """Holder of regexes, band_ids etc. for Sentinel 2 cloudless mosaic."""

    image_regexes: ClassVar[str] = (config.SENTINEL2_MOSAIC_IMAGE_REGEX,)
    filename_regexes: ClassVar[str] = (config.SENTINEL2_MOSAIC_FILENAME_REGEX,)
    masking: ClassVar[None] = None
    all_bands: ClassVar[list[str]] = [
        x.replace("B0", "B") for x in Sentinel2Config.all_bands
    ]
    rbg_bands: ClassVar[dict[str, str]] = {
        key.replace("B0", "B") for key in Sentinel2Config.rbg_bands
    }
    ndvi_bands: ClassVar[dict[str, str]] = {
        key.replace("B0", "B") for key in Sentinel2Config.ndvi_bands
    }


class Sentinel2Band(Sentinel2Config, Band):
    """Band with Sentinel2 specific name variables and regexes."""

    metadata_attributes = Sentinel2Config.metadata_attributes | {
        "boa_add_offset": "_get_boa_add_offset_dict",
    }

    def _get_boa_add_offset_dict(self, xml_file: str) -> int | None:
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
            return None

        dict_ = (
            pd.DataFrame(matches).set_index("band_id")["value"].astype(int).to_dict()
        )

        # some xml files have band ids in range index form
        # converting these to actual band ids (B01 etc.)
        is_integer_coded = [int(i) for i in dict_] == list(range(len(dict_)))

        if is_integer_coded:
            # the xml files contain 13 bandIds for both L1C and L2A
            # eventhough L2A doesn't have band B10
            all_bands = list(self.l1c_bands)
            if len(all_bands) != len(dict_):
                raise ValueError(
                    f"Different number of bands in xml file and config for {self.name}: {all_bands}, {list(dict_)}"
                )
            dict_ = {
                band_id: value
                for band_id, value in zip(all_bands, dict_.values(), strict=True)
            }

        try:
            return dict_[self.band_id]
        except KeyError as e:
            band_id = self.band_id.upper()
            for txt in ["B0", "B", "A"]:
                band_id = band_id.replace(txt, "")
                try:
                    return dict_[band_id]
                except KeyError:
                    continue
            raise KeyError(self.band_id, dict_) from e


class Sentinel2Image(Sentinel2Config, Image):
    """Image with Sentinel2 specific name variables and regexes."""

    band_class: ClassVar[Sentinel2Band] = Sentinel2Band

    def ndvi(
        self,
        red_band: str = "B04",
        nir_band: str = "B08",
        padding: int = 0,
        copy: bool = True,
    ) -> NDVIBand:
        """Calculate the NDVI for the Image."""
        return super().ndvi(
            red_band=red_band, nir_band=nir_band, padding=padding, copy=copy
        )


class Sentinel2Collection(Sentinel2Config, ImageCollection):
    """ImageCollection with Sentinel2 specific name variables and path regexes."""

    image_class: ClassVar[Sentinel2Image] = Sentinel2Image
    band_class: ClassVar[Sentinel2Band] = Sentinel2Band

    def __init__(self, data: str | Path | Sequence[Image], **kwargs) -> None:
        """ImageCollection with Sentinel2 specific name variables and path regexes."""
        level = kwargs.get("level", None_)
        if callable(level) and level() is None:
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
    """Concatenate ImageCollections."""
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


def _slope_2d(array: np.ndarray, res: int | tuple[int], degrees: int) -> np.ndarray:
    resx, resy = _res_as_tuple(res)

    gradient_x, gradient_y = np.gradient(array, resx, resy)

    gradient = abs(gradient_x) + abs(gradient_y)

    if not degrees:
        return gradient

    radians = np.arctan(gradient)
    degrees = np.degrees(radians)

    assert np.max(degrees) <= 90

    return degrees


def _clip_xarray(
    xarr: DataArray,
    mask: tuple[int, int, int, int],
    crs: Any,
    **kwargs,
) -> DataArray:
    # xarray needs a numpy array of polygons
    mask_arr: np.ndarray = to_geoseries(mask).values
    try:
        return xarr.rio.clip(
            mask_arr,
            crs=crs,
            **kwargs,
        )
    except NoDataInBounds:
        return np.array([])


def _get_all_file_paths(path: str) -> set[str]:
    if is_dapla():
        return {_fix_path(x) for x in sorted(set(_glob_func(path + "/**")))}
    else:
        return {
            _fix_path(x)
            for x in sorted(
                set(
                    _glob_func(path + "/**")
                    + _glob_func(path + "/**/**")
                    + _glob_func(path + "/**/**/**")
                    + _glob_func(path + "/**/**/**/**")
                    + _glob_func(path + "/**/**/**/**/**")
                )
            )
        }


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
    with joblib.Parallel(n_jobs=processes, backend="threading") as parallel:
        images: list[Image] = parallel(
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


class _ArrayNotLoadedError(ValueError):
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
        elif self.instance._from_geopandas:
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
    group_values: tuple[Any, ...],
    group_df: pd.DataFrame,
    self: ImageCollection,
    copy: bool,
) -> tuple[tuple[Any], ImageCollection]:
    copied = self.copy() if copy else self
    copied.images = [
        img.copy() if copy else img
        for img in group_df.drop_duplicates("_image_idx")["_image_instance"]
    ]
    if "band_id" in group_df:
        band_ids = set(group_df["band_id"].values)
        for img in copied.images:
            img._bands = [band for band in img if band.band_id in band_ids]

    return (group_values, copied)


def _get_bounds(bounds, bbox, band_bounds: Polygon) -> None | Polygon:
    if bounds is None and bbox is None:
        return None
    elif bounds is not None and bbox is None:
        return to_shapely(bounds).intersection(band_bounds)
    elif bounds is None and bbox is not None:
        return to_shapely(bbox).intersection(band_bounds)
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


def _read_mask_array(self: Band | Image, **kwargs) -> np.ndarray:
    mask_band_id = self.masking["band_id"]
    mask_paths = [path for path in self._all_file_paths if mask_band_id in path]
    if len(mask_paths) > 1:
        raise ValueError(
            f"Multiple file_paths match mask band_id {mask_band_id} for {self.path}"
        )
    elif not mask_paths:
        raise ValueError(
            f"No file_paths match mask band_id {mask_band_id} for {self.path} among "
            + str([Path(x).name for x in _ls_func(self.path)])
        )

    band = Band(
        next(iter(mask_paths)),
        **{**self._common_init_kwargs, "metadata": None},
    )
    band.load(**kwargs)
    boolean_mask = np.isin(band.values, list(self.masking["values"]))
    return boolean_mask


def _load_band(band: Band, **kwargs) -> Band:
    return band.load(**kwargs)


def _band_apply(band: Band, func: Callable, **kwargs) -> Band:
    return band.apply(func, **kwargs)


def _clip_band(band: Band, mask, **kwargs) -> Band:
    return band.clip(mask, **kwargs)


def _merge_by_band(collection: ImageCollection, **kwargs) -> Image:
    return collection.merge_by_band(**kwargs)


def _merge(collection: ImageCollection, **kwargs) -> Band:
    return collection.merge(**kwargs)


def _zonal_one_pair(i: int, poly: Polygon, band: Band, aggfunc, array_func, func_names):
    clipped = band.copy().clip(poly)
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


def _plot_pixels_1d(
    y: np.ndarray,
    x: np.ndarray,
    alpha: float,
    x_var: str,
    y_label: str,
    rounding: int,
    figsize: tuple,
    first_date: pd.Timestamp,
) -> None:
    coef, intercept = np.linalg.lstsq(
        np.vstack([x, np.ones(x.shape[0])]).T,
        y,
        rcond=None,
    )[0]
    predicted = np.array([intercept + coef * x for x in x])

    predicted_start = predicted[0]
    predicted_end = predicted[-1]
    predicted_change = predicted_end - predicted_start

    # Degrees of freedom
    dof = len(x) - 2

    # 95% confidence interval
    t_val = stats.t.ppf(1 - alpha / 2, dof)

    # Mean squared error of the residuals
    mse = np.sum((y - predicted) ** 2) / dof

    # Calculate the standard error of predictions
    pred_stderr = np.sqrt(
        mse * (1 / len(x) + (x - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    )

    # Calculate the confidence interval for predictions
    ci_lower = predicted - t_val * pred_stderr
    ci_upper = predicted + t_val * pred_stderr

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(x, y, color="#2c93db")
    ax.plot(x, predicted, color="#e0436b")
    ax.fill_between(
        x,
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

    if x_var == "date":
        date_labels = pd.to_datetime(
            [first_date + pd.Timedelta(days=int(day)) for day in x]
        )

        _, unique_indices = np.unique(date_labels.strftime("%Y-%m"), return_index=True)

        unique_x = np.array(x)[unique_indices]
        unique_labels = date_labels[unique_indices].strftime("%Y-%m")

        ax.set_xticks(unique_x)
        ax.set_xticklabels(unique_labels, rotation=45, ha="right")

    plt.show()


def pixelwise(
    func: Callable,
    values: np.ndarray,
    mask_array: np.ndarray | None = None,
    index_aligned_kwargs: dict | None = None,
    kwargs: dict | None = None,
    processes: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    """Run a function for each pixel of a 3d array."""
    index_aligned_kwargs = index_aligned_kwargs or {}
    kwargs = kwargs or {}

    if mask_array is not None:
        # skip pixels where all values are masked
        not_all_missing = np.all(mask_array, axis=0) == False
    else:
        mask_array = np.full(values.shape, False)
        not_all_missing = np.full(values.shape[1:], True)

    def select_pixel_values(row: int, col: int) -> np.ndarray:
        return values[~mask_array[:, row, col], row, col]

    # loop through long 1d arrays of aligned row and col indices
    nonmissing_row_indices, nonmissing_col_indices = not_all_missing.nonzero()
    with joblib.Parallel(n_jobs=processes, backend="loky") as parallel:
        results: list[Any] = parallel(
            joblib.delayed(func)(
                select_pixel_values(row, col),
                **kwargs,
                **{
                    key: value[~mask_array[:, row, col]]
                    for key, value in index_aligned_kwargs.items()
                },
            )
            for row, col in (
                zip(nonmissing_row_indices, nonmissing_col_indices, strict=True)
            )
        )

    return nonmissing_row_indices, nonmissing_col_indices, results
