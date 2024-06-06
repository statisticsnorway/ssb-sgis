import functools
import glob
import itertools
import numbers
import os
import random
import re
import warnings
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Sequence
from copy import deepcopy
from json import loads
from pathlib import Path
from typing import Any
from typing import ClassVar

import joblib
import numpy as np
import pandas as pd
import pyproj
import rasterio
from affine import Affine
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from rasterio import features
from rasterio.enums import MergeAlg
from rtree.index import Index
from rtree.index import Property
from shapely import Geometry
from shapely import box
from shapely import unary_union
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import shape

try:
    import dapla as dp
    from dapla.gcs import GCSFileSystem
except ImportError:

    class GCSFileSystem:
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
except ImportError:

    class DataArray:
        """Placeholder."""


try:
    import torch
except ImportError:
    pass

try:
    from gcsfs.core import GCSFile
except ImportError:

    class GCSFile:
        """Placeholder."""


try:
    from torchgeo.datasets.utils import disambiguate_timestamp
except ImportError:

    class torch:
        """Placeholder."""

        class Tensor:
            """Placeholder to reference torch.Tensor."""


try:
    from torchgeo.datasets.utils import BoundingBox
except ImportError:

    class BoundingBox:
        """Placeholder."""

        def __init__(self, *args, **kwargs) -> None:
            """Placeholder."""
            raise ImportError("missing optional dependency 'torchgeo'")


from ..geopandas_tools.bounds import get_total_bounds
from ..geopandas_tools.bounds import to_bbox
from ..geopandas_tools.conversion import to_gdf
from ..geopandas_tools.conversion import to_shapely
from ..geopandas_tools.general import get_common_crs
from ..helpers import get_all_files
from ..helpers import get_numpy_func
from ..io._is_dapla import is_dapla
from ..io.opener import opener
from . import sentinel_config as config
from .base import get_index_mapper
from .indices import ndvi
from .zonal import _aggregate
from .zonal import _make_geometry_iterrows
from .zonal import _no_overlap_df
from .zonal import _prepare_zonal
from .zonal import _zonal_post

if is_dapla():

    def ls_func(*args, **kwargs) -> list[str]:
        return dp.FileClient.get_gcs_file_system().ls(*args, **kwargs)

    def glob_func(*args, **kwargs) -> list[str]:
        return dp.FileClient.get_gcs_file_system().glob(*args, **kwargs)

    def open_func(*args, **kwargs) -> GCSFile:
        return dp.FileClient.get_gcs_file_system().open(*args, **kwargs)

else:
    ls_func = functools.partial(get_all_files, recursive=False)
    open_func = open
    glob_func = glob.glob

TORCHGEO_RETURN_TYPE = dict[str, torch.Tensor | pyproj.CRS | BoundingBox]
FILENAME_COL_SUFFIX = "_filename"
DEFAULT_FILENAME_REGEX = r".*\.(?:tif|tiff|jp2)$"


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
        self, bounds=None, method="median", as_int: bool = True, indexes=None, **kwargs
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
        print("hihihih")
        for img, (group_values, _) in zip(images, self.data, strict=True):
            for attr, group_value in zip(self.by, group_values, strict=True):
                print(attr, group_value)
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
        self, bounds=None, method="median", as_int: bool = True, indexes=None, **kwargs
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
        return f"{self.__class__.__name__}({len(self)})"


class _ImageBase:
    image_regexes: ClassVar[str | None] = None
    filename_regexes: ClassVar[str | tuple[str]] = (DEFAULT_FILENAME_REGEX,)
    date_format: ClassVar[str] = "%Y%m%d"  # T%H%M%S"

    def __init__(self) -> None:

        self._merged = False
        self._from_array = False
        self._from_gdf = False

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

    @property
    def _common_init_kwargs(self) -> dict:
        return {
            "file_system": self.file_system,
            "processes": self.processes,
            "res": self.res,
            "_mask": self._mask,
        }

    @property
    def path(self) -> str:
        try:
            return self._path
        except AttributeError as e:
            raise PathlessImageError(self.__class__.__name__) from e

    @property
    def res(self) -> int:
        """Pixel resolution."""
        return self._res

    @property
    def centroid(self) -> Point:
        return self.unary_union.centroid

    def _name_regex_searcher(
        self, group: str, patterns: tuple[re.Pattern]
    ) -> str | None:
        if not patterns or not any(pat.groups for pat in patterns):
            return None
        for pat in patterns:
            try:
                return re.match(pat, self.name).group(group)
            except (AttributeError, TypeError, IndexError):
                pass
        raise ValueError(
            f"Couldn't find group '{group}' in name {self.name} with regex patterns {patterns}"
        )

    def _create_metadata_df(self, file_paths: list[str]) -> None:
        df = pd.DataFrame({"file_path": file_paths})

        df["filename"] = df["file_path"].apply(lambda x: _fix_path(Path(x).name))
        if not self.single_banded:
            df["image_path"] = df["file_path"].apply(
                lambda x: _fix_path(str(Path(x).parent))
            )
        else:
            df["image_path"] = df["file_path"]

        if not len(df):
            return df

        if self.filename_patterns:
            df, match_cols_filename = _get_regexes_matches_for_df(
                df, "filename", self.filename_patterns, suffix=FILENAME_COL_SUFFIX
            )

            if not len(df):
                return df

            self._match_cols_filename = match_cols_filename
            grouped = (
                df.drop(columns=match_cols_filename, errors="ignore")
                .drop_duplicates("image_path")
                .set_index("image_path")
            )
            for col in ["file_path", "filename", *match_cols_filename]:
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
            grouped, _ = _get_regexes_matches_for_df(
                grouped, "imagename", self.image_patterns, suffix=""
            )

        if "date" in grouped:
            return grouped.sort_values("date")
        else:
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


class _ImageBandBase(_ImageBase):
    def intersects(self, other: GeoDataFrame | GeoSeries | Geometry) -> bool:
        if hasattr(other, "crs") and not pyproj.CRS(self.crs).equals(
            pyproj.CRS(other.crs)
        ):
            raise ValueError(f"crs mismatch: {self.crs} and {other.crs}")
        return self.unary_union.intersects(to_shapely(other))

    @property
    def year(self) -> str:
        return self.date[:4]

    @property
    def month(self) -> str:
        return "".join(self.date.split("-"))[:6]

    @property
    def yyyymmd(self) -> str:
        return "".join(self.date.split("-"))[:7]

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

    @property
    def mint(self) -> float:
        return disambiguate_timestamp(self.date, self.date_format)[0]

    @property
    def maxt(self) -> float:
        return disambiguate_timestamp(self.date, self.date_format)[1]

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


class Band(_ImageBandBase):
    """Band holding a single 2 dimensional array representing an image band."""

    cmap: ClassVar[str | None] = None

    def __init__(
        self,
        data: str | np.ndarray,
        res: int | None,
        crs: Any | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        cmap: str | None = None,
        name: str | None = None,
        file_system: GCSFileSystem | None = None,
        band_id: str | None = None,
        processes: int = 1,
        _mask: GeoDataFrame | GeoSeries | Geometry | tuple[float] | None = None,
        **kwargs,
    ) -> None:
        """Band initialiser."""
        if isinstance(data, (GeoDataFrame | GeoSeries)):
            if res is None:
                raise ValueError("Must specify res when data is vector geometries.")
            bounds = to_bbox(bounds) if bounds is not None else data.total_bounds
            crs = crs if crs else data.crs
            data: np.ndarray = _arr_from_gdf(data, res=res, **kwargs)
            self._from_gdf = True

        if isinstance(data, np.ndarray):
            self._values = data
            if bounds is None:
                raise ValueError("Must specify bounds when data is an array.")
            self._bounds = to_bbox(bounds)
            self._crs = crs
            self.transform = _get_transform_from_bounds(
                self.bounds, shape=self.values.shape
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
        self._mask = _mask
        self._name = name
        self._band_id = band_id
        self.processes = processes

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
        """The numpy array, if loaded."""
        try:
            return self._values
        except AttributeError as e:
            raise ValueError("array is not loaded.") from e

    @values.setter
    def values(self, new_val):
        if isinstance(new_val, np.ndarray):
            raise TypeError(f"{self.__class__.__name__} 'values' must be np.ndarray.")
        self._values = new_val

    @property
    def band_id(self) -> str:
        """Band id."""
        if self._band_id is not None:
            return self._band_id
        return self._name_regex_searcher("band", self.filename_patterns)

    @property
    def height(self) -> int:
        """Pixel heigth of the image band."""
        i = 1 if len(self.values.shape) == 3 else 0
        return self.values.shape[i]

    @property
    def width(self) -> int:
        """Pixel width of the image band."""
        i = 2 if len(self.values.shape) == 3 else 1
        return self.values.shape[i]

    @property
    def tile(self) -> str:
        """Tile name from filename_regex."""
        if hasattr(self, "_tile") and self._tile:
            return self._tile
        return self._name_regex_searcher("tile", self.filename_patterns)

    @property
    def date(self) -> str:
        """Tile name from filename_regex."""
        if hasattr(self, "_date") and self._date:
            return self._date

        return self._name_regex_searcher("date", self.filename_patterns)

    @property
    def crs(self) -> str | None:
        """Coordinate reference system."""
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
        """Bounds as tuple (minx, miny, maxx, maxy)."""
        try:
            return self._bounds
        except AttributeError:
            with opener(self.path, file_system=self.file_system) as file:
                with rasterio.open(file) as src:
                    self._bounds = to_bbox(src.bounds)
                    self._crs = src.crs
                    return self._bounds
        except TypeError:
            return None

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

    def load(self, bounds=None, indexes=None, **kwargs) -> "Band":
        """Load and potentially clip the array.

        The array is stored in the 'values' property.
        """
        bounds = to_bbox(bounds) if bounds is not None else self._mask

        try:
            assert isinstance(self.values, np.ndarray)
            has_array = True
        except (ValueError, AssertionError):
            has_array = False

        if has_array:
            if bounds is None:
                return self
            # bounds_shapely = to_shapely(bounds)
            # if not bounds_shapely.intersects(self.)
            # bounds_arr = GeoSeries([bounds_shapely]).values
            bounds_arr = GeoSeries([to_shapely(bounds)]).values
            try:
                self._values = (
                    to_xarray(
                        self.values,
                        transform=self.transform,
                        crs=self.crs,
                        name=self.name,
                    )
                    .rio.clip(bounds_arr, crs=self.crs)
                    .to_numpy()
                )
            except NoDataInBounds:
                self._values = np.array([])
            self._bounds = bounds
            return self

        with opener(self.path, file_system=self.file_system) as f:
            with rasterio.open(f) as src:
                self._res = int(src.res[0]) if not self.res else self.res
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
        """Write the array as an image file."""
        if not hasattr(self, "_values"):
            raise ValueError(
                "Can only write image band from Band constructed from array."
            )

        if self.crs is None:
            raise ValueError("Cannot write None crs to image.")

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

        self._path = str(path)

    def sample(self, size: int = 1000, mask: Any = None, **kwargs) -> "Image":
        """Take a random spatial sample area of the Band."""
        copied = self.copy()
        if mask is not None:
            point = GeoSeries([copied.unary_union]).clip(mask).sample_points(1)
        else:
            point = GeoSeries([copied.unary_union]).sample_points(1)
        buffered = point.buffer(size / 2).clip(copied.unary_union)
        copied = copied.load(bounds=buffered.total_bounds, **kwargs)
        return copied

    def get_gradient(self, degrees: bool = False, copy: bool = True) -> "Band":
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
        )  # .transpose("y", "x")

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


class Image(_ImageBandBase):
    """Image consisting of one or more Bands."""

    cloud_cover_regexes: ClassVar[tuple[str] | None] = None
    band_class: ClassVar[Band] = Band

    def __init__(
        self,
        data: str | Path | Sequence[Band],
        res: int | None = None,
        # crs: Any | None = None,
        single_banded: bool = False,
        file_system: GCSFileSystem | None = None,
        df: pd.DataFrame | None = None,
        all_file_paths: list[str] | None = None,
        _mask: GeoDataFrame | GeoSeries | Geometry | tuple | None = None,
        processes: int = 1,
    ) -> None:
        """Image initialiser."""
        super().__init__()

        self._res = res
        # self._crs = crs
        self.file_system = file_system
        self._mask = _mask
        self.single_banded = single_banded
        self.processes = processes

        if hasattr(data, "__iter__") and all(isinstance(x, Band) for x in data):
            self._bands = list(data)
            self._bounds = get_total_bounds(self._bands)
            self._crs = get_common_crs(self._bands)
            res = list({band.res for band in self._bands})
            if len(res) == 1:
                self._res = res[0]
            else:
                raise ValueError(f"Different resolutions for the bands: {res}")
            return

        if not isinstance(data, (str | Path | os.PathLike)):
            raise TypeError("'data' must be string, Path-like or a sequence of Band.")

        self._path = str(data)

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
            df = self._create_metadata_df(file_paths)

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

        df = df.loc[lambda x: x["image_path"].str.contains(_fix_path(self.path))]

        if self.filename_patterns and any(pat.groups for pat in self.filename_patterns):
            df = df.loc[
                lambda x: (x[f"band{FILENAME_COL_SUFFIX}"].notna())
            ].sort_values(f"band{FILENAME_COL_SUFFIX}")

        if self.cloud_cover_regexes:
            if all_file_paths is None:
                file_paths = ls_func(self.path)
            else:
                file_paths = [path for path in all_file_paths if self.name in path]
            self.cloud_coverage_percentage = float(
                _get_regex_match_from_xml_in_local_dir(
                    file_paths, regexes=self.cloud_cover_regexes
                )
            )
        else:
            self.cloud_coverage_percentage = None

        self._bands = [
            self.band_class(
                path,
                **self._common_init_kwargs,
            )
            for path in (df["file_path"])
        ]

        if self.filename_patterns and any(pat.groups for pat in self.filename_patterns):
            self._bands = list(sorted(self._bands))

    def get_ndvi(self, red_band: str, nir_band: str) -> NDVIBand:
        """Calculate the NDVI for the Image."""
        red = self[red_band].load().values
        nir = self[nir_band].load().values

        arr: np.ndarray = ndvi(red, nir)

        return NDVIBand(
            arr,
            bounds=self.bounds,
            crs=self.crs,
            **self._common_init_kwargs,
        )

    def get_brightness(self, bounds=None, rbg_bands: list[str] | None = None) -> Band:
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
            **self._common_init_kwargs,
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
        return self._bands

    @property
    def tile(self) -> str:
        """Tile name from filename_regex."""
        if hasattr(self, "_tile") and self._tile:
            return self._tile
        return self._name_regex_searcher("tile", self.image_patterns)

    @property
    def date(self) -> str:
        """Tile name from filename_regex."""
        if hasattr(self, "_date") and self._date:
            return self._date

        return self._name_regex_searcher("date", self.image_patterns)

    @property
    def crs(self) -> str | None:
        """Coordinate reference system of the Image."""
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
        """Bounds of the Image (minx, miny, maxx, maxy)."""
        try:
            return self._bounds
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
    # def year(self) -> str:
    #     return self.date[:4]

    # @property
    # def month(self) -> str:
    #     return "".join(self.date.split("-"))[:6]

    # def write(self, image_path: str | Path, file_type: str = "tif", **kwargs) -> None:
    #     _test = kwargs.pop("_test")
    #     suffix = "." + file_type.strip(".")
    #     img_path = Path(image_path)
    #     for band in self:
    #         band_path = (img_path / band.name).with_suffix(suffix)
    #         if _test:
    #             print(f"{self.__class__.__name__}.write: {band_path}")
    #             continue

    #         band.write(band_path, **kwargs)

    # def read(self, bounds=None, **kwargs) -> np.ndarray:
    #     """Return 3 dimensional numpy.ndarray of shape (n bands, width, height)."""
    #     return np.array(
    #         [(band.load(bounds=bounds, **kwargs).values) for band in self.bands]
    #     )

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
            points = GeoSeries([self.unary_union]).clip(mask).sample_points(n)
        else:
            points = GeoSeries([self.unary_union]).sample_points(n)
        buffered = points.buffer(size / 2).clip(self.unary_union)
        boxes = to_gdf([box(*arr) for arr in buffered.bounds.values], crs=self.crs)
        copied._bands = [band.load(bounds=boxes, **kwargs) for band in copied]
        return copied

    # def get_filepath(self, band: str) -> str:
    #     simple_string_match = [path for path in self.file_paths if str(band) in path]
    #     if len(simple_string_match) == 1:
    #         return simple_string_match[0]

    #     regexes_matches = []
    #     for path in self.file_paths:
    #         for pat in self.filename_patterns:
    #             match_ = re.search(pat, Path(path).name)
    #             if match_ and str(band) == match_.group("band"):
    #                 regexes_matches.append(path)

    #     if len(regexes_matches) == 1:
    #         return regexes_matches[0]

    #     if len(regexes_matches) > 1:
    #         prefix = "Multiple"
    #     elif not regexes_matches:
    #         prefix = "No"

    #     raise KeyError(
    #         f"{prefix} matches for band {band} among paths {[Path(x).name for x in self.file_paths]}"
    #     )

    def __getitem__(
        self, band: str | int | Sequence[str] | Sequence[int]
    ) -> "Band | Image":
        """Get bands by band_id or integer index.

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
        return self.date < other.date

    def __iter__(self) -> Iterator[Band]:
        """Iterate over the Bands."""
        return iter(self.bands)

    def __len__(self) -> int:
        """Number of bands in the Image."""
        return len(self.bands)

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(bands={self.bands})"

    def get_cloud_band(self) -> Band:
        """Get a Band where self.cloud_values have value 1 and the rest have value 0."""
        scl = self[self.cloud_band].load()
        scl._values = np.where(np.isin(scl.values, self.cloud_values), 1, 0)
        return scl

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

    def _get_band(self, band: str) -> Band:
        if not isinstance(band, str):
            raise TypeError(f"band must be string. Got {type(band)}")

        bands = [x for x in self.bands if x.band_id == band]
        if len(bands) == 1:
            return bands[0]
        if len(bands) > 1:
            raise ValueError(
                f"Multiple matches for band_id {band} among {[x for x in self]}"
            )

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

    def __init__(
        self,
        data: str | Path | Sequence[Image],
        res: int,
        level: str | None,
        single_banded: bool = False,
        processes: int = 1,
        file_system: GCSFileSystem | None = None,
        df: pd.DataFrame | None = None,
        _mask: Any | None = None,
    ) -> None:
        """Initialiser."""
        super().__init__()

        self.level = level
        self.processes = processes
        self.file_system = file_system
        self._res = res
        self._mask = _mask
        self._band_ids = None
        self.single_banded = single_banded

        if hasattr(data, "__iter__") and all(isinstance(x, Image) for x in data):
            self._path = None
            self.images = data
            return

        if not isinstance(data, (str | Path | os.PathLike)):
            raise TypeError("'data' must be string, Path-like or a sequence of Image.")

        self._path = str(data)

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
            self._df = self._create_metadata_df(self._all_filepaths)

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
                single_banded=True,
                **self._common_init_kwargs,
                df=self._df,
                all_file_paths=self._all_filepaths,
            )
            for img in self
            for band in img
        ]
        return copied

    def merge(
        self, bounds=None, method="median", as_int: bool = True, indexes=None, **kwargs
    ) -> Band:
        """Merge all areas and all bands to a single Band."""
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
                as_int=as_int,
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
            **self._common_init_kwargs,
        )

        band._merged = True
        return band

    def merge_by_band(
        self,
        bounds: tuple | Geometry | GeoDataFrame | GeoSeries | None = None,
        method: str = "median",
        as_int: bool = True,
        indexes: int | tuple[int] | None = None,
        **kwargs,
    ) -> Image:
        """Merge all areas to a single tile, one band per band_id."""
        bounds = to_bbox(bounds) if bounds is not None else self._mask
        bounds = self.bounds if bounds is None else bounds
        out_bounds = self.bounds if bounds is None else bounds
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
            if method not in list(rasterio.merge.MERGE_METHODS) + ["mean"]:
                arr = band_collection._merge_with_numpy_func(
                    method=method,
                    bounds=bounds,
                    as_int=as_int,
                )
            else:
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
                    **self._common_init_kwargs,
                )
            )

        # return self.image_class(
        image = Image(
            bands,
            **self._common_init_kwargs,
        )

        image._merged = True
        return image

    def _merge_with_numpy_func(
        self,
        method: str | Callable,
        bounds=None,
        as_int: bool = True,
        indexes=None,
        **kwargs,
    ) -> np.ndarray:
        arrs = []
        numpy_func = get_numpy_func(method) if not callable(method) else method
        for (_bounds,), collection in self.groupby("bounds"):
            arr = np.array(
                [
                    band.load(bounds=bounds, indexes=indexes, **kwargs).values
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
                    name=str(_bounds),
                    attrs={"crs": self.crs},
                )
            )

        if bounds is None:
            bounds = self.bounds

        merged = merge_arrays(arrs, bounds=bounds, res=self.res)

        return merged.to_numpy()

    # def write(self, root: str | Path, file_type: str = "tif", **kwargs) -> None:
    #     _test = kwargs.pop("_test")
    #     suffix = "." + file_type.strip(".")
    #     for img in self:
    #         img_path = Path(root) / img.name
    #         for band in img:
    #             band_path = (img_path / band.name).with_suffix(suffix)
    #             if _test:
    #                 print(f"{self.__class__.__name__}.write: {band_path}")
    #                 continue
    #             band.write(band_path, **kwargs)

    def load_bands(self, bounds=None, indexes=None, **kwargs) -> "ImageCollection":
        """Load all image Bands with threading."""
        with joblib.Parallel(n_jobs=self.processes, backend="threading") as parallel:
            parallel(
                joblib.delayed(_load_band)(
                    band, bounds=bounds, indexes=indexes, **kwargs
                )
                for img in self
                for band in img
                if bounds is None or img.intersects(bounds)
            )

        return self

    def set_mask(
        self, mask: GeoDataFrame | GeoSeries | Geometry | tuple[float]
    ) -> "ImageCollection":
        """Set the mask to be used to clip the images to."""
        self._mask = to_bbox(mask)
        # only update images when already instansiated
        if hasattr(self, "_images"):
            for img in self._images:
                img._mask = self._mask
                img._bounds = self._mask
                for band in img:
                    band._mask = self._mask
                    band._bounds = self._mask
        return self

    def filter(
        self,
        bands: str | list[str] | None = None,
        date_ranges: (
            tuple[str | None, str | None]
            | tuple[tuple[str | None, str | None], ...]
            | None
        ) = None,
        bounds: GeoDataFrame | GeoSeries | Geometry | tuple[float] | None = None,
        max_cloud_coverage: int | None = None,
        copy: bool = True,
    ) -> "ImageCollection":
        """Filter images and bands in the collection."""
        copied = self.copy() if copy else self

        if isinstance(bounds, BoundingBox):
            date_ranges = (bounds.mint, bounds.maxt)

        if date_ranges:
            copied = copied._filter_dates(date_ranges, copy=False)

        if max_cloud_coverage is not None:
            copied.images = [
                image
                for image in copied.images
                if image.cloud_coverage_percentage < max_cloud_coverage
            ]

        if bounds is not None:
            copied = copied._filter_bounds(bounds, copy=False)

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

        copied.images = [
            img
            for img in self
            if _date_is_within(
                img.path, date_ranges, copied.image_patterns, copied.date_format
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

                if name not in out:
                    out[name] = band.to_gdf(column=column)
                else:
                    out[name] = f"{self.__class__.__name__}({i})"
        return out

    def sample(self, n: int = 1, size: int = 500) -> "ImageCollection":
        """Sample one or more areas of a given size and set this as mask for the images."""
        images = []
        bbox = to_gdf(self.unary_union).geometry.buffer(-size / 2)
        copied = self.copy()
        for _ in range(n):
            mask = to_bbox(bbox.sample_points(1).buffer(size))
            images += copied.filter(bounds=mask).set_mask(mask).images
        copied.images = images
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

    def __getitem__(
        self,
        item: int | slice | Sequence[int | bool] | BoundingBox | Sequence[BoundingBox],
    ) -> Image | TORCHGEO_RETURN_TYPE:
        """Select one Image by integer index, or multiple Images by slice, list of int or torchgeo.BoundingBox."""
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
                if all(isinstance(x, bool) for x in item):
                    copied.images = [
                        img for x, img in zip(item, copied, strict=True) if x
                    ]
                else:
                    copied.images = [copied.images[i] for i in item]
                return copied
            except Exception as e:
                if hasattr(item, "__iter__"):
                    endnote = f" of length {len(item)} with types {set(type(x) for x in item)}"
                raise TypeError(
                    "ImageCollection indices must be int or BoundingBox. "
                    f"Got {type(item)}{endnote}"
                ) from e

        elif isinstance(item, BoundingBox):
            date_ranges: tuple[str] = (item.mint, item.maxt)
            data: torch.Tensor = numpy_to_torch(
                np.array(
                    [
                        band.values
                        for band in self.filter(
                            bounds=item, date_ranges=date_ranges
                        ).merge_by_band(bounds=item)
                    ]
                )
            )
        else:
            bboxes: list[Polygon] = [to_bbox(x) for x in item]
            date_ranges: list[list[str, str]] = [(x.mint, x.maxt) for x in item]
            data: torch.Tensor = torch.cat(
                [
                    numpy_to_torch(
                        np.array(
                            [
                                band.values
                                for band in self.filter(
                                    bounds=bbox, date_ranges=date_range
                                ).merge_by_band(bounds=bbox)
                            ]
                        )
                    )
                    for bbox, date_range in zip(bboxes, date_ranges, strict=True)
                ]
            )

        crs = get_common_crs(self.images)

        key = "image"  # if self.is_image else "mask"
        sample = {key: data, "crs": crs, "bbox": item}

        return sample

    # def dates_as_float(self) -> list[tuple[float, float]]:
    #     return [disambiguate_timestamp(date, self.date_format) for date in self.dates]

    @property
    def mint(self) -> float:
        """Min timestamp of the images combined."""
        return min(img.mint for img in self)

    @property
    def maxt(self) -> float:
        """Max timestamp of the images combined."""
        return max(img.maxt for img in self)

    @property
    def band_ids(self) -> list[str]:
        """Sorted list of unique band_ids."""
        return list(sorted({band.band_id for img in self for band in img}))

    @property
    def file_paths(self) -> list[str]:
        """Sorted list of all file paths, meaning all band paths."""
        return list(sorted({band.path for img in self for band in img}))

    @property
    def dates(self) -> list[str]:
        """List of image dates."""
        return [img.date for img in self]

    def dates_as_int(self) -> list[int]:
        """List of image dates as 8-length integers."""
        return [int(img.date[:8]) for img in self]

    @property
    def image_paths(self) -> list[str]:
        """List of image paths."""
        return [img.path for img in self]

    @property
    def images(self) -> list["Image"]:
        """List of images in the Collection."""
        try:
            return self._images
        except AttributeError:
            # only fetch images when they are needed
            self._images = _get_images(
                list(self._df["image_path"]),
                all_file_paths=self._all_filepaths,
                df=self._df,
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
        else:
            self._images = list(new_value)
        if not all(isinstance(x, Image) for x in self._images):
            raise TypeError("images should be a sequence of Image.")

    @property
    def index(self) -> Index:
        """Spatial index that makes torchgeo think this class is a RasterDataset."""
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

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}({len(self)})"

    @property
    def unary_union(self) -> Polygon | MultiPolygon:
        """(Multi)Polygon representing the union of all image bounds."""
        return unary_union([img.unary_union for img in self])

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Total bounds for all Images combined."""
        return get_total_bounds([img.bounds for img in self])

    @property
    def crs(self) -> Any:
        """Common coordinate reference system of the Images."""
        return get_common_crs([img.crs for img in self])


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
        **first_collection._common_init_kwargs,
        # res=list(resolutions)[0],
    )
    out_collection._all_filepaths = list(
        sorted(
            set(itertools.chain.from_iterable([x._all_filepaths for x in collections]))
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
    )  # .transpose("y", "x")


def _slope_2d(array: np.ndarray, res: int, degrees: int) -> np.ndarray:
    gradient_x, gradient_y = np.gradient(array, res, res)

    gradient = abs(gradient_x) + abs(gradient_y)

    if not degrees:
        return gradient

    radians = np.arctan(gradient)
    degrees = np.degrees(radians)

    assert np.max(degrees) <= 90

    return degrees


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
                processes=processes,
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


class _RegexError(ValueError):
    pass


class PathlessImageError(ValueError):
    """Exception for when Images, Bands or ImageCollections have no path."""

    def __init__(self, instance: _ImageBase) -> None:
        """Initialise error class."""
        self.instance = instance

    def __str__(self) -> str:
        """String representation."""
        if self.instance._merged:
            what = "that have been merged"
        elif self.isinstance._from_array:
            what = "from arrays"
        elif self.isinstance._from_gdf:
            what = "from GeoDataFrames"

        return (
            f"{self.instance.__class__.__name__} instances {what} "
            "have no 'path' until they are written to file."
        )


def _get_regex_match_from_xml_in_local_dir(
    paths: list[str], regexes: str | tuple[str]
) -> str | dict[str, str]:
    for i, path in enumerate(paths):
        if ".xml" not in path:
            continue
        with open_func(path, "rb") as file:
            filebytes: bytes = file.read()
            try:
                return _extract_regex_match_from_string(
                    filebytes.decode("utf-8"), regexes
                )
            except _RegexError as e:
                if i == len(paths) - 1:
                    raise e


def _extract_regex_match_from_string(
    xml_file: str, regexes: tuple[str]
) -> str | dict[str, str]:
    for regex in regexes:
        if isinstance(regex, dict):
            out = {}
            for key, value in regex.items():
                try:
                    out[key] = re.search(value, xml_file).group(1)
                except (TypeError, AttributeError):
                    continue
            if len(out) != len(regex):
                raise _RegexError()
            return out
        try:
            return re.search(regex, xml_file).group(1)
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
        else:
            match_ = df[match_col].loc[df[match_col].str.match(pat)]
            if len(match_):
                matches.append(match_)

    matches = pd.concat(matches).groupby(level=0, dropna=True).first()

    if isinstance(matches, pd.Series):
        matches = pd.DataFrame({matches.name: matches.values}, index=matches.index)

    match_cols = [f"{col}{suffix}" for col in matches.columns]
    df[match_cols] = matches
    return (
        df.loc[~df[match_cols].isna().all(axis=1)].drop(
            columns=f"{match_col}{suffix}", errors="ignore"
        ),
        match_cols,
    )


def _arr_from_gdf(
    gdf: GeoDataFrame,
    res: int,
    fill: int = 0,
    all_touched: bool = False,
    merge_alg: Callable = MergeAlg.replace,
    default_value: int = 1,
    dtype: Any | None = None,
) -> np.ndarray:
    """Construct Raster from a GeoDataFrame or GeoSeries.

    The GeoDataFrame should have

    Args:
        gdf: The GeoDataFrame to rasterize.
        res: Resolution of the raster in units of the GeoDataFrame's coordinate reference system.
        fill: Fill value for areas outside of input geometries (default is 0).
        all_touched: Whether to consider all pixels touched by geometries,
            not just those whose center is within the polygon (default is False).
        merge_alg: Merge algorithm to use when combining geometries
            (default is 'MergeAlg.replace').
        default_value: Default value to use for the rasterized pixels
            (default is 1).
        dtype: Data type of the output array. If None, it will be
            determined automatically.

    Returns:
        A Raster instance based on the specified GeoDataFrame and parameters.

    Raises:
        TypeError: If 'transform' is provided in kwargs, as this is
        computed based on the GeoDataFrame bounds and resolution.
    """
    if isinstance(gdf, GeoSeries):
        values = gdf.index
        gdf = gdf.to_frame("geometry")
    elif isinstance(gdf, GeoDataFrame):
        if len(gdf.columns) > 2:
            raise ValueError(
                "gdf should have only a geometry column and one numeric column to "
                "use as array values. "
                "Alternatively only a geometry column and a numeric index."
            )
        elif len(gdf.columns) == 1:
            values = gdf.index
        else:
            col: str = next(
                iter([col for col in gdf if col != gdf._geometry_column_name])
            )
            values = gdf[col]

    if isinstance(values, pd.MultiIndex):
        raise ValueError("Index cannot be MultiIndex.")

    shape = _get_shape_from_bounds(gdf.total_bounds, res=res)
    transform = _get_transform_from_bounds(gdf.total_bounds, shape)

    return features.rasterize(
        _gdf_to_geojson_with_col(gdf, values),
        out_shape=shape,
        transform=transform,
        fill=fill,
        all_touched=all_touched,
        merge_alg=merge_alg,
        default_value=default_value,
        dtype=dtype,
    )


def _gdf_to_geojson_with_col(gdf: GeoDataFrame, values: np.ndarray) -> list[dict]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return [
            (feature["geometry"], val)
            for val, feature in zip(
                values, loads(gdf.to_json())["features"], strict=False
            )
        ]


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
        except AssertionError as err:
            raise TypeError(
                "date_ranges should be a tuple of two 8-charactered strings (start and end date)."
                f"Got {date_range} of type {[type(x) for x in date_range]}"
            ) from err
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


def _array_to_geojson(
    array: np.ndarray, transform: Affine, processes: int
) -> list[tuple]:
    if np.ma.is_masked(array):
        array = array.data
    try:
        return _array_to_geojson_loop(array, transform, processes)

    except ValueError:
        try:
            array = array.astype(np.float32)
            return _array_to_geojson_loop(array, transform, processes)

        except Exception as err:
            raise err.__class__(array.shape, err) from err


def _array_to_geojson_loop(array, transform, processes):
    if processes == 1:
        return [
            (value, shape(geom))
            for geom, value in features.shapes(array, transform=transform, mask=None)
        ]
    else:
        with joblib.Parallel(n_jobs=processes, backend="threading") as parallel:
            return parallel(
                joblib.delayed(_value_geom_pair)(value, geom)
                for geom, value in features.shapes(
                    array, transform=transform, mask=None
                )
            )


def _value_geom_pair(value, geom):
    return (value, shape(geom))


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

    # for col in group.columns.difference({"_image_instance", "_image_idx"}):
    #     if not all(
    #         col in dir(band) or col in band.__dict__ for img in copied for band in img
    #     ):
    #         continue
    #     values = set(group[col].values)
    #     for img in copied.images:
    #         img._bands = [band for band in img if getattr(band, col) in values]

    return (i, copied)


def _open_raster(path: str | Path) -> rasterio.io.DatasetReader:
    with opener(path) as file:
        return rasterio.open(file)


def _load_band(band: Band, **kwargs) -> None:
    band.load(**kwargs)


def _merge_by_band(collection: ImageCollection, **kwargs) -> Image:
    print("_merge_by_band", collection.dates)
    return collection.merge_by_band(**kwargs)


def _merge(collection: ImageCollection, **kwargs) -> Band:
    return collection.merge(**kwargs)


def _zonal_one_pair(i: int, poly: Polygon, band: Band, aggfunc, array_func, func_names):
    clipped = band.copy().load(bounds=poly)
    if not np.size(clipped.values):
        return _no_overlap_df(func_names, i, date=band.date)
    return _aggregate(clipped.values, array_func, aggfunc, func_names, band.date, i)


class Sentinel2Config:
    """Holder of Sentinel 2 regexes, band_ids etc."""

    image_regexes: ClassVar[str] = (
        config.SENTINEL2_IMAGE_REGEX,
    )  # config.SENTINEL2_MOSAIC_IMAGE_REGEX,)
    filename_regexes: ClassVar[str] = (
        config.SENTINEL2_FILENAME_REGEX,
        # config.SENTINEL2_MOSAIC_FILENAME_REGEX,
        config.SENTINEL2_CLOUD_FILENAME_REGEX,
    )
    all_bands: ClassVar[list[str]] = list(config.SENTINEL2_BANDS)
    rbg_bands: ClassVar[list[str]] = ["B02", "B03", "B04"]
    ndvi_bands: ClassVar[list[str]] = ["B04", "B08"]
    cloud_band: ClassVar[str] = "SCL"
    cloud_values: ClassVar[tuple[int]] = (3, 8, 9, 10, 11)
    l2a_bands: ClassVar[dict[str, int]] = config.SENTINEL2_L2A_BANDS
    l1c_bands: ClassVar[dict[str, int]] = config.SENTINEL2_L1C_BANDS
    date_format: ClassVar[str] = "%Y%m%d"  # T%H%M%S"


class Sentinel2CloudlessConfig(Sentinel2Config):
    """Holder of regexes, band_ids etc. for Sentinel 2 cloudless mosaic."""

    image_regexes: ClassVar[str] = (config.SENTINEL2_MOSAIC_IMAGE_REGEX,)
    filename_regexes: ClassVar[str] = (config.SENTINEL2_MOSAIC_FILENAME_REGEX,)
    cloud_band: ClassVar[None] = None
    cloud_values: ClassVar[None] = None
    date_format: ClassVar[str] = "%Y%m%d"


class Sentinel2Band(Sentinel2Config, Band):
    """Band with Sentinel2 specific name variables and regexes."""


class Sentinel2Image(Sentinel2Config, Image):
    """Image with Sentinel2 specific name variables and regexes."""

    cloud_cover_regexes: ClassVar[tuple[str]] = config.CLOUD_COVERAGE_REGEXES
    band_class: ClassVar[Sentinel2Band] = Sentinel2Band

    def get_ndvi(
        self,
        red_band: str = Sentinel2Config.ndvi_bands[0],
        nir_band: str = Sentinel2Config.ndvi_bands[1],
    ) -> NDVIBand:
        """Calculate the NDVI for the Image."""
        return super().get_ndvi(red_band=red_band, nir_band=nir_band)


class Sentinel2Collection(Sentinel2Config, ImageCollection):
    """ImageCollection with Sentinel2 specific name variables and regexes."""

    image_class: ClassVar[Sentinel2Image] = Sentinel2Image
    band_class: ClassVar[Sentinel2Band] = Sentinel2Band


class Sentinel2CloudlessBand(Sentinel2CloudlessConfig, Band):
    """Band for cloudless mosaic with Sentinel2 specific name variables and regexes."""


class Sentinel2CloudlessImage(Sentinel2CloudlessConfig, Sentinel2Image):
    """Image for cloudless mosaic with Sentinel2 specific name variables and regexes."""

    # image_regexes: ClassVar[str] = (config.SENTINEL2_MOSAIC_IMAGE_REGEX,)
    # filename_regexes: ClassVar[str] = (config.SENTINEL2_MOSAIC_FILENAME_REGEX,)

    cloud_cover_regexes: ClassVar[None] = None
    band_class: ClassVar[Sentinel2CloudlessBand] = Sentinel2CloudlessBand

    get_ndvi = Sentinel2Image.get_ndvi
    # def get_ndvi(
    #     self,
    #     red_band: str = Sentinel2Config.ndvi_bands[0],
    #     nir_band: str = Sentinel2Config.ndvi_bands[1],
    # ) -> NDVIBand:
    #     """Calculate the NDVI for the Image."""
    #     return super().get_ndvi(red_band=red_band, nir_band=nir_band)


class Sentinel2CloudlessCollection(Sentinel2CloudlessConfig, ImageCollection):
    """ImageCollection with Sentinel2 specific name variables and regexes."""

    # image_regexes: ClassVar[str] = (config.SENTINEL2_MOSAIC_IMAGE_REGEX,)
    # filename_regexes: ClassVar[str] = (config.SENTINEL2_MOSAIC_FILENAME_REGEX,)

    image_class: ClassVar[Sentinel2CloudlessImage] = Sentinel2CloudlessImage
    band_class: ClassVar[Sentinel2Band] = Sentinel2Band
