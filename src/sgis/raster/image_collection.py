import abc
import functools
from copy import deepcopy, copy
import re
from pathlib import Path
from typing import Any
from typing import Sequence, Iterable
from typing import ClassVar, Callable
import random
from dataclasses import dataclass

from affine import Affine
import numpy as np
import dapla as dp
import pandas as pd
from shapely import box
from shapely.geometry import Point, Polygon, MultiPolygon
import pyproj
from geopandas import GeoSeries, GeoDataFrame
import joblib
import glob
import rasterio
from shapely import Geometry

try:
    import torch
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


from . import sentinel_config as config
from ..io._is_dapla import is_dapla
from ..io.opener import opener
from ..geopandas_tools.bounds import to_bbox, get_total_bounds
from ..geopandas_tools.general import get_common_crs
from ..geopandas_tools.conversion import to_gdf, to_shapely
from .raster import Raster
from ..helpers import get_all_files, get_numpy_func
from ..parallel.parallel import Parallel

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


@dataclass
class Band:
    path: str
    band_id: str
    date: str
    cloud_cover_percentage: float
    res: int | None
    file_system: dp.gcs.GCSFileSystem | None
    _mask: GeoDataFrame | GeoSeries | Geometry | tuple[float] | None = None

    def __lt__(self, other: "Band") -> bool:
        """Makes Bands sortable by band_id."""
        return self.band_id < other.band_id

    def load(self, bounds=None, indexes=1, **kwargs) -> np.ndarray:
        bounds = to_bbox(bounds) if bounds is not None else self._mask

        if isinstance(indexes, int):
            _indexes = (indexes,)
        else:
            _indexes = indexes

        with opener(self.path, file_system=self.file_system) as f:
            with rasterio.open(f) as src:
                # if bounds is None:
                #     self.transform = src.transform
                #     arr = src.read(indexes=indexes, **kwargs)
                #     return arr
                arr, transform = rasterio.merge.merge(
                    [src],
                    res=self.res,
                    indexes=_indexes,
                    bounds=bounds,
                    **kwargs,
                )
                self.transform = transform
                if isinstance(indexes, int):
                    return arr[0]
                return arr


class ImageBase(abc.ABC):
    image_regexes: ClassVar[str | None]
    filename_regexes: ClassVar[str | None]

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

    def _add_metadata_to_df(self, file_paths: list[str]) -> None:
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
            df, match_cols_filename = get_regexes_matches_for_df(
                df, "filename", self.filename_patterns, suffix=FILENAME_COL_SUFFIX
            )

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
                assert isinstance(df[col].iloc[0], str)
                grouped[col] = df.groupby("image_path")[col].apply(tuple)

            grouped = grouped.reset_index()
        else:
            grouped = df.drop_duplicates("image_path")

        grouped["imagename"] = grouped["image_path"].apply(
            lambda x: _fix_path(Path(x).name)
        )

        if self.image_patterns and len(grouped):
            grouped, _ = get_regexes_matches_for_df(
                grouped, "imagename", self.image_patterns, suffix=""
            )
            if not len(grouped):
                self._df = grouped
                return

        self._df = grouped.sort_values("date")

    @property
    def df(self):
        return self._df

    def copy(self) -> "ImageBase":
        copied = deepcopy(self)
        for key, value in copied.__dict__.items():
            try:
                setattr(copied, key, value.copy())
            except AttributeError:
                setattr(copied, key, deepcopy(value))
        return copied


class Image(ImageBase):
    filename_regexes: ClassVar[str | None] = None
    image_regexes: ClassVar[str | None] = None
    cloud_cover_regexes: ClassVar[tuple[str] | None] = None

    def __init__(
        self,
        path: str | Path,
        res: int | None = None,
        # crs: Any | None = None,
        file_system: dp.gcs.GCSFileSystem | None = None,
        df: pd.DataFrame | None = None,
        all_file_paths: list[str] | None = None,
        _mask: GeoDataFrame | GeoSeries | Geometry | tuple | None = None,
    ) -> None:
        super().__init__()

        self.path = str(path)
        self.res = res
        # self._crs = crs
        self.file_system = file_system
        self._mask = _mask

        if df is None:
            file_paths = ls_func(self.path)
            self._add_metadata_to_df(file_paths)
        else:
            self._df = df

        self._df = (
            self._df.explode(
                ["file_path", *[x for x in self._df if FILENAME_COL_SUFFIX in x]]
            )
            .loc[
                lambda x: (x["image_path"].str.contains(_fix_path(self.path)))
                & (x[f"band{FILENAME_COL_SUFFIX}"].notna())
            ]
            .sort_values(f"band{FILENAME_COL_SUFFIX}")
        )

        if self.cloud_cover_regexes:
            if all_file_paths is None:
                file_paths = ls_func(self.path)
            try:
                file_paths = [path for path in all_file_paths if self.name in path]
                self.cloud_cover_percentage = float(
                    get_cloud_percentage_in_local_dir(
                        file_paths, regexes=self.cloud_cover_regexes
                    )
                )
            except Exception:
                self.cloud_cover_percentage = None
        else:
            self.cloud_cover_percentage = None

        self._bands = list(
            sorted(
                Band(
                    path,
                    band_id=band_id,
                    date=date,
                    cloud_cover_percentage=self.cloud_cover_percentage,
                    res=res,
                    file_system=self.file_system,
                    _mask=self._mask,
                )
                for path, band_id, date in zip(
                    self.df["file_path"],
                    self.df[f"band{FILENAME_COL_SUFFIX}"],
                    self.df[f"date{FILENAME_COL_SUFFIX}"],
                    strict=True,
                )
            )
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
    def tile(self) -> str:
        return self._image_regex_searcher("tile")

    @property
    def date(self) -> str:
        return self._image_regex_searcher("date")

    @property
    def level(self) -> str:
        return self._image_regex_searcher("level")

    def load(self, bounds=None, indexes=1, **kwargs) -> np.ndarray:
        """Return 3 dimensional numpy.ndarray of shape (n bands, width, height)."""
        return np.array(
            [band.load(bounds=bounds, indexes=indexes, **kwargs) for band in self.bands]
        )

    def sample(
        self, n: int = 1, size: int = 1000, mask: Any = None, **kwargs
    ) -> "Image":
        """Take a random spatial sample of the image."""
        if mask is not None:
            points = GeoSeries(self.unary_union).clip(mask).sample_points(n)
        else:
            points = GeoSeries(self.unary_union).sample_points(n)
        buffered = points.buffer(size / 2).clip(self.unary_union)
        boxes = to_gdf([box(*arr) for arr in buffered.bounds.values], crs=self.crs)
        return self.load(bbox=boxes, **kwargs)

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
            raise TypeError("Image indices should be string or list of string.") from e
        return copied

    def __lt__(self, other: "Image") -> bool:
        """Makes Images sortable by date."""
        return self.date < other.date

    def __iter__(self):
        return iter(self.bands)

    def __len__(self) -> int:
        return len(self.bands)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bands={self.bands})"

    def get_cloud_array(self) -> np.ndarray:
        scl = self[self.cloud_band].load()
        return np.where(np.isin(scl, self.cloud_values), 1, 0)

    def get_cloud_polygons(self) -> GeoSeries:
        clouds = self.get_cloud_array()
        return (
            Raster.from_array(clouds, crs=self.crs, bounds=self.bounds, copy=False)
            .to_gdf()
            .loc[lambda x: x["value"] != 0]
            .geometry
        )

    def intersects(self, other: GeoDataFrame | GeoSeries | Geometry) -> bool:
        if hasattr(other, "crs") and not pyproj.CRS(self.crs).equals(
            pyproj.CRS(other.crs)
        ):
            raise ValueError(f"crs mismatch: {self.crs} and {other.crs}")
        return self.unary_union(box(*to_bbox(other)))

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
    def bounds(self) -> tuple[float, float, float, float] | None:
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
        return BoundingBox(*self._bounds, mint=self.date, maxt=self.date)

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

    def _image_regex_searcher(self, group: str) -> str | None:
        if not self.image_patterns:
            return None
        for pat in self.image_patterns:
            try:
                return re.match(pat, self.name).group(group)
            except AttributeError:
                pass
        raise ValueError(f"Couldn't find {group} in image_regexes")


class ImageCollection(ImageBase):
    image_regexes: ClassVar[str | None] = None
    filename_regexes: ClassVar[str | None] = None
    image_class: ClassVar[Image] = Image

    def __init__(
        self,
        path: str | Path,
        res: int,
        level: str | None,
        processes: int = 1,
        file_system: dp.gcs.GCSFileSystem | None = None,
        df: pd.DataFrame | None = None,
    ) -> None:
        super().__init__()

        self.path = str(path)
        self.level = level
        self.processes = processes
        self.file_system = file_system
        self.res = res
        self._mask = None
        self._band_ids = None

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

        self._all_filepaths = [path for path in self._all_filepaths if level in path]

        if df is not None:
            self._df = df
        else:
            self._add_metadata_to_df(self._all_filepaths)

    def groupby(self, by, **kwargs) -> tuple[tuple[Any], "ImageCollection"]:
        if isinstance(by, str):
            by = (by,)
        by = [col if col in self.df else f"{col}{FILENAME_COL_SUFFIX}" for col in by]

        return tuple(
            sorted(
                (i, self._copy_and_add_df(group))
                for i, group in self.df.explode(
                    ["file_path", *self._match_cols_filename]
                ).groupby(by, **kwargs)
            )
        )

    def aggregate_dates(
        self,
        bounds=None,
        method: str = "mean",
    ):
        if method == "mean":
            _method = "sum"

        bounds = to_bbox(bounds) if bounds is not None else self._mask

        arrs = []
        for (band_id,), band_collection in self.groupby("band"):
            datasets = [_open_raster(path) for path in band_collection.file_paths]
            arr, transform = rasterio.merge.merge(
                datasets, res=self.res, indexes=(1,), method=_method
            )
            arr = arr[0]
            arrs.append(arr)
        arrs = np.array(arrs)

        if method == "mean":
            arrs = arrs / len(self.file_paths)

        self.transform = transform

        return arrs

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

        if date_ranges:
            copied = copied._filter_dates(date_ranges, copy=False)

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
            copied._band_ids = list(bands)
            # print()
            # print()
            # print([img.__len__() for img in copied.images])
            # print([img[bands].__len__() for img in copied.images])

            copied._images = [img[bands] for img in copied.images]

        copied.images = list(sorted(copied._images))

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

        copied.df = copied.df.loc[
            lambda x: x["image_path"].apply(
                lambda y: date_is_within(y, date_ranges, copied.image_patterns)
            )
        ]

        return copied

    def _filter_bounds(
        self, other: GeoDataFrame | GeoSeries | Geometry | tuple, copy: bool = True
    ) -> "ImageCollection":
        copied = self.copy() if copy else self

        other = box(*to_bbox(other))

        with joblib.Parallel(n_jobs=copied.processes, backend="threading") as parallel:
            intersects_list: list[bool] = parallel(
                joblib.delayed(_intesects)(image, other) for image in copied
            )
        copied.images = [
            image for image, intersects in zip(copied, intersects_list) if intersects
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
        self, item: int | Sequence[int] | BoundingBox
    ) -> Image | TORCHGEO_RETURN_TYPE:
        if isinstance(item, int):
            return self.images[item]

        if not isinstance(item, BoundingBox):
            try:
                copied = self.copy()
                copied.images = [copied.images[i] for i in item]
                return copied
            except Exception:
                raise TypeError(
                    "ImageCollection indices must be int or BoundingBox. "
                    f"Got {type(item)}"
                )

        images = self.filter(bbox=item).set_mask(item)

        crs = get_common_crs(images)

        data: torch.Tensor = torch.cat(
            [numpy_to_torch(image.load()) for image in images]
        )

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

    @property
    def band_ids(self) -> list[str]:
        return list(self.df[f"band{FILENAME_COL_SUFFIX}"].explode().unique())

        if self._band_ids is not None:
            return self._band_ids
        else:
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
            # only fetch images when needed
            self._images = list(
                sorted(
                    get_images(
                        self.image_paths,
                        all_file_paths=self._all_filepaths,
                        df=self.df,
                        res=self.res,
                        processes=self.processes,
                        image_class=self.image_class,
                        _mask=self._mask,
                    )
                )
            )
            return self._images

    @images.setter
    def images(self, new_value: list["Image"]) -> list["Image"]:
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


def get_images(
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


def get_bands_for_image(image_path: str | Path) -> list[str]:
    if "L1C" in str(image_path):
        return list(config.SENTINEL2_L1C_BANDS)
    elif "L2A" in str(image_path):
        return list(config.SENTINEL2_L2A_BANDS)
    raise ValueError(image_path)


def get_band_resolutions_for_image(image_path: str | Path) -> dict[str, int]:
    if "L1C" in str(image_path):
        return config.SENTINEL2_L1C_BANDS
    elif "L2A" in str(image_path):
        return config.SENTINEL2_L2A_BANDS
    raise ValueError(image_path)


def numpy_to_torch(array: np.ndarray) -> torch.Tensor:
    """Convert numpy array to a pytorch tensor."""
    # fix numpy dtypes which are not supported by pytorch tensors
    if array.dtype == np.uint16:
        array = array.astype(np.int32)
    elif array.dtype == np.uint32:
        array = array.astype(np.int64)

    return torch.tensor(array)


def get_cloud_percentage_in_local_dir(
    paths: list[str], regexes: str | tuple[str]
) -> str | dict[str, str]:
    for i, path in enumerate(paths):
        if ".xml" not in path:
            continue
        with open_func(path, "rb") as file:
            filebytes: bytes = file.read()
            try:
                return _get_cloud_percentage(filebytes.decode("utf-8"), regexes)
            except RegexError as e:
                if i == len(paths) - 1:
                    raise e


class RegexError(ValueError):
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
                raise RegexError()
            return out
        try:
            return re.search(regexes, xml_file).group(1)
        except (TypeError, AttributeError):
            continue
    raise RegexError()


def _fix_path(path: str) -> str:
    return (
        str(path).replace("\\", "/").replace(r"\"", "/").replace("//", "/").rstrip("/")
    )


def _intesects(x, other) -> bool:
    return box(*x.bounds).intersects(other)


def get_regexes_matches_for_df(
    df, match_col: str, patterns: Sequence[re.Pattern], suffix: str = ""
) -> tuple[pd.DataFrame, list[str]]:
    if not len(df):
        return df, []
    assert df.index.is_unique
    matches: list[pd.DataFrame] = []
    for pat in patterns:
        try:
            matches.append(df[match_col].str.extract(pat))
        except ValueError:
            pass
    matches = pd.concat(matches).groupby(level=0, dropna=True).first()

    match_cols = [f"{col}{suffix}" for col in matches.columns]
    df[match_cols] = matches
    return df.loc[~df[match_cols].isna().all(axis=1)], match_cols


def date_is_within(
    path,
    date_ranges: (
        tuple[str | None, str | None] | tuple[tuple[str | None, str | None], ...] | None
    ),
    image_patterns: Sequence[re.Pattern],
) -> bool:
    for pat in image_patterns:
        try:
            date = re.match(pat, Path(path).name).group("date")
            break
        except AttributeError:
            date = None

    if date is None:
        return False

    date = date[:8]

    if date_ranges is None:
        return True
    if all(x is None or isinstance(x, str) for x in date_ranges):
        date_ranges = (date_ranges,)

    for date_range in date_ranges:

        try:
            date_min, date_max = date_range
            date_min = date_min or "00000000"
            date_max = date_max or "99999999"
            assert isinstance(date_min, str)
            assert len(date_min) == 8
            assert isinstance(date_max, str)
            assert len(date_max) == 8
        except AssertionError:
            raise TypeError(
                "date_ranges should be a tuple of two 8-charactered strings (start and end date)."
            )
        if date > date_min and date < date_max:
            return True

    return False


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
    cloud_band: ClassVar[str] = "SCL"
    cloud_values: ClassVar[tuple[int]] = (3, 8, 9, 10, 11)
    l2a_bands: ClassVar[dict[str, int]] = config.SENTINEL2_L2A_BANDS
    l1c_bands: ClassVar[dict[str, int]] = config.SENTINEL2_L1C_BANDS


class Sentinel2Image(Sentinel2Config, Image):
    cloud_cover_regexes: ClassVar[tuple[str]] = config.CLOUD_COVERAGE_REGEXES


class Sentinel2Collection(Sentinel2Config, ImageCollection):
    """ImageCollection with Sentinel2 specific name variables and regexes."""

    image_class: ClassVar[Sentinel2Image] = Sentinel2Image
