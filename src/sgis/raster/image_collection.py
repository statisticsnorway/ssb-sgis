import abc
import functools
from copy import deepcopy, copy
import re
from pathlib import Path
from typing import Any
from typing import Sequence
from typing import ClassVar, Callable
import random
from dataclasses import dataclass

import numpy as np
import dapla as dp
import pandas as pd
from shapely import box
from shapely.geometry import Point, Polygon, MultiPolygon
import torch
import pyproj
from geopandas import GeoSeries, GeoDataFrame
import joblib
import glob
import rasterio
from shapely import Geometry

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


class ImageBase(abc.ABC):
    image_regex: ClassVar[str | None]
    filename_regex: ClassVar[str | None]

    def __init__(self) -> None:

        if self.filename_regex:
            self.filename_pattern = re.compile(self.filename_regex, flags=re.VERBOSE)
        else:
            self.filename_pattern = None

        if self.image_regex:
            self.image_pattern = re.compile(self.image_regex, flags=re.VERBOSE)
        else:
            self.image_pattern = None

    def _add_metadata_to_df(self, file_paths: list[str]) -> None:
        df = pd.DataFrame({"file_path": [(path) for path in file_paths]})
        for col in ["band", "tile", "date", "resolution"]:
            df[col] = None
        df["filename"] = df["file_path"].apply(lambda x: _fix_path(Path(x).name))
        df["image_path"] = df["file_path"].apply(
            lambda x: _fix_path(str(Path(x).parent))
        )

        # df["fixed_file_path"] = df["file_path"].apply(_fix_path)
        # df["fixed_image_path"] = df["image_path"].apply(_fix_path)

        matches: pd.DataFrame = df["filename"].str.extract(self.filename_pattern)
        df[list(matches.columns)] = matches

        to_pandas_na = {value: pd.NA for value in [" ", "", None, np.nan]}
        for col in df:
            df[col] = df[col].replace(to_pandas_na).fillna(pd.NA)

        self._df = df.sort_values("date")

    @property
    def file_paths(self) -> list[str]:
        return list(sorted(self._df["file_path"].dropna()))

    @property
    def band_ids(self) -> list[str]:
        return list(sorted(self._df["band"].dropna().unique()))

    @property
    def resolutions(self) -> list[str]:
        return list(sorted(self._df["resolutions"].dropna().unique()))

    def copy(self) -> "ImageBase":
        copied = deepcopy(self)
        for key, value in copied.__dict__.items():
            setattr(copied, key, deepcopy(value))
        return copied


class CollectionBase(ImageBase):

    def groupby(self, by, **kwargs) -> list["CollectionBase"]:

        return [self._update_df(group) for _, group in self._df.groupby(by, **kwargs)]

    def _update_df(self, new_df: pd.DataFrame) -> "CollectionBase":
        copied = deepcopy(self)
        for key, value in copied.__dict__.items():
            setattr(copied, key, deepcopy(value))
        copied._df = new_df
        return copied

    @property
    def image_paths(self) -> list[str]:
        return list(sorted(self._df["image_path"]))

    @property
    def dates(self) -> list[str]:
        return list(sorted(self._df["date"].dropna().unique()))

    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, new_value: list["Image"]) -> list["Image"]:
        self._images = list(new_value)
        if not all(isinstance(x, Image) for x in self._images):
            raise TypeError("images should be a sequence of Image.")
        self._df = self._df.loc[
            lambda x: x["image_path"].isin([x.path for x in self._images])
        ]


@dataclass
class Band:
    path: str
    band_id: str
    date: str
    cloud_cover_percentage: float
    res: int | None

    def __lt__(self, other: "Band") -> bool:
        """Makes Bands sortable by band_id."""
        return self.band_id < other.band_id

    def load(
        self, bounds=None, indexes=1, nodata: int | None = None, **kwargs
    ) -> np.ndarray:
        bounds = to_bbox(bounds) if bounds is not None else None

        if isinstance(indexes, int):
            _indexes = (indexes,)
        else:
            _indexes = indexes

        with opener(self.path, file_system=self.file_system) as f:
            with rasterio.open(f) as src:
                if bounds is None:
                    self.transform = src.transform
                    arr = src.read(indexes=indexes, nodata=nodata, **kwargs)
                    return arr
                arr, transform = rasterio.merge.merge(
                    [src],
                    res=self.res,
                    indexes=_indexes,
                    bounds=bounds,
                    nodata=nodata,
                    **kwargs,
                )
                self.transform = transform
                if isinstance(indexes, int):
                    return arr[0]
                return arr


class Image(ImageBase):
    filename_regex: ClassVar[str | None] = None
    image_regex: ClassVar[str | None] = None
    cloud_cover_regexes: ClassVar[tuple[str] | None] = None

    def __init__(
        self,
        path: str | Path,
        res: int | None = None,
        # crs: Any | None = None,
        bands: str | list[str] | None = None,
        df: pd.DataFrame | None = None,
        file_system: dp.gcs.GCSFileSystem | None = None,
    ) -> None:
        super().__init__()

        self.path = str(path)
        self.res = res
        # self._crs = crs
        self.file_system = file_system

        if df is None:
            file_paths = list(sorted(ls_func(self.path)))
            self._df = self._add_metadata_to_df(file_paths)

        self._df = df.loc[
            lambda x: (x["image_path"].str.contains(_fix_path(self.path)))
            & (x["band"].notna())
        ].sort_values("band")

        if bands is not None:
            if isinstance(bands, str):
                bands = [bands]
            self._df = self._df.loc[lambda x: x["band"].isin(bands)]

        if self.cloud_cover_regexes:
            try:
                self.cloud_cover_percentage = float(
                    get_cloud_percentage_in_local_dir(
                        self.file_paths, regexes=self.cloud_cover_regexes
                    )
                )
            except Exception:
                self.cloud_cover_percentage = None
        else:
            self.cloud_cover_percentage = None

        self._bands = [
            Band(
                path,
                band_id,
                date=date,
                cloud_cover_percentage=self.cloud_cover_percentage,
                res=res,
            )
            for path, band_id, date in zip(
                self._df["band"], self._df["file_path"], self._df["date"], strict=True
            )
        ]

    @property
    def bands(self):
        return self._bands

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def tile(self) -> str:
        return re.match(self.image_pattern, self.name).group("tile")

    @property
    def date(self) -> str:
        return re.match(self.image_pattern, self.name).group("date")

    @property
    def level(self) -> str:
        return re.match(self.image_pattern, self.name).group("level")

    def load(
        self, bounds=None, indexes=1, nodata: int | None = None, **kwargs
    ) -> np.ndarray:
        """Return 3 dimensional numpy.ndarray of shape (n bands, width, height)."""
        return np.array(
            [
                band.load(bounds=bounds, indexes=indexes, nodata=nodata, **kwargs)
                for band in self.bands
            ]
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

        regex_matches = []
        for path in self.file_paths:
            match_ = re.search(self.filename_pattern, Path(path).name)
            if match_ and str(band) == match_.group("band"):
                regex_matches.append(path)

        if len(regex_matches) == 1:
            return regex_matches[0]

        if len(regex_matches) > 1:
            prefix = "Multiple"
        elif not regex_matches:
            prefix = "No"

        raise KeyError(
            f"{prefix} matches for band {band} among paths {[Path(x).name for x in self.file_paths]}"
        )

    def get_band(self, band: str) -> Band:
        bands = [band for band in self.bands if band.band_id == band]
        if len(bands) == 1:
            return bands[0]
        more_bands = [band for band in self.bands if band.path == band]
        if len(more_bands) == 1:
            return more_bands[0]

        if len(bands) > 1:
            prefix = "Multiple"
        elif not bands:
            prefix = "No"

        raise KeyError(
            f"{prefix} matches for band {band} among paths {[Path(band.path).name for band in self.bands]}"
        )

    def __getitem__(self, band: str | list[str]) -> "np.ndarray | Image":
        if isinstance(band, str):
            return self.get_band(band).load()
        copied = deepcopy(self)
        self._df = self._df.copy()
        copied._bands = [copied.get_band(x).load() for x in band]
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
        scl = self[self.cloud_band]
        return np.where(np.isin(scl, self.cloud_values), 1, 0)

    def get_cloud_polygons(self) -> MultiPolygon:
        clouds = self.get_cloud_array()
        clouds[clouds == 0] = np.nan
        # clouds[clouds.mask] = np.nan
        return (
            Raster.from_array(clouds, crs=self.crs, bounds=self.bounds, copy=False)
            .to_gdf()
            .unary_union
        )

    @property
    def crs(self) -> str | None:
        try:
            return self._crs
        except AttributeError:
            if not len(self):
                return None
            with opener(self.file_paths[0], file_system=self.file_system) as file:
                with rasterio.open(file) as src:
                    self._bounds = src.bounds
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
                    self._bounds = src.bounds
                    self._crs = src.crs
                    return self._bounds

    @property
    def unary_union(self) -> Polygon:
        return box(*self.bounds)

    @property
    def centroid(self) -> str:
        x = (self.bounds[0] + self.bounds[2]) / 2
        y = (self.bounds[1] + self.bounds[3]) / 2
        return Point(x, y)

    @property
    def bbox(self) -> BoundingBox:
        return BoundingBox(*self._bounds, mint=self.date, maxt=self.date)


class Tile(CollectionBase):
    filename_regex: ClassVar[str | None] = None
    image_regex: ClassVar[str | None] = None
    image_class: ClassVar[Image] = Image

    def __init__(
        self,
        tile_id: str,
        df: pd.DataFrame,
        level: str | None,
        res: int | None,
        file_system: dp.gcs.GCSFileSystem | None,
        processes: int,
    ) -> None:
        super().__init__()

        self.tile_id = str(tile_id)
        self.level = level
        self.file_system = file_system
        self.processes = processes
        self.res = res

        self._df = df.loc[
            lambda x: (x["image_path"].str.contains(self.tile_id))
            & (x["image_path"].str.contains(self.level or ""))
        ].sort_values("date")

    def get_images(
        self,
        bands: str | list[str] | None = None,
        date_ranges: tuple[str, str] | None = None,
        max_cloud_cover: int | None = None,
        copy: bool = True,
    ) -> list[Image]:
        copied = self.copy() if copy else self
        copied.images = get_images(
            copied.image_paths,
            level=copied.level,
            bands=bands,
            df=copied._df,
            date_ranges=date_ranges,
            image_regex=copied.image_regex,
            processes=copied.processes,
            image_class=copied.image_class,
            max_cloud_cover=max_cloud_cover,
        )
        return copied

    def sample_images(self, n: int) -> list[Image]:
        tile = deepcopy(tile)
        paths = tile.image_paths
        if n > len(paths):
            raise ValueError(
                f"n ({n}) is higher than number of images in collection ({len(paths)})"
            )
        sample = []
        for _ in range(n):
            random.shuffle(paths)
            tile = paths.pop()
            sample.append(tile)

        self.images = [
            tile.image_class(path, paths=paths, file_system=tile.file_system)
            for path in sorted(sample)
        ]
        return self

    @property
    def name(self) -> str:
        return self.tile_id

    def intersects(self, other: Any) -> bool:
        return self.unary_union.intersects(to_shapely(other))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        for image in self:
            print(image)
            return image.bounds

    @property
    def unary_union(self) -> Polygon:
        return box(*self.bounds)

    @property
    def crs(self) -> Any:
        for image in self:
            return image.crs

    def __lt__(self, other: "Tile") -> bool:
        """Makes Tiles sortable by tile_id."""
        return self.tile_id < other.tile_id

    # def __iter__(self):
    #     return iter(self.images)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, item: int | slice | list[int]) -> Image | Sequence[Image]:
        if isinstance(item, int):
            return self.image_class(
                self.image_paths[item],
                file_system=self.file_system,
                # file_paths=self.file_paths,
                df=self._df,
            )
        elif isinstance(item, slice):
            paths = [path for path in self.image_paths[item]]
        elif isinstance(item, Sequence) and not isinstance(item, str):
            paths = [self.image_paths[i] for i in item]
        else:
            raise TypeError(
                f"Tile indices must be int, slice or an iterable of int. Got {type(item)}"
            )

        with joblib.Parallel(n_jobs=self.processes, backend="threading") as parallel:
            return parallel(
                joblib.delayed(self.image_class)(
                    path,
                    file_system=self.file_system,
                    df=self._df,  # file_paths=self.file_paths
                )
                for path in paths
            )

    def __repr__(self) -> str:
        if len(self.image_paths) > 6:
            image_paths = (
                [str(tile) for tile in self.image_paths[:3]]
                + ["..."]
                + [str(tile) for tile in self.image_paths[-3:]]
            )
        else:
            image_paths = [str(tile) for tile in self.image_paths]
        return f"{self.__class__.__name__}(tile_id={self.tile_id}, image_paths={image_paths})"


class TileCollection(CollectionBase):
    image_regex: ClassVar[str | None] = None
    filename_regex: ClassVar[str | None] = None
    image_class: ClassVar[Image] = Image
    tile_class: ClassVar[Tile] = Tile

    def __init__(
        self,
        path: str | Path,
        level: str | None = None,
        processes: int = 1,
        res: int | None = None,
        file_system: dp.gcs.GCSFileSystem | None = None,
        df: pd.DataFrame | None = None,
    ) -> None:
        super().__init__()

        self.path = str(path)
        self.level = level
        self.processes = processes
        self.file_system = file_system
        self.res = res

        if df is not None:
            self._df = df
            return

        if self.image_regex:
            self._image_paths = [
                path
                for path in ls_func(self.path)
                if re.search(self.image_pattern, Path(path).name)
                and (self.level or "") in path
            ]
        else:
            self._image_paths = [
                path for path in ls_func(self.path) and (self.level or "") in path
            ]

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

        self._add_metadata_to_df(file_paths)

        # tile_ids = set()
        # for path in self.image_paths:
        #     try:
        #         tile_ids.add(
        #             re.match(self.image_pattern, Path(path).name).group("tile")
        #         )
        #     except AttributeError:
        #         continue

        # band_ids, dates = set(), set()
        # for path in self.file_paths:
        #     try:
        #         band_ids.add(
        #             re.match(self.filename_pattern, Path(path).name).group("band")
        #         )
        #     except AttributeError:
        #         continue
        #     try:
        #         dates.add(
        #             re.match(self.filename_pattern, Path(path).name).group("date")
        #         )
        #     except AttributeError:
        #         continue

        # self._band_ids = list(sorted(band_ids))
        # self._dates = list(sorted(dates))

        self._tiles = [
            self.tile_class(
                tile_id,
                level=self.level,
                df=self._df,
                res=self.res,
                # image_paths=self.image_paths,
                # file_paths=self.file_paths,
                file_system=self.file_system,
                processes=self.processes,
            )
            for tile_id in self.tile_ids
        ]

        self.images = self.get_images()

    @property
    def tiles(self) -> list[Tile]:
        return [tile for tile in self._tiles if tile.tile_id in self.tile_ids]

    # @tiles.setter
    # def tiles(self, values: list[Tile]) -> list[Tile]:
    #     self._tiles = list(values)
    #     if not all(isinstance(x, Tile) for x in self._tiles):
    #         raise TypeError("tiles should be a sequence of Tiles.")
    #     self._df = self._df.loc[lambda x: x["tile"].isin(self._tiles)]

    @property
    def tile_ids(self) -> list[str]:
        return list(sorted(self._df["tile"].dropna().unique()))

    # def load(self, bounds=None, indexes=1, nodata: int | None = None, **kwargs):
    #     bounds = to_bbox(bounds) if bounds is not None else None
    #     if isinstance(indexes, int):
    #         _indexes = (indexes,)
    #     else:
    #         _indexes = indexes

    #     for paths in self.file_paths:
    #         datasets = [_open_raster(path) for path in paths]

    #         arr, transform = rasterio.merge.merge(
    #             datasets,
    #             res=self.res,
    #             indexes=_indexes,
    #             bounds=bounds,
    #             nodata=nodata,
    #             **kwargs,
    #         )
    #         self.transform = transform
    #         if isinstance(indexes, int):
    #             arr = arr[0]

    #     return arr

    def filter(
        self,
        bands: str | list[str] | None = None,
        date_ranges: tuple[str, str] | None = None,
        max_cloud_cover: int | None = None,
        bounds=None,
    ) -> "TileCollection":
        pass

    def aggregate_dates(
        self,
        bands: str | list[str] | None = None,
        date_ranges: tuple[str, str] | None = None,
        max_cloud_cover: int | None = None,
        bounds=None,
        method: str = "mean",
    ):
        if method == "mean":
            _method = "sum"

        bounds = to_bbox(bounds) if bounds is not None else bounds

        arrs = []
        for date_collection in self.groupby("date"):
            for band_collection in date_collection.groupby("band"):
                print(band_collection.file_paths)
                datasets = [_open_raster(path) for path in band_collection.file_paths]
                arr, transform = rasterio.merge.merge(
                    datasets, res=self.res, indexes=(1,), method=_method
                )
                arr = arr[0]
                if method == "mean":
                    arr = arr / len(self.file_paths)
                print(arr.shape)
                arrs.append(arr)
        arrs = np.array(arrs)

        arrs = np.mean(arrs)

        self.transform = transform

        return arr

    def get_images(
        self,
        bands: str | list[str] | None = None,
        date_ranges: tuple[str, str] | None = None,
        bbox: Any | None = None,
        max_cloud_cover: int | None = None,
    ) -> list[Image]:
        copied = self.filter_bounds(bbox) if bbox is not None else self

        copied._images = get_images(
            copied.image_paths,
            level=self.level,
            bands=bands,
            df=self._df,
            date_ranges=date_ranges,
            image_regex=self.image_regex,
            processes=self.processes,
            image_class=self.image_class,
            max_cloud_cover=max_cloud_cover,
        )
        return copied

    def sample_images(self, n: int) -> list[Image]:
        tile: Tile = random.choice(self.tiles)
        return tile.sample_images(n=n)

    def get_tiles(
        self, bbox: GeoDataFrame | GeoSeries | Geometry | None = None
    ) -> list[Tile]:
        copied = self.filter_bounds(bbox) if bbox is not None else self
        return copied.tiles

    def get_dates(self, bbox: GeoDataFrame | GeoSeries | Geometry | None = None):
        copied = self.filter_bounds(bbox) if bbox is not None else self
        return copied.groupby("date").agg(merge)

    def sample_tiles(self, n: int) -> list[Image]:
        copied = deepcopy(self)
        if n > len(copied.tiles):
            raise ValueError(
                f"n ({n}) is higher than number of tiles in collection ({len(copied.tiles)})"
            )
        sample = []
        tiles_copy = [x for x in copied.tiles]
        for _ in range(n):
            random.shuffle(tiles_copy)
            tile = tiles_copy.pop()
            sample.append(tile)
        return sample

    def filter_bounds(
        self, other: GeoDataFrame | GeoSeries | Geometry | tuple
    ) -> "TileCollection":
        copied = deepcopy(self)

        with joblib.Parallel(n_jobs=copied.processes, backend="threading") as parallel:
            intersects_list: list[bool] = parallel(
                joblib.delayed(_intesects)(tile, other) for tile in copied
            )
        copied.tiles = [
            tile for tile, intersects in zip(copied, intersects_list) if intersects
        ]
        return copied

    @property
    def name(self) -> str:
        return Path(self.path).name

    def __iter__(self):
        return iter(self.tiles)

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, item: int | str | BoundingBox) -> Tile | TORCHGEO_RETURN_TYPE:
        if isinstance(item, int):
            return self.tiles[item]
        if isinstance(item, str):
            return self.tiles[self.tiles.index(item)]

        if not isinstance(item, BoundingBox):
            raise TypeError(
                "TileCollection indices must be int or BoundingBox. "
                f"Got {type(item)}"
            )

        images = self.get_images(bbox=item)

        crs = get_common_crs(images)

        data: torch.Tensor = torch.cat(
            [numpy_to_torch(image.load(indexes=1).values) for image in images]
        )

        key = "image"  # if self.is_image else "mask"
        sample = {key: data, "crs": crs, "bbox": item}

        return sample

    def __repr__(self) -> str:
        # if len(self.get_tiles()) > 6:
        #     tiles = (
        #         [str(tile) for tile in self.get_tiles()[:3]]
        #         + ["..."]
        #         + [str(tile) for tile in self.get_tiles()[-3:]]
        #     )
        # else:
        #     tiles = [str(tile) for tile in self.get_tiles()]
        return f"{self.__class__.__name__}(n_tiles={len(self.tiles)}, n_image_paths={len(self.image_paths)})"


class ImageCollection(CollectionBase):
    image_regex: ClassVar[str | None] = None
    filename_regex: ClassVar[str | None] = None
    image_class: ClassVar[Image] = Image

    def __init__(
        self,
        path: str | Path,
        level: str | None = None,
        processes: int = 1,
        res: int | None = None,
        file_system: dp.gcs.GCSFileSystem | None = None,
        df: pd.DataFrame | None = None,
    ) -> None:
        super().__init__()

        self.path = str(path)
        self.level = level
        self.processes = processes
        self.file_system = file_system
        self.res = res

        if df is not None:
            self._df = df
            return

        if self.image_regex:
            self._image_paths = [
                path
                for path in ls_func(self.path)
                if re.search(self.image_pattern, Path(path).name)
                and (self.level or "") in path
            ]
        else:
            self._image_paths = [
                path for path in ls_func(self.path) and (self.level or "") in path
            ]

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

        self._add_metadata_to_df(file_paths)

        self.images = self.get_images()

    @property
    def tile_ids(self) -> list[str]:
        return list(sorted(self._df["tile"].dropna().unique()))

    # def load(self, bounds=None, indexes=1, nodata: int | None = None, **kwargs):
    #     bounds = to_bbox(bounds) if bounds is not None else None
    #     if isinstance(indexes, int):
    #         _indexes = (indexes,)
    #     else:
    #         _indexes = indexes

    #     for paths in self.file_paths:
    #         datasets = [_open_raster(path) for path in paths]

    #         arr, transform = rasterio.merge.merge(
    #             datasets,
    #             res=self.res,
    #             indexes=_indexes,
    #             bounds=bounds,
    #             nodata=nodata,
    #             **kwargs,
    #         )
    #         self.transform = transform
    #         if isinstance(indexes, int):
    #             arr = arr[0]

    #     return arr

    def filter(
        self,
        bands: str | list[str] | None = None,
        date_ranges: tuple[str, str] | None = None,
        max_cloud_cover: int | None = None,
        bounds=None,
    ) -> "TileCollection":
        pass

    def aggregate_dates(
        self,
        bands: str | list[str] | None = None,
        date_ranges: tuple[str, str] | None = None,
        max_cloud_cover: int | None = None,
        bounds=None,
        method: str = "mean",
    ):
        if method == "mean":
            _method = "sum"

        bounds = to_bbox(bounds) if bounds is not None else bounds

        arrs = []
        for date_collection in self.groupby("date"):
            for band_collection in date_collection.groupby("band"):
                print(band_collection.file_paths)
                datasets = [_open_raster(path) for path in band_collection.file_paths]
                arr, transform = rasterio.merge.merge(
                    datasets, res=self.res, indexes=(1,), method=_method
                )
                arr = arr[0]
                if method == "mean":
                    arr = arr / len(self.file_paths)
                print(arr.shape)
                arrs.append(arr)
        arrs = np.array(arrs)

        arrs = np.mean(arrs)

        self.transform = transform

        return arr

    def get_images(
        self,
        bands: str | list[str] | None = None,
        date_ranges: tuple[str, str] | None = None,
        bbox: Any | None = None,
        max_cloud_cover: int | None = None,
    ) -> "ImageCollection":
        copied = self.filter_bounds(bbox) if bbox is not None else self

        copied._images = get_images(
            copied.image_paths,
            level=self.level,
            bands=bands,
            df=self._df,
            date_ranges=date_ranges,
            image_regex=self.image_regex,
            processes=self.processes,
            image_class=self.image_class,
            max_cloud_cover=max_cloud_cover,
        )

        return copied

    def sample_images(self, n: int) -> list[Image]:
        tile: Tile = random.choice(self.tiles)
        return tile.sample_images(n=n)

    def sample_tiles(self, n: int) -> list[Image]:
        copied = deepcopy(self)
        if n > len(copied.tiles):
            raise ValueError(
                f"n ({n}) is higher than number of tiles in collection ({len(copied.tiles)})"
            )
        sample = []
        tiles_copy = [x for x in copied.tiles]
        for _ in range(n):
            random.shuffle(tiles_copy)
            tile = tiles_copy.pop()
            sample.append(tile)
        return sample

    def filter_bounds(
        self, other: GeoDataFrame | GeoSeries | Geometry | tuple
    ) -> "TileCollection":
        copied = deepcopy(self)

        with joblib.Parallel(n_jobs=copied.processes, backend="threading") as parallel:
            intersects_list: list[bool] = parallel(
                joblib.delayed(_intesects)(image, other) for image in copied
            )
        copied.image = [
            image for image, intersects in zip(copied, intersects_list) if intersects
        ]
        return copied

    @property
    def name(self) -> str:
        return Path(self.path).name

    def __iter__(self):
        return iter(self.images)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(
        self, item: int | str | BoundingBox
    ) -> Image | TORCHGEO_RETURN_TYPE:
        if isinstance(item, int):
            return self.images[item]
        if isinstance(item, str):
            return self.images[self.images.index(item)]

        if not isinstance(item, BoundingBox):
            raise TypeError(
                "TileCollection indices must be int or BoundingBox. "
                f"Got {type(item)}"
            )

        images = self.get_images(bbox=item)

        crs = get_common_crs(images)

        data: torch.Tensor = torch.cat(
            [numpy_to_torch(image.load(indexes=1).values) for image in images]
        )

        key = "image"  # if self.is_image else "mask"
        sample = {key: data, "crs": crs, "bbox": item}

        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_tiles={len(self.tiles)}, n_image_paths={len(self.image_paths)})"


def get_images(
    image_paths: list[str],
    *,
    df: pd.DataFrame,
    max_cloud_cover: int | None,
    image_regex: str | None,
    level: str | None,
    bands: str | list[str] | None,
    date_ranges: (
        tuple[str | None, str | None] | tuple[tuple[str | None, str | None], ...]
    ),
    processes: int,
    image_class: Image,
) -> list[Image]:

    if image_regex:
        image_pattern = re.compile(image_regex, flags=re.VERBOSE)

    if date_ranges and not isinstance(date_ranges, (tuple, list)):
        raise TypeError(
            "date_ranges should be a 2-length tuple of strings or None, "
            "or a tuple of tuples for multiple date ranges"
        )

    relevant_paths: set[str] = {
        path
        for path in image_paths
        if (level or "") in path
        and (
            date_is_within(path, date_ranges, image_pattern)
            if date_ranges and image_regex
            else True
        )
    }

    with joblib.Parallel(n_jobs=processes, backend="threading") as parallel:
        images = parallel(
            joblib.delayed(image_class)(path, bands=bands, df=df)
            for path in relevant_paths
        )

    if max_cloud_cover is not None:
        images = [
            image for image in images if image.cloud_cover_percentage < max_cloud_cover
        ]

    if image_pattern:
        # sort by date
        images = list(sorted(images))

    return images


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


def _raster_load(raster, **kwargs):
    return raster.load(**kwargs)


def _raster_clip(raster, mask, **kwargs):
    return raster.clip(mask, **kwargs)


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
    for regex in regexes:
        if isinstance(regex, dict):
            out = {}
            for key, value in regex.items():
                try:
                    out[key] = re.search(value, xml_file).group(1)
                except (TypeError, AttributeError):
                    continue
            if len(out) != len(regex):
                raise RegexError()
            return out
        try:
            return re.search(regex, xml_file).group(1)
        except (TypeError, AttributeError):
            continue
    raise RegexError()


def _fix_path(path: str) -> str:
    return (
        str(path).replace("\\", "/").replace(r"\"", "/").replace("//", "/").rstrip("/")
    )


def _intesects(x, other) -> bool:
    return box(*x.bounds).intersects(box(*to_bbox(other)))


def date_is_within(
    path,
    date_ranges: (
        tuple[str | None, str | None] | tuple[tuple[str | None, str | None], ...] | None
    ),
    image_pattern: re.Pattern,
) -> bool:
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

        try:
            date = re.match(image_pattern, Path(path).name).group("date")
            if date > date_min and date < date_max:
                return True
        except AttributeError:
            pass

    return False


def _open_raster(path: str | Path) -> rasterio.io.DatasetReader:
    with opener(path) as file:
        return rasterio.open(file)


class Sentinel2Image(Image):
    image_regex: ClassVar[str] = config.SENTINEL2_IMAGE_REGEX
    filename_regex: ClassVar[str] = config.SENTINEL2_FILENAME_REGEX
    cloud_cover_regexes: ClassVar[tuple[str]] = config.CLOUD_COVERAGE_REGEXES
    cloud_band: str = "SCL"
    cloud_values: tuple[int] = (3, 8, 9, 10, 11)

    def __init__(
        self,
        path: str | Path,
        res: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(path, res=res, **kwargs)

    @property
    def level(self) -> str:
        if "L1C" in self.path:
            return "L1C"
        elif "L2A" in self.path:
            return "L2A"
        raise ValueError(self.path)


class Sentinel2Tile(Tile):
    image_regex: ClassVar[str] = config.SENTINEL2_IMAGE_REGEX
    filename_regex: ClassVar[str] = config.SENTINEL2_FILENAME_REGEX
    image_class: ClassVar[Image] = Sentinel2Image


class Sentinel2Collection(ImageCollection):
    image_regex: ClassVar[str] = config.SENTINEL2_IMAGE_REGEX
    filename_regex: ClassVar[str] = config.SENTINEL2_FILENAME_REGEX
    image_class: ClassVar[Sentinel2Image] = Sentinel2Image
    tile_class: ClassVar[Sentinel2Tile] = Sentinel2Tile
