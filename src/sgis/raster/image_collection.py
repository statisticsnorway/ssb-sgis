import abc
import functools
from copy import deepcopy, copy
import re
from pathlib import Path
from typing import Any
from typing import Sequence
from typing import ClassVar
import random

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
from ..helpers import get_all_files
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
        self.bands = None
        # self.date_ranges = None
        self.resolution = None
        self.max_cloud_cover = None

        if self.filename_regex:
            self.filename_pattern = re.compile(self.filename_regex, flags=re.VERBOSE)
        else:
            self.filename_pattern = None

        if self.image_regex:
            self.image_pattern = re.compile(self.image_regex, flags=re.VERBOSE)
        else:
            self.image_pattern = None

    # def update(self, **kwargs):
    #     for key, value in kwargs.items():
    #         setattr(self, key, value)
    #     return self


#     @property
#     def _attributes_to_inherite(self) -> dict[str, Any]:
#         return _make_attributes_dict_if_hasattr(self, ["resolution", "bands"])
#         return {
#             attr: getattr(self, attr)
#             for attr in ["resolution", "bands"]
#             if hasattr(self, attr)
#         }
#         return {
#             "resolution": self.resolution,
#             "bands": self.bands,
#         }


# def _make_attributes_dict_if_hasattr(obj, attributes: list[str]):
#     return {attr: getattr(obj, attr) for attr in attributes if hasattr(obj, attr)}


class CollectionBase(ImageBase):

    # def __init__(
    #     self,
    #     bands: str | Sequence[str] | None = None,
    #     bbox:
    #     date_ranges: (
    #         tuple[str | None, str | None] | tuple[tuple[str | None, str | None], ...]
    #     ) | None = None,
    # ) -> None:
    #     super().__init__()
    #     self.bands = bands

    #     if self.bands:
    #         self = self.filter_bands(bands)
    #     if self.bbox:
    #         self = self.filter_bounds(bbox)
    #     if self.date_ranges:
    #         self = self.filter_dates(date_ranges)

    def filter_dates(
        self,
        date_ranges: (
            tuple[str | None, str | None] | tuple[tuple[str | None, str | None], ...]
        ),
    ):
        # self.date_ranges = date_ranges

        if not isinstance(date_ranges, (tuple, list)):
            raise TypeError(
                "date_ranges should be a 2-length tuple of strings or None, "
                "or a tuple of tuples for multiple date ranges"
            )

        if all(x is None or isinstance(x, str) for x in date_ranges):
            date_ranges = (date_ranges,)

        self.image_paths = [
            path
            for path in self.image_paths
            for date_range in date_ranges
            if date_is_within(path, *date_range, self.image_pattern)
        ]

        return self

    def filter_bands(self, bands: str | Sequence[str]):
        if isinstance(bands, str):
            self.bands = [bands]
        else:
            self.bands = list(bands)
        self.file_paths = [
            path for path in self.file_paths if any(band in path for band in self.bands)
        ]

        self.bands = bands
        return self

    def filter_max_cloud_cover(self, max_cloud_cover: int):
        self.max_cloud_cover = max_cloud_cover
        return self

    def to_crs(self, new_crs):
        pass

    def set_resolution(self, resolution: int):
        self.resolution = resolution
        return self

    @property
    def _attributes_to_inherite(self) -> dict[str, Any]:
        return super()._attributes_to_inherite | _make_attributes_dict_if_hasattr(
            self, ["max_cloud_cover"]
        )

        return super()._attributes_to_inherite | {
            "max_cloud_cover": self.max_cloud_cover,
        }
        return {
            "resolution": self.resolution,
            "max_cloud_cover": self.max_cloud_cover,
        }


class Image(ImageBase):
    filename_regex: ClassVar[str | None] = None
    image_regex: ClassVar[str | None] = None
    cloud_cover_regexes: ClassVar[tuple[str] | None] = None
    # crs_regexes: ClassVar[tuple[str] | None] = None
    # bounds_regexes: ClassVar[tuple[dict[str, str]] | None] = None

    def __init__(
        self,
        path: str | Path,
        res: int | None = None,
        # crs: Any | None = None,
        bands: str | list[str] | None = None,
        file_paths: list[str] | None = None,
        file_system: dp.gcs.GCSFileSystem | None = None,
    ) -> None:
        super().__init__()

        self.path = str(path)
        self.res = res
        # self._crs = crs
        self.file_system = file_system

        if file_paths is None:
            self.file_paths = list(sorted(ls_func(self.path)))
        else:
            path_fixed = _fix_path(self.path)
            self.file_paths = list(
                sorted(path for path in file_paths if path_fixed in _fix_path(path))
            )

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

        # if self.crs_regexes:
        #     self._crs = get_cloud_percentage_in_local_dir(
        #         self.file_paths, regexes=self.crs_regexes
        #     )
        #     self._crs = pyproj.CRS(self._crs)
        # else:
        #     self._crs = None

        # if self.bounds_regexes:
        #     corner = get_cloud_percentage_in_local_dir(
        #         self.file_paths, regexes=self.bounds_regexes
        #     )
        #     maxy = int(corner["maxy"])
        #     minx = int(corner["minx"])
        #     miny = maxy - 110_000
        #     maxx = minx + 110_000
        #     self._bounds = (minx, miny, maxx, maxy)

        # else:
        #     self._bounds = None

        if self.filename_regex:
            self.file_paths = [
                path
                for path in self.file_paths
                if re.search(self.filename_pattern, Path(path).name)
            ]

        if bands:
            if isinstance(bands, str):
                self.bands = [bands]
            else:
                self.bands = list(bands)
            self.file_paths = [
                path
                for path in self.file_paths
                if any(band in path for band in self.bands)
            ]

        elif self.filename_regex:
            self.bands = [
                re.search(self.filename_pattern, Path(path).name).group("band")
                for path in self.file_paths
            ]
        else:
            self.bands = None
        # if res is not None:
        #     self.data = [
        #         Raster.from_path(
        #             path,
        #             res=res,
        #             file_system=self.file_system,
        #             filename_regex=self.filename_regex,
        #         )
        #         for path in self.file_paths
        #     ]
        # else:
        #     self.data = [
        #         Raster.from_path(
        #             path,
        #             res=res,
        #             file_system=self.file_system,
        #             filename_regex=self.filename_regex,
        #         )
        #         for path, res in zip(
        #             self.file_paths, self.resolutions.values(), strict=False
        #         )
        #     ]

        with opener(self.file_paths[0], file_system=self.file_system) as file:
            with rasterio.open(file) as src:
                self._bounds = src.bounds
                # if crs is None:
                #     self._crs = src.crs

                self._crs = src.crs

        # dictionary of filepaths and arrays, will be filled when bands are used
        self.raster_dict = {}

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def tile(self) -> str:
        return re.match(self.image_pattern, self.name).group("tile")

    @property
    def date(self) -> str:
        return re.match(self.image_pattern, self.name).group("date")

    # def load(self, **kwargs) -> "Image":
    #     relevant_paths = [path if raster.path not in self.raster_dict)

    #     with joblib.Parallel(n_jobs=self.processes, backend="threading") as parallel:
    #         rasters = parallel(joblib.delayed(_raster_load)(raster, **kwargs) for raster in self if raster.path not in self.raster_dict)

    #     for path
    #     return self

    # def clip(self, mask, copy: bool = True, **kwargs) -> "Image":
    #     copy = self.copy() if copy else self
    #     with joblib.Parallel(n_jobs=copy.processes, backend="threading") as parallel:
    #         parallel(
    #             joblib.delayed(_raster_clip)(raster, mask, **kwargs) for raster in copy
    #         )

    #     return copy

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
        return self.clip(boxes, **kwargs)

    def get_band(self, band: str) -> Raster:
        try:
            path = self.get_path(band)
        except KeyError as e:
            if band in self.file_paths:
                path = band
            else:
                raise e
        if path in self.raster_dict:
            return self.raster_dict[path]
        raster = Raster.from_path(
            path,
            res=self.res,
            file_system=self.file_system,
            filename_regex=self.filename_regex,
        )
        self.raster_dict[path] = raster
        return raster

    def get_bands(self) -> list[Raster]:
        if self.bands:
            return [self.get_band(band) for band in self.bands]
        else:
            return [self.get_band(path) for path in self.file_paths]

    def get_arrays(self) -> list[np.ndarray]:
        return [raster.values for raster in self.get_bands()]

    def __getitem__(self, band: str | list[str]) -> np.ndarray:
        if isinstance(band, str):
            return self.get_band(band)
        return [self.get_band(x) for x in band]

    def __iter__(self):
        return iter(self.bands)

    def __lt__(self, other: "Image") -> bool:
        """Makes the tiles sortable by tile_id"""
        return self.date < other.date

    def keys(self):
        return iter(self.bands)

    def values(self):
        return iter(self.file_paths)

    def items(self):
        return iter(zip(self.bands, self.file_paths))

    def __len__(self) -> int:
        return len(self.bands)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bands={self.bands})"

    def get_cloud_array(self) -> np.ndarray:
        scl = self[self.cloud_band].load()
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
    def crs(self) -> str:
        return self._crs
        return get_common_crs(self.data)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self._bounds
        return get_total_bounds(self.data)

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
        return get_total_bounds(self.data)

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

    @property
    def level(self) -> str:
        raise NotImplementedError()

    # @property
    # def bands(self) -> list[str]:
    #     raise NotImplementedError()

    # @property
    # def resolutions(self) -> dict[str, int]:
    #     raise NotImplementedError()

    def copy(self, deep: bool = True) -> "Image":
        """Returns a (deep) copy of the class instance and its attributes.

        Args:
            deep: Whether to return a deep or shallow copy. Defaults to True.
        """
        copied = deepcopy(self) if deep else copy(self)
        copied.data = [raster.copy() for raster in copied]
        return copied


class Tile(CollectionBase):
    filename_regex: ClassVar[str | None] = None
    image_regex: ClassVar[str | None] = None
    image_class: ClassVar[Image] = Image

    def __init__(
        self,
        tile_id: str,
        image_paths: list[str],
        file_paths: list[str],
        level: str | None,
        file_system: dp.gcs.GCSFileSystem | None,
        processes: int,
    ) -> None:
        super().__init__()

        self.tile_id = str(tile_id)
        self.level = level
        self.file_system = file_system
        self.processes = processes
        self.file_paths = file_paths

        self.image_paths = [
            path
            for path in image_paths
            if self.tile_id in path and (self.level or "") in path
        ]

        if self.image_regex:
            # sort by date
            dates = [
                re.search(self.image_pattern, Path(path).name).group("date")
                for path in self.image_paths
            ]
            self.image_paths = list(
                pd.Series(self.image_paths, index=dates).sort_index()
            )

        # self._images = self.get_images()
        # self._images = [None for _ in range(len(self.image_paths))]

    def get_images(
        self,
        bands: str | list[str] | None = None,
        date_ranges: tuple[str, str] | None = None,
        # bbox: Any | None = None,
        max_cloud_cover: int | None = None,
    ) -> list[Image]:
        return get_images(
            self.image_paths,
            level=self.level,
            bands=bands,
            file_paths=self.file_paths,
            date_ranges=date_ranges,
            # bbox=None,  # bbox,
            image_regex=self.image_regex,
            processes=self.processes,
            image_class=self.image_class,
            max_cloud_cover=max_cloud_cover,
        )

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

        return [
            tile.image_class(path, paths=paths, file_system=tile.file_system)
            for path in sorted(sample)
        ]

    # @property
    # def images(self) -> list[Image]:
    #     try:
    #         return self._images
    #     except AttributeError as e:
    #         raise AttributeError(
    #             "Images aren't loaded yet. Do so with the get_images or sample_images method."
    #         ) from e

    # @images.setter
    # def images(self, new_value) -> list[Image]:
    #     if not all(isinstance(x, Image) for x in new_value):
    #         raise TypeError("images attribute must be an iterable of images.")
    #     self._images = list(new_value)

    @property
    def name(self) -> str:
        return self.tile_id

    def intersects(self, other: Any) -> bool:
        return self.unary_union.intersects(to_shapely(other))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        for image in self:
            return image.bounds

    @property
    def unary_union(self) -> Polygon:
        return box(*self.bounds)

    @property
    def crs(self) -> Any:
        for image in self:
            return image.crs

    def __lt__(self, other: "Tile") -> bool:
        """Makes the tiles sortable by tile_id"""
        return self.tile_id < other.tile_id

    # def __iter__(self):
    #     return iter(self.images)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, item: int | slice | list[int]) -> Image | Sequence[Image]:
        if isinstance(item, int):
            return self.image_class(
                self.image_paths[item],
                file_system=self.file_system,
                file_paths=self.file_paths,
            )
        elif isinstance(item, slice):
            return [
                self.image_class(
                    path,
                    file_system=self.file_system,
                    file_paths=self.file_paths,
                )
                for path in self.image_paths[item]
            ]
        elif isinstance(item, Sequence) and not isinstance(item, str):
            return [
                self.image_class(
                    self.image_paths[i],
                    file_system=self.file_system,
                    file_paths=self.file_paths,
                )
                for i in item
            ]
        else:
            raise TypeError(
                f"Tile indices must be int, slice or an iterable of int. Got {type(item)}"
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
        file_system: dp.gcs.GCSFileSystem | None = None,
    ) -> None:
        super().__init__()

        self.path = str(path)
        self.level = level
        self.processes = processes
        self.file_system = file_system

        if self.filename_regex:
            self.filename_pattern = re.compile(self.filename_regex, flags=re.VERBOSE)
        else:
            self.filename_pattern = None
        if self.image_regex:
            self.image_pattern = re.compile(self.image_regex, flags=re.VERBOSE)
            self.image_paths = [
                path
                for path in ls_func(self.path)
                if re.search(self.image_pattern, Path(path).name)
                and (self.level or "") in path
            ]
        else:
            self.image_pattern = None
            self.image_paths = [
                path for path in ls_func(self.path) and (self.level or "") in path
            ]

        self.file_paths = set(
            glob_func(self.path + "/**/**")
            + glob_func(self.path + "/**/**/**")
            + glob_func(self.path + "/**/**/**/**")
            + glob_func(self.path + "/**/**/**/**/**")
        )

        tile_ids = set()
        for path in self.image_paths:
            try:
                tile_ids.add(
                    re.match(self.image_pattern, Path(path).name).group("tile")
                )
            except AttributeError:
                continue

        self._tiles = [
            self.tile_class(
                tile_id,
                level=self.level,
                image_paths=self.image_paths,
                file_paths=self.file_paths,
                # filename_regex=self.filename_regex,
                # image_regex=self.image_regex,
                file_system=self.file_system,
                # image_class=self.image_class,
                processes=self.processes,
            )  # .update(**self._attributes_to_inherite)
            for tile_id in sorted(tile_ids)
        ]

    @property
    def tiles(self) -> list[Tile]:
        return self._tiles

    @tiles.setter
    def tiles(self, values: list[Tile]) -> list[Tile]:
        self._tiles = list(values)
        if not all(isinstance(x, Tile) for x in self._tiles):
            raise TypeError("tiles should be a sequence of Tiles.")
        self.image_paths = [
            path
            for path in self.image_paths
            if any(tile.tile_id in path for tile in self._tiles)
        ]

    def get_images(
        self,
        bands: str | list[str] | None = None,
        date_ranges: tuple[str, str] | None = None,
        bbox: Any | None = None,
        max_cloud_cover: int | None = None,
    ) -> list[Image]:
        copied = self.filter_bounds(bbox) if bbox is not None else self

        return get_images(
            copied.image_paths,
            level=self.level,
            bands=bands,
            file_paths=self.file_paths,
            date_ranges=date_ranges,
            # bbox=bbox,
            image_regex=self.image_regex,
            processes=self.processes,
            image_class=self.image_class,
            max_cloud_cover=max_cloud_cover,
        )

    def sample_images(self, n: int) -> list[Image]:
        tile: Tile = random.choice(self.tiles)
        return tile.sample_images(n=n)

    def get_tiles(
        self, bbox: GeoDataFrame | GeoSeries | Geometry | None = None
    ) -> list:
        copied = self.filter_bounds(bbox) if bbox is not None else self
        return copied.tiles

    def get_tile(self, tile_id: str) -> Tile:
        matches = [tile for tile in self.get_tiles() if tile_id == tile.tile_id]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            prefix = "Multiple"
        else:
            prefix = "No"

        raise KeyError(
            f"{prefix} matches for tile_id {tile_id} among tile_ids {[tile.tile_id for tile in self.get_tiles()]}"
        )

    def sample_tiles(self, n: int) -> "TileCollection":
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
        copied.tiles = list(sorted(sample))
        return copied

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

    # def filter_dates(
    #     self,
    #     date_ranges: (
    #         tuple[str | None, str | None] | tuple[tuple[str | None, str | None], ...]
    #     ),
    # ):
    #     self.date_ranges = date_ranges
    #     return self

    # def filter_bands(self, bands: str | Sequence[str]):
    #     self.bands = bands
    #     return self

    # def filter_max_cloud_cover(self, max_cloud_cover: int):
    #     self.max_cloud_cover = max_cloud_cover
    #     return self

    @property
    def name(self) -> str:
        return Path(self.path).name

    def __iter__(self):
        return iter(self.tiles)

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, item: int | BoundingBox) -> Tile | TORCHGEO_RETURN_TYPE:
        if isinstance(item, int):
            return self.tiles[item]

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
        if len(self.get_tiles()) > 6:
            tiles = (
                [str(tile) for tile in self.get_tiles()[:3]]
                + ["..."]
                + [str(tile) for tile in self.get_tiles()[-3:]]
            )
        else:
            tiles = [str(tile) for tile in self.get_tiles()]
        return f"{self.__class__.__name__}(tiles={tiles}, nimage_paths={len(self.image_paths)})"


def get_images(
    image_paths: list[str],
    *,
    file_paths: list[str],
    max_cloud_cover: int | None,
    image_regex: str | None,
    level: str | None,
    bands: str | list[str] | None,
    date_ranges: (
        tuple[str | None, str | None] | tuple[tuple[str | None, str | None], ...]
    ),
    # bbox: Any | None,
    processes: int,
    image_class: Image,
) -> list[Image]:

    if image_regex:
        image_pattern = re.compile(image_regex, flags=re.VERBOSE)

    if not isinstance(date_ranges, (tuple, list)):
        raise TypeError(
            "date_ranges should be a 2-length tuple of strings or None, "
            "or a tuple of tuples for multiple date ranges"
        )

    if date_ranges is None:
        date_ranges = (None,)
    elif all(x is None or isinstance(x, str) for x in date_ranges):
        date_ranges = (date_ranges,)

    with joblib.Parallel(n_jobs=processes, backend="threading") as parallel:
        images = parallel(
            joblib.delayed(image_class)(path, bands=bands, file_paths=file_paths)
            for path in image_paths
            for date_range in date_ranges
            if (level or "") in path
            and (
                date_is_within(path, *date_range, image_pattern) if date_range else True
            )
            # ]
            # and (re.search(image_pattern, Path(path).name) if image_pattern else 1)
        )

    if max_cloud_cover is not None:
        images = [image for image in images if image.cloud_cover < max_cloud_cover]

    # if bbox is not None:
    #     if isinstance(bbox, BoundingBox):
    #         date_ranges = (bbox.mint, bbox.maxt)
    #     bbox = box(*to_bbox(bbox))
    #     images = [image for image in images if box(*image.bounds).intersects(bbox)]

    if image_pattern:
        images = list(sorted(images))
        # dates = [
        #     re.search(image_pattern, Path(image.path).name).group("date")
        #     for image in images
        # ]
        # images = list(pd.Series(images, index=dates).sort_index())

    return images

    if date_ranges is None:
        return images

    try:
        date_min, date_max = date_ranges
        assert isinstance(date_min, str)
        assert len(date_min) == 8
        assert isinstance(date_max, str)
        assert len(date_max) == 8
    except AssertionError:
        raise TypeError(
            "date_ranges should be a tuple of two 8-charactered strings (start and end date)."
        )

    return [
        image for image in images if image.date >= date_min and image.date <= date_max
    ]


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
    path, date_min: str, date_max: str, image_pattern: re.Pattern
) -> bool:
    try:
        assert date_min is None or isinstance(date_min, str)
        assert date_min is None or len(date_min) == 8
        assert date_max is None or isinstance(date_max, str)
        assert date_max is None or len(date_max) == 8
    except AssertionError:
        raise TypeError(
            "date_ranges should be a tuple of two 8-charactered strings (start and end date)."
        )

    try:
        date = re.match(image_pattern, Path(path).name).group("date")
        return date > date_min and date < date_max
    except AttributeError:
        return False


class Sentinel2Image(Image):
    image_regex: ClassVar[str] = config.SENTINEL2_IMAGE_REGEX
    filename_regex: ClassVar[str] = config.SENTINEL2_FILENAME_REGEX
    cloud_cover_regexes: ClassVar[tuple[str]] = config.CLOUD_COVERAGE_REGEXES
    # crs_regexes: ClassVar[tuple[str]] = config.CRS_REGEX
    # bounds_regexes: ClassVar[tuple[dict[str, str]]] = config.BOUNDS_REGEX
    cloud_band: str = "SCL"
    cloud_values: tuple[int] = (8, 7, 6)
    sample_size: int = 1000

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


class Sentinel2Collection(TileCollection):
    image_regex: ClassVar[str] = config.SENTINEL2_IMAGE_REGEX
    filename_regex: ClassVar[str] = config.SENTINEL2_FILENAME_REGEX
    image_class: ClassVar[Sentinel2Image] = Sentinel2Image
    tile_class: ClassVar[Sentinel2Tile] = Sentinel2Tile
