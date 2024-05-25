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
from shapely.geometry import Point, Polygon
import torch
import pyproj
from geopandas import GeoSeries


try:
    from torchgeo.datasets.utils import BoundingBox
except ImportError:

    class BoundingBox:
        """Placeholder."""

        def __init__(self, *args, **kwargs) -> None:
            """Placeholder."""
            raise ImportError("missing optional dependency 'torchgeo'")


from ..io._is_dapla import is_dapla
from . import sentinel_config as config
from ..geopandas_tools.bounds import to_bbox, get_total_bounds
from ..geopandas_tools.general import get_common_crs
from ..geopandas_tools.conversion import to_gdf
from .raster import Raster
from ..helpers import get_all_files

if is_dapla():
    ls_func = lambda *args, **kwargs: dp.FileClient.get_gcs_file_system().ls(
        *args, **kwargs
    )
else:
    ls_func = functools.partial(get_all_files, recursive=False)

TORCHGEO_RETURN_TYPE = dict[str, torch.Tensor | pyproj.CRS | BoundingBox]


class ImageBase:
    def get_images(
        self,
        level: str | None = None,
        bands: str | list[str] | None = None,
        date_range: tuple[str, str] | None = None,
        bbox: Any | None = None,
    ) -> list["Image"]:
        return get_images(
            self.image_paths,
            level=level,
            bands=bands,
            date_range=date_range,
            bbox=bbox,
            filename_regex=self.filename_regex,
            image_regex=self.image_regex,
        )

    def sample_images(self, n: int) -> "ImageBase":
        images = self.get_images()
        if n > len(images):
            raise ValueError(
                f"n ({n}) is higher than number of images in collection ({len(images)})"
            )
        sample = []
        for _ in range(n):
            random.shuffle(images)
            tile = images.pop()
            sample.append(tile)

        return sample


class ImageCollection(ImageBase):
    image_regex: ClassVar[str]
    filename_regex: ClassVar[str]

    def __init__(
        self,
        path: str | Path,
        file_system: dp.gcs.GCSFileSystem | None = None,
    ) -> None:
        self.path = str(path)
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
            ]
        else:
            self.image_pattern = None
            self.image_paths = [path for path in ls_func(self.path)]

        tile_ids = set()
        for path in self.image_paths:
            try:
                tile_ids.add(
                    re.match(self.image_pattern, Path(path).name).group("tile")
                )
            except AttributeError:
                continue

        self._tiles = [
            ImageTile(
                tile_id,
                paths=self.image_paths,
                filename_regex=self.filename_regex,
                root=self.path,
                file_system=self.file_system,
            )
            for tile_id in tile_ids
        ]

    def get_tiles(self) -> list:
        return self._tiles

    def sample_tiles(self, n: int) -> "ImageCollection":
        if n > len(self._tiles):
            raise ValueError(
                f"n ({n}) is higher than number of tiles in collection ({len(self._tiles)})"
            )
        sample = []
        tiles_copy = [x for x in self._tiles]
        for _ in range(n):
            random.shuffle(tiles_copy)
            tile = tiles_copy.pop()
            sample.append(tile)
        return sample

    def sample_change(self, size: int = 1000) -> "list[Image, Image]":
        oldest_and_newest_image = self.sample_tiles(1)[0][[0, -1]]
        bounds = GeoSeries(oldest_and_newest_image[0].unary_union)
        sample_area = bounds.sample_points(1).buffer(size / 2).clip(bounds)
        return [image.clip(sample_area) for image in oldest_and_newest_image]

    @property
    def name(self) -> str:
        return Path(self.path).name

    def __getitem__(self, bbox: BoundingBox) -> TORCHGEO_RETURN_TYPE:
        images = self.get_images(bbox=bbox)

        crs = get_common_crs(images)

        data: torch.Tensor = torch.cat(
            [numpy_to_torch(image.load(indexes=1).values) for image in images]
        )

        key = "image"  # if self.is_image else "mask"
        sample = {key: data, "crs": crs, "bbox": bbox}

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
        return f"{self.__class__.__name__}(tiles={tiles}, n_image_paths={len(self.image_paths)})"


class Sentinel2(ImageCollection):
    image_regex: ClassVar[str] = config.SENTINEL2_IMAGE_REGEX
    filename_regex: ClassVar[str] = config.SENTINEL2_FILENAME_REGEX


class Image:
    def __init__(
        self,
        path: str | Path,
        paths: list[str] | None = None,
        bands: str | list[str] | None = None,
        res: int | None = None,
        filename_regex: str = config.SENTINEL2_FILENAME_REGEX,
        image_regex: str = config.SENTINEL2_IMAGE_REGEX,
        file_system: dp.gcs.GCSFileSystem | None = None,
    ) -> None:
        self.path = str(path)
        self.filename_regex = filename_regex
        self.filename_pattern = re.compile(filename_regex, flags=re.VERBOSE)
        self.image_regex = image_regex
        self.image_pattern = re.compile(image_regex, flags=re.VERBOSE)
        assert self.level, (self.path, filename_regex)

        self.file_system = file_system

        if paths is None:
            self.filepaths = ls_func(self.path)
        else:
            self.filepaths = paths

        if filename_regex:
            self.filepaths = [
                path
                for path in self.filepaths
                if re.search(self.filename_pattern, Path(path).name)
            ]
        if bands:
            if isinstance(bands, str):
                bands = [bands]
            self.filepaths = [
                file for file in self.filepaths if any(band in file for band in bands)
            ]
        if res is not None:
            self.data = [
                Raster.from_path(
                    path,
                    res=res,
                    file_system=self.file_system,
                    filename_regex=self.filename_regex,
                )
                for path in self.filepaths
            ]
        else:
            self.data = [
                Raster.from_path(
                    path,
                    res=res,
                    file_system=self.file_system,
                    filename_regex=self.filename_regex,
                )
                for path, res in zip(
                    self.filepaths, self.resolutions.values(), strict=False
                )
            ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data={self.data})"

    def load(self, **kwargs) -> "Image":
        for raster in self:
            raster.load(**kwargs)
        return self

    def clip(self, mask, **kwargs) -> "Image":
        copy = self.copy()
        clipped = [raster.copy().clip(mask, **kwargs) for raster in copy]
        copy.data = clipped
        return copy

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
        print(self.__dict__)
        return self.clip(boxes, **kwargs)

    def copy(self, deep: bool = True) -> "Image":
        """Returns a (deep) copy of the class instance and its attributes.

        Args:
            deep: Whether to return a deep or shallow copy. Defaults to True.
        """
        copied = deepcopy(self) if deep else copy(self)
        copied.data = [raster.copy() for raster in copied]
        return copied

    def __getitem__(self, band: str) -> Raster:
        simple_string_match = [
            raster for raster in self.data if str(band) in raster.path
        ]
        if len(simple_string_match) == 1:
            return simple_string_match[0]

        regex_matches = []
        for raster in self.data:
            match_ = re.search(self.filename_pattern, Path(raster.path).name)
            if match_ and str(band) == match_.group("band"):
                regex_matches.append(raster)

        if len(regex_matches) == 1:
            return regex_matches[0]

        if len(regex_matches) > 1:
            prefix = "Multiple"
        elif not regex_matches:
            prefix = "No"

        raise ValueError(
            f"{prefix} matches for band {band} among paths {[Path(x).name for x in self.filepaths]}"
        )

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    @property
    def centroid(self) -> str:
        x = (self.bounds[0] + self.bounds[2]) / 2
        y = (self.bounds[1] + self.bounds[3]) / 2
        return Point(x, y)

    @property
    def crs(self) -> str:
        return get_common_crs(self.data)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return get_total_bounds(self.data)

    @property
    def unary_union(self) -> Polygon:
        return box(*self.bounds)

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
    def format(self) -> str:
        return Path(self.path).suffix.strip(".")

    def get_path(self, band: str) -> str:
        simple_string_match = [path for path in self.filepaths if str(band) in path]
        if len(simple_string_match) == 1:
            return simple_string_match[0]

        regex_matches = []
        for path in self.filepaths:
            match_ = re.search(self.filename_pattern, Path(path).name)
            if match_ and str(band) == match_.group("band"):
                regex_matches.append(path)

        if len(regex_matches) == 1:
            return regex_matches[0]

        if len(regex_matches) > 1:
            prefix = "Multiple"
        elif not regex_matches:
            prefix = "No"

        raise ValueError(
            f"{prefix} matches for band {band} among paths {[Path(x).name for x in self.filepaths]}"
        )

    @property
    def level(self) -> str:
        if "L1C" in self.path:
            return "L1C"
        elif "L2A" in self.path:
            return "L2A"
        raise ValueError(self.path)

    @property
    def bands(self) -> list[str]:
        return get_bands_for_image(self.path)

    @property
    def resolutions(self) -> dict[str, int]:
        return get_band_resolutions_for_image(self.path)


class ImageTile(ImageBase):

    def __init__(
        self,
        tile_id: str,
        paths: list[str] | None = None,
        root: str | None = None,
        filename_regex: str = config.SENTINEL2_FILENAME_REGEX,
        image_regex: str = config.SENTINEL2_IMAGE_REGEX,
        file_system: dp.gcs.GCSFileSystem | None = None,
    ) -> None:
        self.tile_id = str(tile_id)
        self.file_system = file_system
        self.filename_regex = filename_regex
        self.image_regex = image_regex
        if not paths and not root:
            raise ValueError("Must specify either 'paths' or 'root'")
        if not paths:
            self.image_paths = [path for path in ls_func(root) if self.tile_id in path]
        else:
            self.image_paths = [path for path in paths if self.tile_id in path]

        self.image_regex = image_regex
        self.filename_regex = filename_regex
        self.image_pattern = re.compile(self.image_regex, flags=re.VERBOSE)
        self.filename_pattern = re.compile(self.filename_regex, flags=re.VERBOSE)

        dates = [
            re.search(self.image_pattern, Path(path).name).group("date")
            for path in self.image_paths
        ]

        self.image_paths = list(pd.Series(self.image_paths, index=dates).sort_index())

    @property
    def name(self) -> str:
        return self.tile_id

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        for image in self:
            return image.bounds

    @property
    def crs(self) -> Any:
        for image in self:
            return image.crs

    def __getitem__(self, item: int | slice | list[int]) -> Image | Sequence[Image]:
        if isinstance(item, int):
            return Image(
                self.image_paths[item],
                file_system=self.file_system,
            )
        elif isinstance(item, slice):
            return [
                Image(
                    path,
                    file_system=self.file_system,
                )
                for path in self.image_paths[item]
            ]
        elif isinstance(item, Sequence) and not isinstance(item, str):
            return [
                Image(
                    self.image_paths[i],
                    file_system=self.file_system,
                )
                for i in item
            ]
        else:
            raise TypeError(f"ImageTile indices must be int or slice. Got {type(item)}")

    def __repr__(self) -> str:
        if len(self.image_paths) > 6:
            image_paths = (
                [str(tile) for tile in self.image_paths[:3]]
                + ["..."]
                + [str(tile) for tile in self.image_paths[-3:]]
            )
        else:
            image_paths = [str(tile) for tile in self.image_paths]
        return f"{self.__class__.__name__}(tile_id={self.tile_id}, paths={image_paths})"


def get_images(
    image_paths: list[str],
    *,
    filename_regex: str | None = None,
    image_regex: str | None = None,
    level: str | None = None,
    bands: str | list[str] | None = None,
    date_range: tuple[str, str] | None = None,
    bbox: Any | None = None,
) -> list[Image]:

    if image_regex:
        image_regex = re.compile(image_regex, flags=re.VERBOSE)
        images = [
            Image(path, bands=bands, filename_regex=filename_regex)
            for path in image_paths
            if (level or "") in path and re.search(image_regex, Path(path).name)
        ]
    else:
        images = [
            Image(path, bands=bands, filename_regex=filename_regex)
            for path in image_paths
            if (level or "") in path
        ]

    if bbox is not None:
        if isinstance(bbox, BoundingBox):
            date_range = (bbox.mint, bbox.maxt)
        bbox = box(*to_bbox(bbox))
        images = [image for image in images if box(*image.bounds).intersects(bbox)]

    if image_regex:
        dates = [
            re.search(image_regex, Path(image.path).name).group("date")
            for image in images
        ]
        images = list(pd.Series(images, index=dates).sort_index())

    if date_range is None:
        return images

    try:
        date_min, date_max = date_range
        assert isinstance(date_min, str)
        assert len(date_min) == 8
        assert isinstance(date_max, str)
        assert len(date_max) == 8
    except AssertionError:
        raise TypeError(
            "date_range should be a tuple of two 8-charactered strings (start and end date)."
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
