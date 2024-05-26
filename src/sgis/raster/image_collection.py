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
import joblib
import glob
import rasterio

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
from ..geopandas_tools.conversion import to_gdf
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


class ImageBase:
    def get_images(
        self,
        bands: str | list[str] | None = None,
        date_range: tuple[str, str] | None = None,
        bbox: Any | None = None,
    ) -> list["Image"]:
        return get_images(
            self.image_paths,
            level=self.level,
            bands=bands,
            paths=self.filepaths,
            date_range=date_range,
            bbox=bbox,
            image_regex=self.image_regex,
            processes=self.processes,
            image_type=self.image_type,
        )

    def sample_images(self, n: int) -> "ImageBase":
        paths = self.image_paths
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
            self.image_type(path, paths=paths, file_system=self.file_system)
            for path in sample
        ]


class Image:
    filename_regex: ClassVar[str | None] = None
    image_regex: ClassVar[str | None] = None
    cloud_cover_regexes: ClassVar[tuple[str] | None] = None
    crs_regexes: ClassVar[tuple[str] | None] = None
    bounds_regexes: ClassVar[tuple[dict[str, str]] | None] = None

    def __init__(
        self,
        path: str | Path,
        paths: list[str] | None = None,
        bands: str | list[str] | None = None,
        res: int | None = None,
        file_system: dp.gcs.GCSFileSystem | None = None,
    ) -> None:
        self.path = str(path)
        if self.filename_regex:
            self.filename_pattern = re.compile(self.filename_regex, flags=re.VERBOSE)
        else:
            self.filename_pattern = None
        if self.image_regex:
            self.image_pattern = re.compile(self.image_regex, flags=re.VERBOSE)
        else:
            self.image_pattern = None

        self.file_system = file_system

        if paths is None:
            self.filepaths = list(sorted(ls_func(self.path)))
        else:
            path_fixed = _fix_path(self.path)
            self.filepaths = list(
                sorted(path for path in paths if path_fixed in _fix_path(path))
            )

        if self.cloud_cover_regexes:
            try:
                self.cloud_cover_percentage = float(
                    get_cloud_percentage_in_local_dir(
                        self.filepaths, regexes=self.cloud_cover_regexes
                    )
                )
            except Exception:
                self.cloud_cover_percentage = None
        else:
            self.cloud_cover_percentage = None

        # if self.crs_regexes:
        #     self._crs = get_cloud_percentage_in_local_dir(
        #         self.filepaths, regexes=self.crs_regexes
        #     )
        #     self._crs = pyproj.CRS(self._crs)
        # else:
        #     self._crs = None

        # if self.bounds_regexes:
        #     corner = get_cloud_percentage_in_local_dir(
        #         self.filepaths, regexes=self.bounds_regexes
        #     )
        #     maxy = int(corner["maxy"])
        #     minx = int(corner["minx"])
        #     miny = maxy - 110_000
        #     maxx = minx + 110_000
        #     self._bounds = (minx, miny, maxx, maxy)

        # else:
        #     self._bounds = None

        if self.filename_regex:
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
        # if res is not None:
        #     self.data = [
        #         Raster.from_path(
        #             path,
        #             res=res,
        #             file_system=self.file_system,
        #             filename_regex=self.filename_regex,
        #         )
        #         for path in self.filepaths
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
        #             self.filepaths, self.resolutions.values(), strict=False
        #         )
        #     ]

        for path in self.filepaths:
            with opener(path, self.file_system) as file:
                with rasterio.open(file) as src:
                    self._bounds = src.bounds
                    self._crs = src.crs
            break

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bands={self.bands})"

    def load(self, **kwargs) -> "Image":
        with joblib.Parallel(n_jobs=self.processes, backend="threading") as parallel:
            parallel(joblib.delayed(_raster_load)(raster**kwargs) for raster in self)

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
        return self.clip(boxes, **kwargs)

    def copy(self, deep: bool = True) -> "Image":
        """Returns a (deep) copy of the class instance and its attributes.

        Args:
            deep: Whether to return a deep or shallow copy. Defaults to True.
        """
        copied = deepcopy(self) if deep else copy(self)
        copied.data = [raster.copy() for raster in copied]
        return copied

    def __getitem__(self, band: str) -> str:
        return self.get_band(band)

    def get_bands(self, res: int | None = None) -> list[Raster]:
        return [
            Raster.from_path(
                path,
                res=res,
                file_system=self.file_system,
                filename_regex=self.filename_regex,
            )
            for path in self.filepaths
        ]

    def get_band(self, band: str, res: int | None = None) -> Raster:
        path = self.get_path(band)
        return Raster.from_path(
            path,
            res=res,
            file_system=self.file_system,
            filename_regex=self.filename_regex,
        )

    # def __iter__(self):
    #     return iter(self.data)

    def __len__(self) -> int:
        return len(self.filepaths)

    @property
    def centroid(self) -> str:
        x = (self.bounds[0] + self.bounds[2]) / 2
        y = (self.bounds[1] + self.bounds[3]) / 2
        return Point(x, y)

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

        raise KeyError(
            f"{prefix} matches for band {band} among paths {[Path(x).name for x in self.filepaths]}"
        )

    @property
    def level(self) -> str:
        raise NotImplemented()

    @property
    def bands(self) -> list[str]:
        raise NotImplementedError()

    @property
    def resolutions(self) -> dict[str, int]:
        raise NotImplementedError()


class ImageTile(ImageBase):

    def __init__(
        self,
        tile_id: str,
        filename_regex: str,
        image_regex: str,
        image_paths: list[str],
        filepaths: list[str],
        level: str | None = None,
        file_system: dp.gcs.GCSFileSystem | None = None,
        image_type: Image = Image,
        processes: int = 1,
    ) -> None:
        self.tile_id = str(tile_id)
        self.level = level
        self.file_system = file_system
        self.processes = processes
        self.image_type = image_type
        self.filename_regex = filename_regex
        self.image_regex = image_regex

        self.image_paths = [
            path
            for path in image_paths
            if self.tile_id in path and (self.level or "") in path
        ]
        self.filepaths = [
            path
            for path in filepaths
            if self.tile_id in path and (self.level or "") in path
        ]

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
            return self.image_type(
                self.image_paths[item],
                file_system=self.file_system,
                paths=self.filepaths,
            )
        elif isinstance(item, slice):
            return [
                self.image_type(
                    path,
                    file_system=self.file_system,
                    paths=self.filepaths,
                )
                for path in self.image_paths[item]
            ]
        elif isinstance(item, Sequence) and not isinstance(item, str):
            return [
                self.image_type(
                    self.image_paths[i],
                    file_system=self.file_system,
                    paths=self.filepaths,
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
        return f"{self.__class__.__name__}(tile_id={self.tile_id}, image_paths={image_paths})"


class ImageCollection(ImageBase):
    image_regex: ClassVar[str]
    filename_regex: ClassVar[str]
    image_type: ClassVar[Image] = Image

    def __init__(
        self,
        path: str | Path,
        level: str | None = None,
        processes: int = 1,
        file_system: dp.gcs.GCSFileSystem | None = None,
    ) -> None:
        self.path = str(path)
        self.level = level
        self.file_system = file_system
        self.processes = processes

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

        self.filepaths = set(
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
            ImageTile(
                tile_id,
                level=self.level,
                image_paths=self.image_paths,
                filename_regex=self.filename_regex,
                image_regex=self.image_regex,
                file_system=self.file_system,
                image_type=self.image_type,
                processes=self.processes,
                filepaths=self.filepaths,
            )
            for tile_id in sorted(tile_ids)
        ]

    def get_images(
        self,
        bands: str | list[str] | None = None,
        date_range: tuple[str, str] | None = None,
        bbox: Any | None = None,
    ) -> list["Image"]:
        if bbox is not None:
            with joblib.Parallel(
                n_jobs=self.processes, backend="threading"
            ) as parallel:
                image_paths = parallel(
                    tile.image_paths
                    for tile in self.get_tiles()
                    if joblib.delayed(_intesects)(tile, bbox)
                )
        else:
            image_paths = self.image_paths

        return get_images(
            image_paths,
            level=self.level,
            bands=bands,
            paths=self.filepaths,
            date_range=date_range,
            bbox=bbox,
            image_regex=self.image_regex,
            processes=self.processes,
            image_type=self.image_type,
        )

    def get_tiles(self) -> list:
        return self._tiles

    def get_tile(self, tile_id: str) -> ImageTile:
        matches = [tile for tile in self.get_tiles() if tile_id == tile.tile_id]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            prefix = "Multiple"
        else:
            prefix = "No"

        raise KeyError(
            f"{prefix} matches for tile_id {tile_id} among tile_ids {[tile.tile_id for tile in self.get_tiles()]}"
        )

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

    def sample_yearly(self):
        pass

    def sample_change(
        self,
        size: int,
        cloud_band: str,
        cloud_values: tuple[int],
        max_cloud_cover: int = 50,
    ) -> "list[Image, Image]":
        tile: ImageTile = self.sample_tiles(1)[0]
        n_images = len(tile.image_paths)

        i, j = 1, -1
        while True:
            if i > n_images / 2:
                raise ValueError(
                    f"Not enough cloud free images for tile {tile.tile_id}"
                )
            first_img = tile[i]
            last_img = tile[j]
            mask = GeoSeries(first_img.unary_union).sample_points(1).buffer(size / 2)
            try:
                first_scl = first_img[cloud_band].clip(mask)
            except KeyError:
                return first_img, last_img

            cloud_percentage_first = np.mean(
                np.where(np.isin(first_scl, cloud_values), 1, 0)
            )
            if cloud_percentage_first > max_cloud_cover:
                i += 1
                j -= 1
                continue

            last_scl = last_img[cloud_band].clip(mask)
            cloud_percentage_last = np.mean(
                np.where(np.isin(last_scl, cloud_values), 1, 0)
            )
            if cloud_percentage_last > max_cloud_cover:
                i += 1
                j -= 1
                continue

            return first_img, last_img

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


def get_images(
    image_paths: list[str],
    *,
    paths: list[str],
    max_cloud_cover: int | None = None,
    image_regex: str | None = None,
    level: str | None = None,
    bands: str | list[str] | None = None,
    date_range: tuple[str, str] | None = None,
    bbox: Any | None = None,
    processes: int = 1,
    image_type: Image = Image,
) -> list[Image]:

    if image_regex:
        image_regex = re.compile(image_regex, flags=re.VERBOSE)

    with joblib.Parallel(n_jobs=processes, backend="threading") as parallel:
        images = parallel(
            joblib.delayed(image_type)(path, bands=bands, paths=paths)
            for path in image_paths
            if (level or "") in path
            and (re.search(image_regex, Path(path).name) if image_regex else 1)
        )
        # images = [
        #     image_type(path, bands=bands, filename_regex=filename_regex)
        #     for path in image_paths
        #     if (level or "") in path and re.search(image_regex, Path(path).name)
        # ]
    # else:
    #     images = [
    #         image_type(path, bands=bands, filename_regex=filename_regex)
    #         for path in image_paths
    #         if (level or "") in path
    #     ]

    if max_cloud_cover is not None:
        images = [image for image in images if image.cloud_cover < max_cloud_cover]

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


def _raster_load(raster, **kwargs):
    return raster.load(**kwargs)


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


class Sentinel2Image(Image):
    image_regex: ClassVar[str] = config.SENTINEL2_IMAGE_REGEX
    filename_regex: ClassVar[str] = config.SENTINEL2_FILENAME_REGEX
    cloud_cover_regexes: ClassVar[tuple[str]] = config.CLOUD_COVERAGE_REGEXES
    crs_regexes: ClassVar[tuple[str]] = config.CRS_REGEX
    bounds_regexes: ClassVar[tuple[dict[str, str]]] = config.BOUNDS_REGEX

    @property
    def bands(self) -> list[str]:
        return get_bands_for_image(self.path)

    @property
    def resolutions(self) -> dict[str, int]:
        return get_band_resolutions_for_image(self.path)

    @property
    def level(self) -> str:
        if "L1C" in self.path:
            return "L1C"
        elif "L2A" in self.path:
            return "L2A"
        raise ValueError(self.path)


class Sentinel2(ImageCollection):
    image_regex: ClassVar[str] = config.SENTINEL2_IMAGE_REGEX
    filename_regex: ClassVar[str] = config.SENTINEL2_FILENAME_REGEX
    image_type: ClassVar[Image] = Sentinel2Image

    def sample_change(
        self,
        size: int = 1000,
        cloud_band: str = "SCL",
        cloud_values: tuple[int] = (8, 7, 6),
        max_cloud_cover: int = 50,
    ) -> "list[Image, Image]":
        return super().sample_change(
            size=size,
            max_cloud_cover=max_cloud_cover,
            cloud_band=cloud_band,
            cloud_values=cloud_values,
        )


class NamedList:
    def __init__(self, *values):
        self.values = values

    def __getitem__(self, item):
        try:
            return self.values[item]
        except TypeError:
            return self.values[self.values.index(item)]
