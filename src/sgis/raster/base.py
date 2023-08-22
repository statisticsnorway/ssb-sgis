import numbers
import re
from contextlib import contextmanager

import numpy as np
import pyproj
import rasterio
from affine import Affine

from ..geopandas_tools.bounds import to_bbox
from ..helpers import is_property


@contextmanager
def memfile_from_array(array, **profile):
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(array, indexes=profile["indexes"])
        with memfile.open() as dataset:
            yield dataset


class RasterHasChangedError(ValueError):
    def __init__(self, method: str):
        self.method = method

    def __str__(self):
        return (
            f"{self.method} requires reading of image files, but the "
            "current file paths are outdated. "
            "Use the write method to save new files. "
            "This also updates the file paths of the rasters."
        )


def get_index_mapper(df):
    idx_mapper = dict(enumerate(df.index))
    idx_name = df.index.name
    return idx_mapper, idx_name


class RasterBase:
    BASE_RASTER_PROPERTIES = ["_path", "_band_index", "_crs"]
    NEED_ONE_ATTR = ["transform", "bounds"]

    BASE_CUBE_COLS = [
        "raster_id",  # tile + date + name
        "name",  # to be implemented in Raster subclasses from e.g. regex
        "date",
        "tile",
        "band_index",  # the rasterio/gdal index (starts at 1)
        "subfolder",  # path below root without file name
        "path",
        "raster",
    ]

    MORE_RASTER_ATTR = [
        "shape",
        "res",
    ]

    CUBE_GEOM_COL = "box"

    PROFILE_ATTRS = [
        "driver",
        "dtype",
        "nodata",
        "crs",
        "height",
        "width",
        "blockysize",
        "blockxsize",
        "tiled",
        "compress",
        "interleave",
        "count",  # TODO: this should be based on band_index / array depth, so will have no effect
        "indexes",  # TODO
    ]

    ALL_ATTRS = list(
        set(
            BASE_CUBE_COLS + NEED_ONE_ATTR + MORE_RASTER_ATTR + PROFILE_ATTRS
        ).difference({"raster"})
    )

    ALLOWED_KEYS = ALL_ATTRS + ["array", "res"]

    @staticmethod
    def crs_to_string(crs):
        if crs is None:
            return "None"
        crs = pyproj.CRS(crs)
        crs_str = str(crs.to_json_dict()["name"])
        pattern = r"\d{4,5}"
        try:
            return re.search(pattern, crs_str).group()
        except AttributeError:
            return crs_str

    @property
    def properties(self) -> list[str]:
        out = []
        for attr in dir(self):
            try:
                if is_property(self, attr):
                    out.append(attr)
            except AttributeError:
                pass
        return out

    @classmethod
    def validate_dict(cls, dict_like):
        missing = []
        for attr in cls.BASE_RASTER_PROPERTIES:
            if any(
                [
                    attr in dict_like,
                    f"_{attr}" in dict_like,
                    attr.lstrip("_") in dict_like,
                ]
            ):
                continue
            missing.append(attr)
        if missing:
            raise AttributeError(f"Missing nessecary key(s) {', '.join(missing)}")

        if not any(attr in dict_like for attr in cls.NEED_ONE_ATTR):
            raise AttributeError("Must specify at least 'transform' or 'bounds'.")

    @classmethod
    def validate_key(cls, key):
        if key not in cls.ALLOWED_KEYS:
            raise ValueError(
                f"Got an unexpected key {key!r}. Allowed keys are ",
                ", ".join(cls.ALLOWED_KEYS),
            )

    @staticmethod
    def get_transform_from_bounds(obj, shape: tuple[float, ...]) -> Affine:
        minx, miny, maxx, maxy = to_bbox(obj)
        if len(shape) == 2:
            width, height = shape
        elif len(shape) == 3:
            _, width, height = shape
        else:
            raise ValueError
        return rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

    @staticmethod
    def get_shape_from_bounds(obj, res: int):
        resx, resy = (res, res) if isinstance(res, numbers.Number) else res

        minx, miny, maxx, maxy = to_bbox(obj)
        diffx = maxx - minx
        diffy = maxy - miny
        width = int(diffx / resx)
        heigth = int(diffy / resy)
        return width, heigth

    @staticmethod
    def _to_2d_array_list(array: np.ndarray) -> list[np.ndarray]:
        if len(array.shape) == 2:
            return [array]
        elif len(array.shape) == 3:
            return list(array)
        else:
            raise ValueError

    @classmethod
    def run_func_as_memfile(cls, func, arrays, profiles, **kwargs):
        if isinstance(arrays, np.ndarray):
            arrays = [arrays]
        if not all(isinstance(arr, np.ndarray) for arr in arrays):
            raise TypeError("arrays should be ndarrays.")
        datasets = []
        with rasterio.MemoryFile() as memfile:
            for array, profile in zip(arrays, profiles, strict=True):
                with memfile.open(**profile) as dataset:
                    for i, arr in enumerate(cls._to_2d_array_list(array)):
                        dataset.write(arr, i + 1)
                        datasets.append(dataset)

            with memfile.open() as dataset:
                return func(datasets, **kwargs)
