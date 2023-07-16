import numbers
import re

import numpy as np
import pandas as pd
import pyproj
import rasterio
from affine import Affine

from ..geopandas_tools.bounds import to_bbox
from ..helpers import is_property


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


class RasterBase:
    BASE_RASTER_PROPERTIES = ["_path", "_band_index", "_crs"]
    NEED_ONE_ATTR = ["transform", "bounds"]

    BASE_CUBE_COLS = [
        "name",  # file stem
        "path",  # entire path
        "subfolder",  # path below root without name
        "band_index",  # the rasterio/gdal index
        "band_name",
        "raster",
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
    ]

    ALL_ATTRS = list(
        set(BASE_CUBE_COLS + NEED_ONE_ATTR + PROFILE_ATTRS).difference({"raster"})
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
    def properties(self):
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

    def check_not_array_mess(self):
        return (
            "Cannot load the arrays more than once. "
            "Use the write method to write the arrays as image files. "
            "This also updates the 'path' column of the cube's df."
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
        if isinstance(res, numbers.Number):
            resx, resy = res, res
        elif not hasattr(res, "__iter__"):
            raise TypeError
        elif len(res) == 2:
            resx, resy = res
        else:
            raise TypeError

        minx, miny, maxx, maxy = to_bbox(obj)
        diffx = maxx - minx
        diffy = maxy - miny
        width = int(diffx / resx)
        heigth = int(diffy / resy)
        return width, heigth
