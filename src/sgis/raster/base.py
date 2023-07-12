import numpy as np
import pandas as pd
import pyproj

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
    # BASE_RASTER_PROPERTIES = ["_path", "_band_indexes", "_crs", "_width", "_height"]
    BASE_RASTER_PROPERTIES = ["_path", "_band_indexes", "_crs"]
    NEED_ONE_ATTR = ["transform", "bounds"]

    BASE_CUBE_COLS = ["name", "path", "band_indexes", "crs"]

    PROFILE_ATTRS = [
        "driver",
        "dtype",  # kan endre med astype
        "nodata",
        "height",
        "width",
        "blockysize",
        "blockxsize",
        "tiled",  # Denne kan endres med merge?
        "compress",
        "interleave",
    ]

    ALL_ATTRS = list(set(BASE_CUBE_COLS + NEED_ONE_ATTR + PROFILE_ATTRS))

    ALLOWED_KEYS = ALL_ATTRS + ["array", "res"]

    @staticmethod
    def _crs_to_string(crs):
        if crs is None:
            return "None"
        crs = pyproj.CRS(crs)
        return str(crs.to_json_dict()["name"])

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
    def verify_dict(cls, dict_like):
        for attr in cls.BASE_RASTER_PROPERTIES:
            if any(
                [
                    attr in dict_like,
                    f"_{attr}" in dict_like,
                    attr.lstrip("_") in dict_like,
                ]
            ):
                continue
            raise AttributeError(f"Missing nessecary key {attr.lstrip('_')!r}")

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
