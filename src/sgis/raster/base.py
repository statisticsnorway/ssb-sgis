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
    BASE_RASTER_PROPERTIES = ["_path", "_band_indexes", "_crs", "_width", "_height"]
    NEED_ONE_ATTR = ["transform", "bounds"]

    PROFILE_ATTRS = [
        "driver",
        "dtype",  # kan endre med astype
        "nodata",
        "blockysize",
        "blockxsize",
        "tiled",  # Denne kan endres med merge?
        "compress",
        "interleave",
    ]

    BASE_CUBE_COLS = (
        ["name"] + [col.lstrip("_") for col in BASE_RASTER_PROPERTIES] + NEED_ONE_ATTR
    )

    ALL_ATTRS = BASE_CUBE_COLS + PROFILE_ATTRS

    ALLOWED_KEYS = ALL_ATTRS + []

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
            raise AttributeError("Must specify at least transform or bounds.")

    @classmethod
    def validate_key(cls, key):
        if key not in cls.ALLOWED_KEYS:
            raise ValueError(
                f"meta dict got an unexpected key {key!r}. Allowed keys are ",
                ", ".join(cls.ALLOWED_KEYS),
            )
