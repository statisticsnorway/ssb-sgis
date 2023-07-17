import functools
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from pandas import DataFrame

from .raster import Raster


# from .cube import GeoDataCube, concat_cubes, starmap_concat


class Sentinel2(Raster):
    band_colors = {
        "B1": "coastal aerosol",
        "B2": "blue",
        "B3": "green",
        "B4": "red",
        "B5": "vegetation red edge",
        "B6": "vegetation red edge",
        "B7": "vegetation red edge",
        "B8": "nir",
        "B8A": "narrow nir",
        "B9": "water vapour",
        "B10": "swir - cirrus",
        "B11": "swir",
        "B12": "swir",
    }

    nodata = 0
    dtype = np.uint8

    def __init__(self, dtype=np.uint8, **kwargs):
        self._band_index = 1
        super().__init__(dtype=dtype, **kwargs)

    @property
    def band_name(self):
        if self.is_mask or not self.path:
            return None
        try:
            name = re.search(r"B\d{1,2}A", Path(self.path).name).group()
        except AttributeError:
            name = re.search(r"B\d{1,2}", Path(self.path).name).group()
        if name in self.band_colors:
            return name
        else:
            raise ValueError(name)

    @property
    def band_color(self):
        if not self.band_name:
            return None
        return self.colors[self.band_name]

    @property
    def date(self):
        pattern = r"\d{8}"
        try:
            return re.search(pattern, self.subfolder).group()
        except AttributeError:
            return None

    @property
    def is_mask(self):
        return "masks" in str(self.path).lower()


"""
class SentinelCube(GeoDataCube):
    def __init__(
        self,
        data: Sentinel2 | Iterable[Sentinel2] | None = None,
        df: DataFrame | None = None,
        root: str | None = None,
        copy: bool = False,
    ) -> None:
        super().__init__(data, df=df, root=root, copy=copy)
        if not all(isinstance(r, Sentinel2) for r in self):
            raise ValueError("Rasters must be of type Sentinel2")

    @classmethod
    def from_root(
        cls,
        root: str | Path,
        with_masks: bool = False,
        **kwargs,
    ):
        raster_type = kwargs.pop("raster_type", Sentinel2)
        cube = super().from_root(
            root,
            raster_type=raster_type,
            **kwargs,
        )
        if not with_masks:
            cube.df = cube.df.loc[~cube.raster_attribute("is_mask")]
        return cube

    def keep_masks(self, copy=True):
        if copy:
            self = self.copy()
        self.df = self.df.loc[self.raster_attribute("is_mask")]
        return self

    def drop_masks(self, copy=True):
        if copy:
            self = self.copy()
        self.df = self.df.loc[~self.raster_attribute("is_mask")]
        return self

    @property
    def is_mask(self):
        return self.raster_attribute("is_mask")
"""
