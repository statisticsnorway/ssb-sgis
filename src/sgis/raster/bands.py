import numpy as np

from .raster import Raster


SENTINEL2_FILENAME_REGEX = r"""
    ^SENTINEL2X_
    (?P<date>\d{8})
    .*T(?P<tile>\d{2}[A-Z]{3})
    .*(?:_(?P<resolution>{}m))?
    .*(?P<band>B\d{1,2}A|B\d{1,2})
    .*\..*$
"""


class Sentinel2(Raster):
    filename_regex = SENTINEL2_FILENAME_REGEX
    date_format: str = "%Y%m%d"

    _profile = {
        "driver": "GTiff",
        "compress": "LZW",
        "dtype": np.uint16,
        "nodata": 0,
        "indexes": 1,
    }

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

    @property
    def band_color(self):
        if not self.band:
            return None
        return self.band_colors[self.band]
