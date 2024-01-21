import numpy as np

from .raster import Raster


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

    name_regex = r"B\d{1,2}A|B\d{1,2}"
    date_regex = r"(\d{8})"  # r"20\d{8}"
    date_format: str = "%Y%m%d"
    nodata = 0
    _dtype = np.uint16

    def __init__(self, raster=None, indexes=1, **kwargs):
        super().__init__(raster, indexes=indexes, **kwargs)

    @property
    def band_color(self):
        if not self.name:
            return None
        return self.band_colors[self.name]

    @property
    def is_mask(self):
        return "masks" in str(self.path).lower()
