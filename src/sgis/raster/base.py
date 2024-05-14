from contextlib import contextmanager

import numpy as np
import pandas as pd
import rasterio


@contextmanager
def memfile_from_array(array: np.ndarray, **profile) -> rasterio.MemoryFile:
    """Yield a memory file from a numpy array."""
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(array, indexes=profile["indexes"])
        with memfile.open() as dataset:
            yield dataset


def get_index_mapper(df: pd.DataFrame) -> tuple[dict[int, int], str]:
    """Get a dict of index mapping and the name of the index."""
    idx_mapper = dict(enumerate(df.index))
    idx_name = df.index.name
    return idx_mapper, idx_name


NESSECARY_META = [
    "path",
    "type",
    "bounds",
    "crs",
]

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

ALLOWED_KEYS = (
    NESSECARY_META
    + PROFILE_ATTRS
    + ["array", "res", "transform", "name", "date", "regex"]
)
