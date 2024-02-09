from contextlib import contextmanager

import rasterio


@contextmanager
def memfile_from_array(array, **profile):
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(array, indexes=profile["indexes"])
        with memfile.open() as dataset:
            yield dataset


def get_index_mapper(df):
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
    NESSECARY_META + PROFILE_ATTRS + ["array", "res", "transform", "name", "date"]
)
