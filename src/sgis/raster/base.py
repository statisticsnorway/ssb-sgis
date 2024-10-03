import json
import numbers
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import joblib
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from rasterio import features
from rasterio.enums import MergeAlg
from shapely import Geometry
from shapely.geometry import shape

from ..geopandas_tools.conversion import to_bbox


def _get_transform_from_bounds(
    obj: GeoDataFrame | GeoSeries | Geometry | tuple, shape: tuple[float, ...]
) -> Affine:
    minx, miny, maxx, maxy = to_bbox(obj)
    if len(shape) == 2:
        height, width = shape
    elif len(shape) == 3:
        _, height, width = shape
    else:
        return None
        # raise ValueError(shape)
    return rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)


def _get_shape_from_bounds(
    obj: GeoDataFrame | GeoSeries | Geometry | tuple,
    res: int,
    indexes: int | tuple[int],
) -> tuple[int, int]:
    resx, resy = (res, res) if isinstance(res, numbers.Number) else res

    minx, miny, maxx, maxy = to_bbox(obj)

    # minx = math.floor(minx / res) * res
    # maxx = math.ceil(maxx / res) * res
    # miny = math.floor(miny / res) * res
    # maxy = math.ceil(maxy / res) * res

    # # Compute output array shape. We guarantee it will cover the output
    # # bounds completely
    # width = round((maxx - minx) // res)
    # height = round((maxy - miny) // res)

    # if not isinstance(indexes, int):
    #     return len(indexes), height, width
    # return height, width

    diffx = maxx - minx
    diffy = maxy - miny
    width = int(diffx / resx)
    height = int(diffy / resy)
    if not isinstance(indexes, int):
        return len(indexes), width, height
    return height, width


def _array_to_geojson(
    array: np.ndarray, transform: Affine, processes: int
) -> list[tuple]:
    if hasattr(array, "mask"):
        if isinstance(array.mask, np.ndarray):
            mask = array.mask == False
        else:
            mask = None
        array = array.data
    else:
        mask = None

    try:
        return _array_to_geojson_loop(array, transform, mask, processes)
    except ValueError:
        try:
            array = array.astype(np.float32)
            return _array_to_geojson_loop(array, transform, mask, processes)

        except Exception as err:
            raise err.__class__(array.shape, err) from err


def _array_to_geojson_loop(array, transform, mask, processes):
    if processes == 1:
        return [
            (value, shape(geom))
            for geom, value in features.shapes(array, transform=transform, mask=mask)
        ]
    else:
        with joblib.Parallel(n_jobs=processes, backend="threading") as parallel:
            return parallel(
                joblib.delayed(_value_geom_pair)(value, geom)
                for geom, value in features.shapes(
                    array, transform=transform, mask=mask
                )
            )


def _value_geom_pair(value, geom):
    return (value, shape(geom))


def _gdf_to_arr(
    gdf: GeoDataFrame,
    res: int | float,
    fill: int = 0,
    all_touched: bool = False,
    merge_alg: Callable = MergeAlg.replace,
    default_value: int = 1,
    dtype: Any | None = None,
) -> np.ndarray:
    """Construct Raster from a GeoDataFrame or GeoSeries.

    The GeoDataFrame should have

    Args:
        gdf: The GeoDataFrame to rasterize.
        res: Resolution of the raster in units of the GeoDataFrame's coordinate reference system.
        fill: Fill value for areas outside of input geometries (default is 0).
        all_touched: Whether to consider all pixels touched by geometries,
            not just those whose center is within the polygon (default is False).
        merge_alg: Merge algorithm to use when combining geometries
            (default is 'MergeAlg.replace').
        default_value: Default value to use for the rasterized pixels
            (default is 1).
        dtype: Data type of the output array. If None, it will be
            determined automatically.

    Returns:
        A Raster instance based on the specified GeoDataFrame and parameters.

    Raises:
        TypeError: If 'transform' is provided in kwargs, as this is
        computed based on the GeoDataFrame bounds and resolution.
    """
    if isinstance(gdf, GeoSeries):
        values = gdf.index
        gdf = gdf.to_frame("geometry")
    elif isinstance(gdf, GeoDataFrame):
        if len(gdf.columns) > 2:
            raise ValueError(
                "gdf should have only a geometry column and one numeric column to "
                "use as array values. "
                "Alternatively only a geometry column and a numeric index."
            )
        elif len(gdf.columns) == 1:
            values = gdf.index
        else:
            col: str = next(
                iter([col for col in gdf if col != gdf._geometry_column_name])
            )
            values = gdf[col]

    if isinstance(values, pd.MultiIndex):
        raise ValueError("Index cannot be MultiIndex.")

    shape = _get_shape_from_bounds(gdf.total_bounds, res=res, indexes=1)
    transform = _get_transform_from_bounds(gdf.total_bounds, shape)

    return features.rasterize(
        _gdf_to_geojson_with_col(gdf, values),
        out_shape=shape,
        transform=transform,
        fill=fill,
        all_touched=all_touched,
        merge_alg=merge_alg,
        default_value=default_value,
        dtype=dtype,
    )


def _gdf_to_geojson_with_col(gdf: GeoDataFrame, values: np.ndarray) -> list[dict]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return [
            (feature["geometry"], val)
            for val, feature in zip(
                values, json.loads(gdf.to_json())["features"], strict=False
            )
        ]


def _shapely_to_raster(
    geometry: Geometry,
    res: int | float,
    fill: int = 0,
    all_touched: bool = False,
    merge_alg: Callable = MergeAlg.replace,
    default_value: int = 1,
    dtype: Any | None = None,
) -> np.array:
    shape = _get_shape_from_bounds(geometry.bounds, res=res, indexes=1)
    transform = _get_transform_from_bounds(geometry.bounds, shape)

    return features.rasterize(
        [(geometry, default_value)],
        out_shape=shape,
        transform=transform,
        fill=fill,
        all_touched=all_touched,
        merge_alg=merge_alg,
        default_value=default_value,
        dtype=dtype,
    )


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
