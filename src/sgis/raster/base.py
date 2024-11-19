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
import shapely
from affine import Affine
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from rasterio import features
from rasterio.enums import MergeAlg
from shapely import Geometry
from shapely.geometry import shape

from ..geopandas_tools.conversion import to_bbox


def _get_res_from_bounds(
    obj: GeoDataFrame | GeoSeries | Geometry | tuple, shape: tuple[int, ...]
) -> tuple[int, int] | None:
    minx, miny, maxx, maxy = to_bbox(obj)
    try:
        height, width = shape[-2:]
    except IndexError:
        return None
    resx = (maxx - minx) / width
    resy = (maxy - miny) / height
    return resx, resy


def _get_transform_from_bounds(
    obj: GeoDataFrame | GeoSeries | Geometry | tuple, shape: tuple[int, ...]
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


def _res_as_tuple(res: int | float | tuple[int | float]) -> tuple[int | float]:
    return (res, res) if isinstance(res, numbers.Number) else res


def _get_shape_from_bounds(
    obj: GeoDataFrame | GeoSeries | Geometry | tuple,
    res: int,
    indexes: int | tuple[int],
) -> tuple[int, int]:
    resx, resy = _res_as_tuple(res)

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
    res: int | float | None = None,
    out_shape: tuple[int, int] | None = None,
    bounds: tuple[float] | None = None,
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
        bounds: Optional bounds to box 'gdf' into (so both clip and extend to).
        fill: Fill value for areas outside of input geometries (default is 0).
        all_touched: Whether to consider all pixels touched by geometries,
            not just those whose center is within the polygon (default is False).
        merge_alg: Merge algorithm to use when combining geometries
            (default is 'MergeAlg.replace').
        default_value: Default value to use for the rasterized pixels
            (default is 1).
        dtype: Data type of the output array. If None, it will be
            determined automatically.
        out_shape: Optional 2 dimensional shape of the resulting array.

    Returns:
        A Raster instance based on the specified GeoDataFrame and parameters.

    Raises:
        TypeError: If 'transform' is provided in kwargs, as this is
        computed based on the GeoDataFrame bounds and resolution.
    """
    if res is not None and out_shape is not None:
        raise TypeError("Cannot specify both 'res' and 'out_shape'")
    if res is None and out_shape is None:
        raise TypeError("Must specify either 'res' or 'out_shape'")

    if isinstance(gdf, GeoSeries):
        gdf = gdf.to_frame("geometry")
    elif not isinstance(gdf, GeoDataFrame):
        raise TypeError(type(gdf))

    if bounds is not None:
        gdf = gdf.clip(bounds)
        bounds_gdf = GeoDataFrame({"geometry": [shapely.box(*bounds)]}, crs=gdf.crs)

    if len(gdf.columns) > 2:
        raise ValueError(
            "gdf should have only a geometry column and one numeric column to "
            "use as array values. "
            "Alternatively only a geometry column and a numeric index."
        )
    elif len(gdf.columns) == 1:
        values = np.full(len(gdf), default_value)
    else:
        col: str = next(iter(gdf.columns.difference({gdf.geometry.name})))
        values = gdf[col].values

    if bounds is not None:
        gdf = pd.concat([bounds_gdf, gdf])
        values = np.concatenate([np.array([fill]), values])

    if out_shape is None:
        assert res is not None
        out_shape = _get_shape_from_bounds(gdf.total_bounds, res=res, indexes=1)

    if not len(gdf):
        return np.full(out_shape, fill)

    transform = _get_transform_from_bounds(gdf.total_bounds, out_shape)

    return features.rasterize(
        _gdf_to_geojson_with_col(gdf, values),
        out_shape=out_shape,
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
