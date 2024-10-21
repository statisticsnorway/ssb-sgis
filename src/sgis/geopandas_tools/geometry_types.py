"""Check and set geometry type."""

import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from geopandas.array import GeometryArray
from shapely import Geometry


def make_all_singlepart(
    gdf: GeoDataFrame | GeoSeries, index_parts: bool = False, ignore_index: bool = False
) -> GeoDataFrame | GeoSeries:
    """Make all geometries single part.

    This means doing either 0, 1 or 2 calls to the explode method.

    Args:
        gdf: GeoDataFrame
        index_parts: Defaults to False.
        ignore_index: Defaults to False.

    Returns:
        A GeoDataFrame of singlepart geometries.
    """
    # only explode if nessecary
    if index_parts or ignore_index:
        gdf = gdf.explode(index_parts=index_parts, ignore_index=ignore_index)

    while not np.all(
        np.isin(_get_geom_types(gdf), ["Polygon", "Point", "LineString", "LinearRing"])
    ):
        gdf = gdf.explode(index_parts=index_parts, ignore_index=ignore_index)

    return gdf


def to_single_geom_type(
    gdf: GeoDataFrame | GeoSeries | Geometry | GeometryArray | np.ndarray,
    geom_type: str,
    ignore_index: bool = False,
) -> GeoDataFrame | GeoSeries:
    """Returns only the specified geometry type in a GeoDataFrame or GeoSeries.

    GeometryCollections are first exploded, then only the rows with the given
    geometry_type is kept. Both multipart and singlepart geometries are kept.
    LinearRings are considered lines.

    Args:
        gdf: GeoDataFrame or GeoSeries
        geom_type: the geometry type to keep, either 'point', 'line' or 'polygon'. Both
            multi- and singlepart geometries are included.
        ignore_index: If True, the resulting axis will be labeled 0, 1, …, n - 1.
            Defaults to False.

    Returns:
        A GeoDataFrame with a single geometry type.

    Raises:
        TypeError: If incorrect gdf type.
        ValueError: If 'geom_type' is neither 'polygon', 'line' or 'point'.

    Examples:
    ---------
    First create a GeoDataFrame of mixed geometries.

    >>> from sgis import to_gdf, to_single_geom_type
    >>> from shapely.geometry import LineString, Polygon
    >>> gdf = to_gdf([
    ...     (0, 0),
    ...     LineString([(1, 1), (2, 2)]),
    ...     Polygon([(3, 3), (4, 4), (3, 4), (3, 3)])
    ...     ])
    >>> gdf
                                                geometry
    0                            POINT (0.00000 0.00000)
    1      LINESTRING (1.00000 1.00000, 2.00000 2.00000)
    2  POLYGON ((3.00000 3.00000, 4.00000 4.00000, 3....

    >>> to_single_geom_type(gdf, "line")
                                            geometry
    1  LINESTRING (1.00000 1.00000, 2.00000 2.00000)
    >>> to_single_geom_type(gdf, "polygon")
                                                geometry
    2  POLYGON ((3.00000 3.00000, 4.00000 4.00000, 3....

    Also keeps multigeometries and geometries within GeometryCollections.

    >>> gdf = gdf.dissolve()
    >>> gdf
                                                geometry
    0  GEOMETRYCOLLECTION (POINT (0.00000 0.00000), L...

    >>> to_single_geom_type(gdf, "line")
                                            geometry
    2  LINESTRING (1.00000 1.00000, 2.00000 2.00000)
    """
    if all(g not in geom_type for g in ["polygon", "line", "point"]):
        raise ValueError(
            f"Invalid geom_type {geom_type!r}. Should be 'polygon', 'line' or 'point'"
        )

    if isinstance(gdf, Geometry):
        return _shapely_to_single_geom_type(gdf, geom_type)

    if isinstance(gdf, (np.ndarray, GeometryArray)):
        arr = np.vectorize(_shapely_to_single_geom_type)(gdf, geom_type)
        return arr[~shapely.is_empty(arr)]

    if not isinstance(gdf, (GeoDataFrame, GeoSeries)):  # type: ignore [unreachable]
        raise TypeError(f"'gdf' should be GeoDataFrame or GeoSeries, got {type(gdf)}")

    # explode collections to single-typed geometries
    collections = _get_geom_types(gdf) == "GeometryCollection"
    if collections.any():
        collections = make_all_singlepart(gdf[collections], ignore_index=ignore_index)

        gdf = pd.concat([gdf, collections], ignore_index=ignore_index)

    if "poly" in geom_type:
        is_polygon = np.isin(_get_geom_types(gdf), ["Polygon", "MultiPolygon"])
        if not is_polygon.all():
            gdf = gdf.loc[is_polygon]
    elif "line" in geom_type:
        is_line = np.isin(
            _get_geom_types(gdf), ["LineString", "MultiLineString", "LinearRing"]
        )
        if not is_line.all():
            gdf = gdf.loc[is_line]
    else:
        is_point = np.isin(_get_geom_types(gdf), ["Point", "MultiPoint"])
        if not is_point.all():
            gdf = gdf.loc[is_point]

    return gdf.reset_index(drop=True) if ignore_index else gdf


def _shapely_to_single_geom_type(geom: Geometry, geom_type: str) -> Geometry:
    parts = shapely.get_parts(geom)
    return shapely.union_all(
        [part for part in parts if geom_type.lower() in part.geom_type.lower()]
    )


def get_geom_type(gdf: GeoDataFrame | GeoSeries) -> str:
    """Returns a string of the geometry type in a GeoDataFrame or GeoSeries.

    Args:
        gdf: GeoDataFrame or GeoSeries

    Returns:
        A string that is either "polygon", "line", "point", or "mixed".

    Raises:
        TypeError: If 'gdf' is not of type GeoDataFrame or GeoSeries.

    Examples:
    ---------
    >>> from sgis import to_gdf, get_geom_type
    >>> gdf = to_gdf([0, 0])
    >>> gdf
                      geometry
    0  POINT (0.00000 0.00000)
    >>> get_geom_type(gdf)
    'point'
    """
    polys = ["Polygon", "MultiPolygon", None]
    lines = ["LineString", "MultiLineString", "LinearRing", None]
    points = ["Point", "MultiPoint", None]
    if isinstance(gdf, Geometry):
        if gdf.geom_type in polys:
            return "polygon"
        if gdf.geom_type in lines:
            return "line"
        if gdf.geom_type in points:
            return "point"
        return "mixed"

    if np.all(np.isin(_get_geom_types(gdf), polys)):
        return "polygon"
    if np.all(np.isin(_get_geom_types(gdf), lines)):
        return "line"
    if np.all(np.isin(_get_geom_types(gdf), points)):
        return "point"
    return "mixed"


def is_single_geom_type(gdf: GeoDataFrame | GeoSeries) -> bool:
    """Returns True if all geometries in a GeoDataFrame are of the same type.

    The types are either polygon, line or point. Multipart and singlepart are
    considered the same type.

    Args:
        gdf: GeoDataFrame or GeoSeries

    Returns:
        True if all geometries are the same type, False if not.

    Raises:
        TypeError: If 'gdf' is not of type GeoDataFrame or GeoSeries.

    Examples:
    ---------
    >>> from sgis import to_gdf, get_geom_type
    >>> gdf = to_gdf([0, 0])
    >>> gdf
                                                geometry
    0                            POINT (0.00000 0.00000)
    >>> is_single_geom_type(gdf)
    True
    """
    return (
        np.all(np.isin(_get_geom_types(gdf), ["Polygon", "MultiPolygon"]))
        or np.all(
            np.isin(
                _get_geom_types(gdf), ["LineString", "MultiLineString", "LinearRing"]
            )
        )
        or np.all(np.isin(_get_geom_types(gdf), ["Point", "MultiPoint"]))
    )


def _get_geom_types(
    gdf: GeoDataFrame | GeoSeries | GeometryArray | np.ndarray,
) -> np.ndarray:
    try:
        return GeometryArray(gdf.geometry.values).geom_type
    except AttributeError:
        return getattr(GeometryArray(gdf, "geom_type"))
