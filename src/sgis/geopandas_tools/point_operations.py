"""Functions for point geometries."""

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from shapely.ops import nearest_points, snap, unary_union

from ..geopandas_tools.general import to_lines
from ..geopandas_tools.geometry_types import get_geom_type, to_single_geom_type


def snap_within_distance(
    points: GeoDataFrame | GeoSeries,
    to: GeoDataFrame | GeoSeries,
    max_distance: int | float,
    *,
    distance_col: str | None = None,
) -> GeoDataFrame | GeoSeries:
    """Snaps points to nearest geometry if within given distance.

    It takes a GeoDataFrame of points and snaps them to the nearest geometry in a
    second GeoDataFrame if the snap distance is less than 'max_distance'.
    Adds a distance column if specified.

    Args:
        points: The GeoDataFrame of points to snap.
        to: The GeoDataFrame to snap to.
        max_distance: The maximum distance to snap to.
        distance_col: Name of column with the snap distance. Defaults to
            'snap_distance'. Set to None to not get any distance column. This will make
            the function a bit faster.

    Returns:
        A GeoDataFrame or GeoSeries with the points snapped to the nearest point in the
        'to' GeoDataFrame or GeoSeries.

    Notes:
        If there are  geometries equally close to the points, one geometry will be
        chosen as the snap geometry. This will usually only happen with constructed
        data like grids or in the examples below.

        The snap point might be in between vertices of lines and polygons. Convert the
        'to' geometries to multipoint before snapping if the snap points should be
        vertices.

    Examples
    --------
    Create som points.

    >>> from sgis import snap_within_distance, to_gdf
    >>> points = to_gdf([(0, 0), (1, 1)])
    >>> points
                    geometry
    0  POINT (0.00000 0.00000)
    1  POINT (1.00000 1.00000)
    >>> to = to_gdf([(2, 2), (3, 3)])
    >>> to
                    geometry
    0  POINT (2.00000 2.00000)
    1  POINT (3.00000 3.00000)

    Snap 'points' to closest geometry in 'to' if distance is less 1, 2 and 3.

    >>> snap_within_distance(points, to, 1)
                    geometry snap_distance
    0  POINT (0.00000 0.00000)          <NA>
    1  POINT (1.00000 1.00000)          <NA>
    >>> snap_within_distance(points, to, 2)
                        geometry snap_distance
    0  POINT (0.00000 0.00000)          <NA>
    1  POINT (2.00000 2.00000)      1.414214
    >>> snap_within_distance(points, to, 3)
                    geometry snap_distance
    0  POINT (2.00000 2.00000)      2.828427
    1  POINT (2.00000 2.00000)      1.414214
    """

    if isinstance(points, GeoSeries):
        points = GeoDataFrame(points)
        _was_geoseries = True
    else:
        _was_geoseries = False

    if isinstance(to, GeoSeries):
        to = GeoDataFrame(to)

    to = _polygons_to_lines(to)

    copied = points.copy()

    geom_col = points._geometry_column_name
    copied[geom_col] = _series_snap(
        points=copied[geom_col],
        to=to,
        max_distance=max_distance,
    )

    if distance_col:
        copied[distance_col] = copied.distance(points)
        copied[distance_col] = np.where(
            copied[distance_col] == 0, pd.NA, copied[distance_col]
        )
    elif _was_geoseries:
        return copied[geom_col]

    return copied


def snap_all(
    points: GeoDataFrame | GeoSeries,
    to: GeoDataFrame | GeoSeries,
    *,
    distance_col: str | None = None,
) -> GeoDataFrame | GeoSeries:
    """Snaps points to the nearest geometry.

    It takes a GeoDataFrame of points and snaps them to the nearest geometry in a
    second GeoDataFrame. Adds a distance column if specified.

    Args:
        points: The GeoDataFrame of points to snap.
        to: The GeoDataFrame to snap to.
        distance_col: Name of column with the snap distance. Defaults to None.

    Returns:
        A GeoDataFrame or GeoSeries with the points snapped to the nearest point in the
        'to' GeoDataFrame or GeoSeries.

    Notes:
        If there are  geometries equally close to the points, one geometry will be
        chosen as the snap geometry. This will usually only happen with constructed
        data like grids or in the examples below.

        The snap point might be in between vertices of lines and polygons. Convert the
        'to' geometries to multipoint before snapping if the snap points should be
        vertices.

    Examples
    --------
    Create som points.

    >>> from sgis import snap_all, to_gdf
    >>> points = to_gdf([(0, 0), (1, 1)])
    >>> points
                    geometry
    0  POINT (0.00000 0.00000)
    1  POINT (1.00000 1.00000)
    >>> to = to_gdf([(2, 2), (3, 3)])
    >>> to["snap_idx"] = to.index
    >>> to
                    geometry  snap_idx
    0  POINT (2.00000 2.00000)            0
    1  POINT (3.00000 3.00000)            1

    Snap all points to closest geometry in 'to'.

    >>> snap_all(points, to)
                    geometry  snap_distance
    0  POINT (2.00000 2.00000)       2.828427
    1  POINT (2.00000 2.00000)       1.414214
    """

    if isinstance(points, GeoSeries):
        points = GeoDataFrame(points)
        _was_geoseries = True
    else:
        _was_geoseries = False

    if isinstance(to, GeoSeries):
        to = GeoDataFrame(to)

    to = _polygons_to_lines(to)

    copied = points.copy()

    geom_col = points._geometry_column_name
    copied[geom_col] = _series_snap(
        points=copied[geom_col],
        to=to,
        max_distance=None,
    )

    if distance_col:
        copied[distance_col] = copied.distance(points)
        copied[distance_col] = np.where(
            copied[distance_col] == 0, pd.NA, copied[distance_col]
        )
    elif _was_geoseries:
        return copied[geom_col]

    return copied


def _polygons_to_lines(gdf):
    if get_geom_type(gdf) == "polygon":
        return to_lines(gdf)
    if get_geom_type(gdf) == "mixed":
        gdf_points = to_single_geom_type(gdf, "point")
        gdf_lines = to_single_geom_type(gdf, "line")
        gdf_polys = to_lines(to_single_geom_type(gdf, "polygon"))
        return pd.concat([gdf_points, gdf_lines, gdf_polys])
    return gdf


def _series_snap(
    points: GeoSeries,
    to: GeoSeries | GeoDataFrame,
    max_distance: int | float | None = None,
) -> GeoSeries:
    def snapfunc(point, to):
        nearest = nearest_points(point, to)[1]
        if not max_distance:
            return nearest
        return snap(point, nearest, tolerance=max_distance)

    if isinstance(to, GeoDataFrame):
        unioned = to.unary_union
    elif isinstance(to, GeoSeries):
        unioned = to.to_frame().unary_union
    else:
        unioned = unary_union(to)

    return points.apply(lambda point: snapfunc(point, unioned))
