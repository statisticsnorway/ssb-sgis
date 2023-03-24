"""Functions for point geometries."""

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from shapely import Geometry
from shapely.ops import nearest_points, snap, unary_union

from .general import to_lines
from .geometry_types import get_geom_type


def snap_within_distance(
    points: GeoDataFrame,
    to: GeoDataFrame,
    max_dist: int | float,
    *,
    distance_col: str | None = "snap_distance",
) -> GeoDataFrame:
    """Snaps points to nearest geometry if within given distance.

    It takes a GeoDataFrame of points and snaps them to the nearest geometry in a
    second GeoDataFrame if the snap distance is less than 'max_dist'. Also returns
    distance column.

    Args:
        points: The GeoDataFrame of points to snap.
        to: The GeoDataFrame to snap to.
        max_dist: The maximum distance to snap to.
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
    geom1 = points._geometry_column_name

    copied = points.copy()

    copied[geom1] = _series_snap(
        points=copied[geom1],
        to=to,
        max_dist=max_dist,
    )

    if distance_col:
        copied[distance_col] = copied.distance(points)
        copied[distance_col] = np.where(
            copied[distance_col] == 0, pd.NA, copied[distance_col]
        )

    return copied


def snap_all(
    points: GeoDataFrame,
    to: GeoDataFrame,
    *,
    distance_col: str | None = "snap_distance",
) -> GeoDataFrame:
    """Snaps points to the nearest geometry.

    It takes a GeoDataFrame of points and snaps them to the nearest geometry in a
    second GeoDataFrame. Also returns distance column.

    Args:
        points: The GeoDataFrame of points to snap.
        to: The GeoDataFrame to snap to.
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
    geom1 = points._geometry_column_name

    copied = points.copy()

    copied[geom1] = _series_snap(
        points=copied[geom1],
        to=to,
    )

    if distance_col:
        copied[distance_col] = copied.distance(points)
        copied[distance_col] = np.where(
            copied[distance_col] == 0, pd.NA, copied[distance_col]
        )

    return copied


def snap_and_get_ids(
    points: GeoDataFrame,
    to: GeoDataFrame,
    *,
    max_dist: int | None = None,
    id_col: str,
    distance_col: str | None = "snap_distance",
) -> GeoDataFrame:
    """Snaps a set of points to the nearest geometry and gets id of the snap geometry.

    It takes a GeoDataFrame of points and snaps them to the nearest geometry in a
    second GeoDataFrame. Also returns distance column and id values of the 'to'
    geometries.

    Args:
        points: The GeoDataFrame of points to snap.
        to: The GeoDataFrame to snap to.
        max_dist: The maximum distance to snap to. Defaults to None, meaning all points
            will be snapped.
        id_col: Name of a column in the to data to use as an identifier for
            the geometry it was snapped to.
        distance_col: Name of column with the snap distance. Defaults to
            'snap_distance'. Set to None to not get any distance column. This will make
            the function a bit faster.

    Returns:
        A GeoDataFrame or GeoSeries with the points snapped to the nearest point in the
        'to' GeoDataFrame or GeoSeries.

    Notes:
        If there are identical geometries in 'to', and 'id_col' is specified,
        duplicate rows will be returned for each id that intersects with the snapped
        geometry. This does not happen if 'id_col' is None.

        If there are  geometries equally close to the points, one geometry will be
        chosen as the snap geometry. This will usually only happen with constructed
        data like grids or in the examples below.

        The snap point might be in between vertices of lines and polygons. Convert the
        'to' geometries to multipoint before snapping if the snap points should be
        vertices.

    Examples
    --------

    Create som points.

    >>> from sgis import snap_and_get_ids, to_gdf
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

    >>> snap_and_get_ids(points, to, id_col="snap_idx")
                    geometry  snap_distance  snap_idx
    0  POINT (2.00000 2.00000)       2.828427            0
    1  POINT (2.00000 2.00000)       1.414214            0

    Snap only points closer than 'max_dist'.

    >>> snap_and_get_ids(points, to, id_col="snap_idx", max_dist=1.5)
                    geometry  snap_distance  snap_idx
    0  POINT (0.00000 0.00000)            NaN          NaN
    1  POINT (2.00000 2.00000)       1.414214          0.0

    If there are identical distances, one point will be chosen as the snap point. The
    id values will be true to the snapped geometry.

    >>> point = to_gdf([0, 0])
    >>> to = to_gdf([(0, 1), (1, 0)])
    >>> to["snap_idx"] = to.index
    >>> snap_and_get_ids(point, to, id_col="snap_idx")
                    geometry  snap_distance  snap_idx
    0  POINT (0.00000 1.00000)            1.0            0

    If there are identical geometries in 'to', duplicates will be returned if 'id_col'
    is specified.

    >>> point = to_gdf([0, 0])
    >>> to = to_gdf([(0, 1), (0, 1)])
    >>> to["snap_idx"] = to.index
    >>> snap_and_get_ids(point, to, id_col="snap_idx")
                    geometry  snap_distance  snap_idx
    0  POINT (0.00000 1.00000)            1.0            0
    0  POINT (0.00000 1.00000)            1.0            1
    """
    geom1 = points._geometry_column_name

    geom2 = to._geometry_column_name

    if distance_col and distance_col in points:
        points = points.rename(columns={distance_col: distance_col + "_left"})
        _rename = True
    elif (
        distance_col
        and distance_col + "_left" in points
        and distance_col + "_right" in points
    ):
        raise ValueError(
            f"Too many {distance_col!r} columns in the axis. "
            f"Choose a different distance_col string value or set distance_col=None."
        )
    else:
        _rename = False

    copied = points.copy()

    copied[geom1] = _series_snap(
        points=copied[geom1],
        to=to,
        max_dist=max_dist,
    )

    if not distance_col:
        return copied

    to = to[[id_col, geom2]]

    # polygons to lines to get correct snap distance
    to = to.explode(ignore_index=True)
    to.loc[to.geom_type.isin(["Polygon", "MultiPolygon"]), "geometry"] = to_lines(
        to
    ).geometry

    points = points.sjoin_nearest(to, distance_col=distance_col)

    if max_dist is not None:
        points = points.loc[points[distance_col] <= max_dist]

    # map distances from non-duplicate indices
    distances = points.loc[~points.index.duplicated(), distance_col]
    copied[distance_col] = copied.index.map(distances)

    if _rename:
        copied = copied.rename(columns={distance_col: distance_col + "_right"})

    # at this point, we only need the 'id_col' values. Since sjoin_nearest returns
    # duplicates for identical distances, and shapely.snap doesn't, we need to filter
    # out the ids from sjoin_nearest that were actually not snapped to. Doing a spatial
    # join (sjoin) between the snapped points with duplicate ids and the relevant 'to'
    # geometries.

    # if there are no duplicates, the ids can be mapped directly
    if len(points) == len(copied) and points.index.is_unique:
        copied[id_col] = copied.index.map(points[id_col])
        return copied

    # get all rows with duplicate indices from sjoin_nearest
    all_dups = points.index.duplicated(keep=False)
    duplicated = points.loc[all_dups]

    # get the relevant snapped points and 'to' geometries
    duplicated_snapped = copied.loc[copied.index.isin(duplicated.index)]
    maybe_snapped_to = to.loc[to[id_col].isin(duplicated[id_col])]

    # the snap points sometimes need to be buffered to intersect
    duplicated_snapped[geom1] = duplicated_snapped.buffer(
        duplicated_snapped[distance_col] / 10
    )

    # get the 'to' ids from the intersecting geometries
    snapped_to = duplicated_snapped.sjoin(maybe_snapped_to, how="inner")[id_col]

    # combine the duplicate ids with the non-duplicated
    not_duplicated = points.loc[~all_dups, id_col]
    ids = pd.concat([not_duplicated, snapped_to])

    copied = copied.join(ids)

    return copied


def _series_snap(
    points: GeoSeries,
    to: GeoSeries | GeoDataFrame | Geometry,
    max_dist: int | float | None = None,
) -> GeoSeries:
    def snapfunc(point, to):
        nearest = nearest_points(point, to)[1]
        if not max_dist:
            return nearest
        return snap(point, nearest, tolerance=max_dist)

    if isinstance(to, GeoDataFrame):
        unioned = to.unary_union
    elif isinstance(to, GeoSeries):
        unioned = to.to_frame().unary_union
    else:
        unioned = unary_union(to)

    return points.apply(lambda point: snapfunc(point, unioned))
