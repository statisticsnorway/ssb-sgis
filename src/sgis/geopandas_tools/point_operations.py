"""Functions for point geometries."""

import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from shapely import Geometry
from shapely.ops import nearest_points, snap, unary_union

from .general import to_lines, to_multipoint


def snap_to(
    points: GeoDataFrame,
    to: GeoDataFrame,
    *,
    max_dist: int | None = None,
    to_node: bool = False,
    id_col: str | None = None,
    distance_col: str | None = "snap_distance",
    copy: bool = True,
) -> GeoDataFrame:
    """Snaps a set of points to the nearest geometry.

    It takes a GeoDataFrame of points and snaps them to the nearest geometry in a
    second GeoDataFrame.

    Notes:
        If there are identical geometries in 'to', and 'id_col' is specified,
        duplicate rows will be returned for each id that intersects with the snapped
        geometry. This does not happen if 'id_col' is None.

        If there are multiple unique geometries equally close to the points, one
        geometry will be chosen as the snap geometry. This will usually only happen
        with constructed data like grids or in the examples below.

    Args:
        points: The GeoDataFrame of points to snap.
        to: The GeoDataFrame to snap to.
        max_dist: The maximum distance to snap to. Defaults to None.
        to_node: If False (the default), the points will be snapped to the nearest
            point on the to geometry, which can be between two vertices if the
            to geometry is line or polygon. If True, the points will snap to the
            nearest node of the to geometry.
        id_col: Name of a column in the to data to use as an identifier for
            the geometry it was snapped to. Defaults to None.
        distance_col: Name of column with the snap distance. Defaults to
            'snap_distance'. Set to None to not get any distance column. This will make
            the function a bit faster.
        copy: If True, a copy of the GeoDataFrame is returned. Otherwise, the original
            GeoDataFrame. Defaults to True

    Returns:
        A GeoDataFrame or GeoSeries with the points snapped to the nearest point in the
        'to' GeoDataFrame or GeoSeries.

    Examples
    --------

    Create som points.

    >>> from sgis import snap_to, to_gdf
    >>> points = to_gdf([(0, 0), (1, 1)])
    >>> points
                    geometry
    0  POINT (0.00000 0.00000)
    1  POINT (1.00000 1.00000)
    >>> to = to_gdf([(2, 2), (3, 3)])
    >>> to["snap_to_idx"] = to.index
    >>> to
                    geometry  snap_to_idx
    0  POINT (2.00000 2.00000)            0
    1  POINT (3.00000 3.00000)            1

    Snap all 'points' to closest geometry in 'to'.

    >>> snap_to(points, to)
                    geometry  snap_distance
    0  POINT (2.00000 2.00000)       2.828427
    1  POINT (2.00000 2.00000)       1.414214

    Set 'id_col' to identify what geometry each point was snapped to.

    >>> snap_to(points, to, id_col="snap_to_idx")
                    geometry  snap_distance  snap_to_idx
    0  POINT (2.00000 2.00000)       2.828427            0
    1  POINT (2.00000 2.00000)       1.414214            0

    Snap only points closer than 'max_dist'.

    >>> snap_to(points, to, id_col="snap_to_idx", max_dist=1.5)
                    geometry  snap_distance  snap_to_idx
    0  POINT (0.00000 0.00000)            NaN          NaN
    1  POINT (2.00000 2.00000)       1.414214          0.0

    If there are identical distances, one point will be chosen as the snap point. The
    id values will be true to the snapped geometry.

    >>> point = to_gdf([0, 0])
    >>> to = to_gdf([(0, 1), (1, 0)])
    >>> to["snap_to_idx"] = to.index
    >>> snap_to(point, to, id_col="snap_to_idx")
                    geometry  snap_distance  snap_to_idx
    0  POINT (0.00000 1.00000)            1.0            0

    If there are identical geometries in 'to', duplicates will be returned if 'id_col'
    is specified.

    >>> point = to_gdf([0, 0])
    >>> to = to_gdf([(0, 1), (0, 1)])
    >>> to["snap_to_idx"] = to.index
    >>> snap_to(point, to, id_col="snap_to_idx")
                    geometry  snap_distance  snap_to_idx
    0  POINT (0.00000 1.00000)            1.0            0
    0  POINT (0.00000 1.00000)            1.0            1
    """
    geom1 = points._geometry_column_name

    if isinstance(to, GeoDataFrame):
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

    if distance_col or id_col:
        unsnapped = points.copy()

    if id_col:
        id_col_ = id_col
    else:
        to["temp__idx__"] = range(len(to))
        id_col_ = "temp__idx__"

    if copy:
        points = points.copy()

    points[geom1] = _series_snap_to(
        points=points[geom1],
        snap_to=to,
        max_dist=max_dist,
        to_node=to_node,
    )

    if not distance_col and not id_col:
        return points

    to = to[[id_col_, geom2]]

    # sjoin_nearest to get distance and/or id
    unsnapped = unsnapped.sjoin_nearest(to_lines(to), distance_col=distance_col)

    if max_dist:
        unsnapped = unsnapped.loc[unsnapped[distance_col] <= max_dist]

    # map distances from non-duplicate indices
    distances = unsnapped.loc[~unsnapped.index.duplicated(), distance_col]
    points[distance_col] = points.index.map(distances)

    if _rename:
        points = points.rename(columns={distance_col: distance_col + "_right"})

    if not id_col:
        return points

    # at this point, we only need the 'id_col' values. Since sjoin_nearest returns
    # duplicates for identical distances, and shapely.snap doesn't, we need to filter
    # out the ids from sjoin_nearest that were actually not snapped to. Doing a spatial
    # join (sjoin) between the snapped points with duplicate ids and the relevant 'to'
    # geometries.

    # if there are no duplicates, the ids can be mapped directly
    if len(unsnapped) == len(points) and unsnapped.index.is_unique:
        points[id_col] = points.index.map(unsnapped[id_col])
        return points

    # get all rows with duplicate indices from sjoin_nearest
    all_dups = unsnapped.index.duplicated(keep=False)
    duplicated = unsnapped.loc[all_dups]

    # get the relevant snapped points and 'to' geometries
    duplicated_snapped = points.loc[points.index.isin(duplicated.index)]
    maybe_snapped_to = to.loc[to[id_col_].isin(duplicated[id_col_])]

    # the snap points sometimes need to be buffered to intersect
    duplicated_snapped[geom1] = duplicated_snapped.buffer(
        duplicated_snapped[distance_col] / 10
    )

    # get the 'to' ids from the intersecting geometries
    snapped_to = duplicated_snapped.sjoin(maybe_snapped_to, how="inner")[id_col]

    # combine the duplicate ids with the non-duplicated
    not_duplicated = unsnapped.loc[~all_dups, id_col]
    ids = pd.concat([not_duplicated, snapped_to])

    points = points.join(ids)

    return points


def _series_snap_to(
    points: GeoSeries,
    snap_to: GeoSeries | GeoDataFrame | Geometry,
    max_dist: int | float | None = None,
    to_node: bool = False,
) -> GeoSeries:
    if to_node:
        snap_to = to_multipoint(snap_to)

    def snapfunc(point, snap_to):
        nearest = nearest_points(point, snap_to)[1]
        if not max_dist:
            return nearest
        return snap(point, nearest, tolerance=max_dist)

    if isinstance(snap_to, GeoDataFrame):
        unioned = snap_to.unary_union
    elif isinstance(snap_to, GeoSeries):
        unioned = snap_to.to_frame().unary_union
    else:
        unioned = unary_union(snap_to)

    return points.apply(lambda point: snapfunc(point, unioned))
