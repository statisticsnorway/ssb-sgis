"""Functions for point geometries."""

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from shapely import (
    Geometry,
    force_2d,
    wkt,
)
from shapely.ops import nearest_points, snap, unary_union


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

    >>> from gis_utils import snap_to, to_gdf
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
    geom2 = to._geometry_column_name

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
    unsnapped = unsnapped.sjoin_nearest(to, distance_col=distance_col)

    if max_dist:
        unsnapped = unsnapped.loc[unsnapped[distance_col] <= max_dist]

    # map distances from non-duplicate indices
    distances = unsnapped.loc[~unsnapped.index.duplicated(), distance_col]
    points[distance_col] = points.index.map(distances)

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

    if hasattr(snap_to, "unary_union"):
        unioned = snap_to.unary_union
    else:
        unioned = unary_union(snap_to)

    return points.apply(lambda point: snapfunc(point, unioned))


def to_multipoint(
    gdf: GeoDataFrame | GeoSeries | Geometry, copy: bool = False
) -> GeoDataFrame | GeoSeries | Geometry:
    """Creates a multipoint geometry of any geometry object.

    Takes a GeoDataFrame, GeoSeries or Shapely geometry and turns it into a MultiPoint.
    If the input is a GeoDataFrame or GeoSeries, the rows and columns will be preserved,
    but with a geometry column of MultiPoints.

    Args:
        gdf: The geometry to be converted to MultiPoint. Can be a GeoDataFrame,
            GeoSeries or a shapely geometry.
        copy: If True, the geometry will be copied. Defaults to False

    Returns:
        A GeoDataFrame with the geometry column as a MultiPoint, or Point if the
        original geometry was a point.

    Examples
    --------

    Let's create a GeoDataFrame with a point, a line and a polygon.

    >>> from gis_utils import to_multipoint, to_gdf
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

    >>> to_multipoint(gdf)
                                                geometry
    0                            POINT (0.00000 0.00000)
    1      MULTIPOINT (1.00000 1.00000, 2.00000 2.00000)
    2  MULTIPOINT (3.00000 3.00000, 3.00000 4.00000, ...
    """
    if copy:
        gdf = gdf.copy()

    if isinstance(gdf, (GeoDataFrame, GeoSeries)) and gdf.is_empty.any():
        raise ValueError("Cannot create multipoints from empty geometry.")
    if isinstance(gdf, Geometry) and gdf.is_empty:
        raise ValueError("Cannot create multipoints from empty geometry.")

    def _to_multipoint(gdf):
        koordinater = "".join(
            [x for x in gdf.wkt if x.isdigit() or x.isspace() or x == "." or x == ","]
        ).strip()

        alle_punkter = [
            wkt.loads(f"POINT ({punkt.strip()})") for punkt in koordinater.split(",")
        ]

        return unary_union(alle_punkter)

    if isinstance(gdf, GeoDataFrame):
        gdf[gdf._geometry_column_name] = (
            gdf[gdf._geometry_column_name]
            .pipe(force_2d)
            .apply(lambda x: _to_multipoint(x))
        )

    elif isinstance(gdf, gpd.GeoSeries):
        gdf = force_2d(gdf)
        gdf = gdf.apply(lambda x: _to_multipoint(x))

    else:
        gdf = force_2d(gdf)
        gdf = _to_multipoint(unary_union(gdf))

    return gdf
