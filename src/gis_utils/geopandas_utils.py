"""Functions that extend and simplify geopandas operations."""

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from numpy.random import random as np_random
from shapely import (
    Geometry,
    area,
    force_2d,
    get_exterior_ring,
    get_interior_ring,
    get_num_interior_rings,
    get_parts,
    polygons,
    wkb,
    wkt,
)
from shapely.geometry import Point
from shapely.ops import nearest_points, snap, unary_union


def close_holes(
    polygons: GeoDataFrame | GeoSeries | Geometry,
    max_km2: int | float | None = None,
    copy: bool = True,
) -> GeoDataFrame | GeoSeries | Geometry:
    """Closes holes in polygons, either all holes or holes smaller than 'max_km2'.

    It takes a GeoDataFrame, GeoSeries or shapely geometry of polygons object and
    returns the outer circle. Closes only holes smaller than 'max_km2' if speficied.
    km2 as in square kilometers.

    Args:
        polygons: GeoDataFrame, GeoSeries or shapely Geometry.
        max_km2: if None (default), all holes are closed.
            Otherwise, closes holes with an area below the specified number in
            square kilometers if the crs unit is in meters.
        copy: if True (default), the input GeoDataFrame or GeoSeries is copied.
            Defaults to True.

    Returns:
        A GeoDataFrame, GeoSeries or shapely Geometry with closed holes in the geometry
        column.

    Examples
    --------

    Let's create a circle with a hole in it.

    >>> from gis_utils import close_holes, buff
    >>> point = to_gdf([260000, 6650000], crs=25833)
    >>> point
                            geometry
    0  POINT (260000.000 6650000.000)
    >>> circle = buff(point, 1000)
    >>> small_circle = buff(point, 500)
    >>> circle_with_hole = circle.overlay(small_circle, how="difference")
    >>> circle_with_hole.area
    0    2.355807e+06
    dtype: float64

    Now we close the hole.

    >>> holes_closed = close_holes(circle_with_hole)
    >>> holes_closed.area
    0    3.141076e+06
    dtype: float64

    The hole will not be closed if it is larger in square kilometers than 'max_km2'.

    >>> holes_closed = close_holes(
    ...     circle_with_hole,
    ...     max_km2=0.1
    ... )
    >>> holes_closed.area
    0    2.355807e+06
    dtype: float64
    """
    if copy:
        polygons = polygons.copy()

    if isinstance(polygons, GeoDataFrame):
        polygons["geometry"] = polygons.geometry.map(
            lambda x: _close_holes_poly(x, max_km2)
        )

    elif isinstance(polygons, gpd.GeoSeries):
        polygons = polygons.map(lambda x: _close_holes_poly(x, max_km2))
        polygons = gpd.GeoSeries(polygons)

    else:
        polygons = _close_holes_poly(polygons, max_km2)

    return polygons


def clean_geoms(
    gdf: GeoDataFrame | GeoSeries,
    geom_type: str | None = None,
    ignore_index: bool = False,
) -> GeoDataFrame | GeoSeries:
    """Fixes geometries and removes invalid, empty, NaN and None geometries.

    Optionally keeps only the specified 'geom_type' ('point', 'line' or 'polygon').

    Args:
        gdf: GeoDataFrame or GeoSeries to be cleaned.
        geom_type: the geometry type to keep, either 'point', 'line' or 'polygon'. Both
            multi- and singlepart geometries are included. GeometryCollections will be
            exploded first, so that no geometries of the correct type are excluded.
        ignore_index: If True, the resulting axis will be labeled 0, 1, …, n - 1.
            Defaults to False

    Returns:
        GeoDataFrame or GeoSeries with fixed geometries and only the rows with valid,
        non-empty and not-NaN/-None geometries.

    Examples
    --------

    Removing None and empty geometries.

    >>> from gis_utils import clean_geoms, to_gdf, gdf_concat
    >>> import pandas as pd
    >>> from shapely import wkt
    >>> gdf = to_gdf([
    ...         "POINT (0 0)",
    ...         "LINESTRING (1 1, 2 2)",
    ...         "POLYGON ((3 3, 4 4, 3 4, 3 3))"
    ...         ])
    >>> gdf
                                                geometry
    0                            POINT (0.00000 0.00000)
    1      LINESTRING (1.00000 1.00000, 2.00000 2.00000)
    2  POLYGON ((3.00000 3.00000, 4.00000 4.00000, 3....



    >>> missing = pd.DataFrame({"geometry": [None]})
    >>> empty = to_gdf(wkt.loads("POINT (0 0)").buffer(0))
    >>> gdf = pd.concat([gdf, missing, empty])
    >>> gdf
                                                geometry
    0                            POINT (0.00000 0.00000)
    1      LINESTRING (1.00000 1.00000, 2.00000 2.00000)
    2  POLYGON ((3.00000 3.00000, 4.00000 4.00000, 3....
    0                                               None
    0                                      POLYGON EMPTY
    >>> clean_geoms(gdf)
                                                geometry
    0                            POINT (0.00000 0.00000)
    1      LINESTRING (1.00000 1.00000, 2.00000 2.00000)
    2  POLYGON ((3.00000 3.00000, 4.00000 4.00000, 3....

    Specify geom_type to keep only one geometry type.

    >>> clean_geoms(gdf, geom_type="polygon")
                                                geometry
    2  POLYGON ((3.00000 3.00000, 4.00000 4.00000, 3....
    """
    warnings.filterwarnings("ignore", "GeoSeries.notna", UserWarning)

    if isinstance(gdf, GeoDataFrame):
        geom_col = gdf._geometry_column_name
        gdf[geom_col] = gdf.make_valid()
        gdf = gdf.loc[
            (gdf[geom_col].is_valid)
            & (~gdf[geom_col].is_empty)
            & (gdf[geom_col].notna())
        ]
    elif isinstance(gdf, GeoSeries):
        gdf = gdf.make_valid()
        gdf = gdf.loc[(gdf.is_valid) & (~gdf.is_empty) & (gdf.notna())]
    else:
        raise TypeError(f"'gdf' should be GeoDataFrame or GeoSeries, got {type(gdf)}")

    if geom_type:
        gdf = to_single_geom_type(gdf, geom_type=geom_type, ignore_index=ignore_index)

    if ignore_index:
        gdf = gdf.reset_index(drop=True)

    return gdf


def series_snap_to(
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

    points[geom1] = series_snap_to(
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
    if len(unsnapped) == len(points):
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


def find_neighbours(
    gdf: GeoDataFrame | GeoSeries,
    possible_neighbours: GeoDataFrame | GeoSeries,
    id_col: str,
    max_dist: int = 0,
) -> list[str]:
    """Returns a list of neigbours for a GeoDataFrame.

    Finds all the geometries in 'possible_neighbours' that intersect with 'gdf'. If
    max_dist is specified, neighbours

    Args:
        gdf: GeoDataFrame or GeoSeries
        possible_neighbours: GeoDataFrame or GeoSeries
        id_col: The column in the GeoDataFrame that contains the unique identifier for
            each geometry.
        max_dist: The maximum distance between the two geometries. Defaults to 0

    Returns:
        A list of unique values from the id_col column in the joined dataframe.

    Examples
    --------
    >>> points = to_gdf([(0, 0), (0.5, 0.5), (2, 2)])
    >>> points["idx"] = points.index
    >>> points
                    geometry  idx
    0  POINT (0.00000 0.00000)    0
    1  POINT (0.50000 0.50000)    1
    2  POINT (2.00000 2.00000)    2
    >>> p1 = points.iloc[[0]]
    >>> find_neighbours(p1, points, id_col="idx")
    [0]
    >>> find_neighbours(p1, points, id_col="idx", max_dist=1)
    [0, 1]
    >>> find_neighbours(p1, points, id_col="idx", max_dist=3)
    [0, 1, 2]
    """
    if max_dist:
        if gdf.crs == 4326:
            warnings.warn(
                "'gdf' has latlon crs, meaning the 'max_dist' paramter "
                "will not be in meters, but degrees."
            )
        gdf = gdf.buffer(max_dist).to_frame()
    else:
        gdf = gdf.geometry.to_frame()

    if gdf.crs:
        possible_neighbours = possible_neighbours.to_crs(gdf.crs)

    joined = gdf.sjoin(possible_neighbours, how="inner")

    return [x for x in joined[id_col].unique()]


def to_single_geom_type(
    gdf: GeoDataFrame | GeoSeries,
    geom_type: str,
    ignore_index: bool = False,
) -> GeoDataFrame | GeoSeries:
    """Returns only the specified geometry type in a GeoDataFrame or GeoSeries.

    GeometryCollections are first exploded, then only the rows with the given
    geometry_type is kept. Both multipart and singlepart geometries are kept.
    LinearRings are considered lines. GeometryCollections are exploded to
    single-typed geometries before the selection.

    Args:
        gdf: GeoDataFrame or GeoSeries
        geom_type: the geometry type to keep, either 'point', 'line' or 'polygon'. Both
            multi- and singlepart geometries are included. GeometryCollections will be
            exploded first, so that no geometries of the correct type are excluded.
        ignore_index: If True, the resulting axis will be labeled 0, 1, …, n - 1.
            Defaults to False

    Returns:
        A GeoDataFrame with a single geometry type.

    Raises:
        TypeError: If incorrect gdf type. ValueError if incorrect geom_type.
        ValueError: If 'geom_type' is neither 'polygon', 'line' or 'point'.

    See also:
        clean_geoms: fixes geometries and returns single geometry type if
            'geom_type' is specified.

    Examples
    --------

    Create a GeoDataFrame of mixed geometries.

    >>> from gis_utils import to_gdf, to_single_geom_type
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
    if not isinstance(gdf, (GeoDataFrame, GeoSeries)):
        raise TypeError(f"'gdf' should be GeoDataFrame or GeoSeries, got {type(gdf)}")

    # explode collections to single-typed geometries
    collections = gdf.loc[gdf.geom_type == "GeometryCollection"]
    if len(collections):
        collections = collections.explode(ignore_index=ignore_index)
        gdf = gdf_concat([gdf, collections])

    if "poly" in geom_type:
        gdf = gdf.loc[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]
    elif "line" in geom_type:
        gdf = gdf.loc[
            gdf.geom_type.isin(["LineString", "MultiLineString", "LinearRing"])
        ]
    elif "point" in geom_type:
        gdf = gdf.loc[gdf.geom_type.isin(["Point", "MultiPoint"])]
    else:
        raise ValueError(
            f"Invalid geom_type {geom_type!r}. Should be 'polygon', 'line' or 'point'"
        )

    if ignore_index:
        gdf = gdf.reset_index(drop=True)

    return gdf


def is_single_geom_type(
    gdf: GeoDataFrame | GeoSeries,
) -> bool:
    """Returns True if all geometries in a GeoDataFrame are of the same type.

    The types are either polygon, line or point. Multipart and singlepart are
    considered the same type.

    Args:
        gdf: GeoDataFrame or GeoSeries

    Returns:
        True if all geometries are the same type, False if not.

    Raises:
        TypeError: If 'gdf' is not of type GeoDataFrame or GeoSeries.
    """
    if not isinstance(gdf, (GeoDataFrame, GeoSeries)):
        raise TypeError(f"'gdf' should be GeoDataFrame or GeoSeries, got {type(gdf)}")

    if all(gdf.geom_type.isin(["Polygon", "MultiPolygon"])):
        return True
    if all(gdf.geom_type.isin(["LineString", "MultiLineString", "LinearRing"])):
        return True
    if all(gdf.geom_type.isin(["Point", "MultiPoint"])):
        return True

    return False


def get_geom_type(
    gdf: GeoDataFrame | GeoSeries,
) -> str:
    """Returns a string of the geometry type in a GeoDataFrame or GeoSeries.

    Args:
        gdf: GeoDataFrame or GeoSeries

    Returns:
        A string that is either "polygon", "line", "point", or "mixed".

    Raises:
        TypeError: If 'gdf' is not of type GeoDataFrame or GeoSeries.
    """
    if not isinstance(gdf, (GeoDataFrame, GeoSeries)):
        raise TypeError(f"'gdf' should be GeoDataFrame or GeoSeries, got {type(gdf)}")

    if all(gdf.geom_type.isin(["Polygon", "MultiPolygon"])):
        return "polygon"
    if all(gdf.geom_type.isin(["LineString", "MultiLineString", "LinearRing"])):
        return "line"
    if all(gdf.geom_type.isin(["Point", "MultiPoint"])):
        return "point"
    return "mixed"


def to_gdf(
    geom: Geometry | str | bytes | list | tuple | dict | GeoSeries | pd.Series,
    crs: str | None = None,
    **kwargs,
) -> GeoDataFrame:
    """Converts geometry objects to a GeoDataFrame.

    Constructs a GeoDataFrame from any geometry object, or an interable of geometry
    objects. Accepted types are wkt, wkb, coordinate tuples, shapely objects,
    GeoSeries, Series/DataFrame.
    The geometry column is always named 'geometry'. The index will be preserved if the
    input type is (Geo)Series or dictionary.

    Args:
        geom: the object to be converted to a GeoDataFrame
        crs: if None (the default), it uses the crs of the GeoSeries if GeoSeries
            is the input type.
        **kwargs: additional keyword arguments taken by the GeoDataFrame constructor

    Returns:
        A GeoDataFrame with one column, the geometry column.

    Raises:
        TypeError: If geom is a GeoDataFrame

    Examples
    --------
    from gis_utils import to_gdf
    >>> wkt = "POINT (10 60)"
    >>> to_gdf(wkt, crs=4326)
                        geometry
    0  POINT (10.00000 60.00000)

    >>> coords = (10, 60)
    >>> to_gdf(coords, crs=4326)
                        geometry
    0  POINT (10.00000 60.00000)

    Lists of tuples will be converted to Points, unless a line or polygon geometry is
    constructed beforehand.

    >>> coordslist = [(10, 60), (11, 59)]
    >>> to_gdf(coordslist, crs=4326)
                        geometry
    0  POINT (10.00000 60.00000)
    1  POINT (11.00000 59.00000)

    >>> from shapely.geometry import LineString
    >>> to_gdf(LineString(coordslist), crs=4326)
                                                geometry
    0  LINESTRING (10.00000 60.00000, 11.00000 59.00000)

    >>> d = {1: (10, 60), 3: (11, 59)}
    >>> to_gdf(d)
                            geometry
    1  POINT (10.00000 60.00000)
    3  POINT (11.00000 59.00000)

    >>> from pandas import Series
    >>> to_gdf(Series(d))
                            geometry
    1  POINT (10.00000 60.00000)
    3  POINT (11.00000 59.00000)
    """
    if isinstance(geom, (GeoDataFrame)):
        raise TypeError("'to_gdf' doesn't accept GeoDataFrames as input type.")

    # convert the non-iterables
    if isinstance(geom, (str, bytes, Geometry)):
        geom = _make_shapely_geom(geom)
        return GeoDataFrame(
            {"geometry": GeoSeries(geom)}, geometry="geometry", crs=crs, **kwargs
        )

    # DataFrame with geometry column
    if isinstance(geom, pd.DataFrame) and hasattr(geom, "geometry"):
        return GeoDataFrame(geom, geometry="geometry", crs=crs, **kwargs)

    if isinstance(geom, GeoSeries):
        crs = geom.crs if not crs else crs
        return GeoDataFrame({"geometry": geom}, crs=crs, **kwargs)

    if isinstance(geom, (dict)):
        geom = pd.Series(geom)

    # preserve the index if there is any
    if isinstance(geom, pd.Series) and "index" not in kwargs:
        index = geom.index
    else:
        index = None

    if not hasattr(geom, "__iter__"):
        raise ValueError(f"Wrong input type. Got {type(geom)!r}.")

    if all(isinstance(i, (int, float)) for i in geom):
        geom = Point(geom)

    elif all(isinstance(i, GeoSeries) for i in geom):
        if not crs:
            crs = geom[0].crs
        return gdf_concat(
            [
                GeoDataFrame(
                    {"geometry": GeoSeries(x)},
                    geometry="geometry",
                    crs=crs,
                    **kwargs,
                )
                for x in geom
            ]
        )

    else:
        geom = map(_make_shapely_geom, geom)

    gdf = GeoDataFrame(
        {"geometry": GeoSeries(geom)}, geometry="geometry", crs=crs, **kwargs
    )

    if index is not None:
        gdf.index = index

    return gdf


def gdf_concat(
    gdfs: list[GeoDataFrame] | tuple[GeoDataFrame],
    crs: str | int | None = None,
    ignore_index: bool = True,
    geometry: str = "geometry",
    **kwargs,
) -> GeoDataFrame:
    """Sets common crs and concatinates GeoDataFrames rowwise while ignoring index.

    If no crs is given, chooses the first crs in the list of GeoDataFrames.

    Args:
        gdfs: list or tuple of GeoDataFrames to be concatinated.
        crs: common coordinate reference system each GeoDataFrames
            will be converted to before concatination. If None, it uses
            the crs of the first GeoDataFrame in the list or tuple.
        ignore_index: If True, the resulting axis will be labeled 0, 1, …, n - 1.
            Defaults to True
        geometry: name of geometry column. Defaults to 'geometry'
        **kwargs: additional keyword argument taken by pandas.condat

    Returns:
        A GeoDataFrame.

    Raises:
        ValueError: If all GeoDataFrames have 0 rows.
    """
    gdfs = [gdf for gdf in gdfs if len(gdf)]

    if not len(gdfs):
        raise ValueError("All GeoDataFrames have 0 rows")

    if not crs:
        crs = gdfs[0].crs

    try:
        gdfs = [gdf.to_crs(crs) for gdf in gdfs]
    except ValueError:
        warnings.warn(
            "Not all your GeoDataFrames have crs. If you are concatenating "
            "GeoDataFrames with different crs, the results will be wrong. First use "
            "set_crs to set the correct crs then the crs can be changed with to_crs()",
            stacklevel=2,
        )

    return GeoDataFrame(
        pd.concat(gdfs, ignore_index=ignore_index, **kwargs), geometry=geometry, crs=crs
    )


def push_geom_col(gdf: GeoDataFrame) -> GeoDataFrame:
    """Makes the geometry column the rightmost column in the GeoDataFrame.

    Args:
        gdf: GeoDataFrame

    Returns:
        The GeoDataFrame with the geometry column pushed all the way to the right
    """
    geom_col = gdf._geometry_column_name
    return gdf.reindex(columns=[c for c in gdf.columns if c != geom_col] + [geom_col])


def clean_clip(
    gdf: GeoDataFrame | GeoSeries,
    mask: GeoDataFrame | GeoSeries | Geometry,
    geom_type: str | None = None,
    **kwargs,
) -> GeoDataFrame | GeoSeries:
    """Clips geometries to the mask extent, then cleans the geometries.

    Geopandas.clip does a fast and durty clipping, with no guarantee for valid outputs.
    Here, geometries are made valid, then invalid, empty, nan and None geometries are
    removed. If the clip fails, it tries to clean the geometries before retrying the
    clip.

    Args:
        gdf: GeoDataFrame or GeoSeries to be clipped
        mask: the geometry to clip gdf
        geom_type (optional): geometry type to keep in 'gdf' before and after the clip
        **kwargs: Additional keyword arguments passed to GeoDataFrame.clip

    Returns:
        The cleanly clipped GeoDataFrame.

    Raises:
        TypeError: If gdf is not of type GeoDataFrame or GeoSeries.
    """
    if not isinstance(gdf, (GeoDataFrame, GeoSeries)):
        raise TypeError(f"'gdf' should be GeoDataFrame or GeoSeries, got {type(gdf)}")

    try:
        return gdf.clip(mask, **kwargs).pipe(clean_geoms, geom_type=geom_type)
    except Exception:
        gdf = clean_geoms(gdf, geom_type=geom_type)
        mask = clean_geoms(mask, geom_type="polygon")
        return gdf.clip(mask, **kwargs).pipe(clean_geoms, geom_type=geom_type)


def sjoin(
    left_df: GeoDataFrame, right_df: GeoDataFrame, drop_dupcol: bool = False, **kwargs
) -> GeoDataFrame:
    """Geopandas.sjoin that removes index columns before and after.

    geopandas.sjoin returns the column 'index_right', which throws an error the next
    time.

    Args:
        left_df: GeoDataFrame
        right_df: GeoDataFrame
        drop_dupcol: optionally remove all columns in right_df that is in left_df.
            Defaults to False, meaning no columns are dropped.
        **kwargs: keyword arguments taken by geopandas.sjoin.

    Returns:
        A GeoDataFrame with the geometries of left_df duplicated one time for each
        geometry from right_gdf that intersects. The GeoDataFrame also gets all
        columns from left_gdf and right_gdf, unless drop_dupcol is True.

    """
    INDEX_COLS = "index|level_"

    left_df = left_df.loc[:, ~left_df.columns.str.contains(INDEX_COLS)]
    right_df = right_df.loc[:, ~right_df.columns.str.contains(INDEX_COLS)]

    if drop_dupcol:
        right_df = right_df.loc[
            :,
            right_df.columns.difference(
                left_df.columns.difference([left_df._geometry_column_name])
            ),
        ]

    try:
        joined = left_df.sjoin(right_df, **kwargs)
    except Exception:
        left_df = clean_geoms(left_df)
        right_df = clean_geoms(right_df)
        joined = left_df.sjoin(right_df, **kwargs)

    return joined.loc[:, ~joined.columns.str.contains(INDEX_COLS)]


def random_points(n: int) -> GeoDataFrame:
    """Creates a GeoDataFrame with n random points.

    Args:
        n: number of points/rows to create.

    Returns:
        A GeoDataFrame of points with n rows.
    """
    if isinstance(n, (str, float)):
        n = int(n)

    x = np_random(n)
    y = np_random(n)

    return GeoDataFrame(
        (Point(geom) for geom in zip(x, y, strict=True)), columns=["geometry"]
    )


def find_neighbors(
    gdf: GeoDataFrame | GeoSeries,
    possible_neighbors: GeoDataFrame | GeoSeries,
    id_col: str,
    max_dist: int = 0,
) -> list[str]:
    """American alias for find_neighbours."""
    return find_neighbours(gdf, possible_neighbors, id_col, max_dist)


def _close_holes_poly(poly, max_km2=None):
    """Closes holes within one shapely geometry of polygons."""
    # dissolve the exterior ring(s)
    if max_km2 is None:
        holes_closed = polygons(get_exterior_ring(get_parts(poly)))
        return unary_union(holes_closed)

    # start with a list containing the polygon,
    # then append all holes smaller than 'max_km2' to the list.
    holes_closed = [poly]
    singlepart = get_parts(poly)
    for part in singlepart:
        n_interior_rings = get_num_interior_rings(part)

        if not (n_interior_rings):
            continue

        for n in range(n_interior_rings):
            hole = polygons(get_interior_ring(part, n))

            if area(hole) / 1_000_000 < max_km2:
                holes_closed.append(hole)

    return unary_union(holes_closed)


def _make_shapely_geom(geom):
    if isinstance(geom, (tuple, list, np.ndarray)):
        if len(geom) == 2 or len(geom) == 3:
            return Point(geom)
        else:
            raise ValueError(
                "If 'geom' is list/tuple/ndarray, each item should consist of "
                "2 or 3 coordinates (x, y, z)."
            )
    if isinstance(geom, str):
        return wkt.loads(geom)

    if isinstance(geom, bytes):
        return wkb.loads(geom)

    return geom
