"""Functions that extend and simplify geopandas operations."""

import warnings
from types import GeneratorType
import numbers
from collections.abc import Iterator, Sized

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from pandas.api.types import is_dict_like
from numpy.random import random as _np_random
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
    geom: Geometry
    | str
    | bytes
    | list
    | tuple
    | dict
    | GeoSeries
    | pd.Series
    | pd.DataFrame
    | Iterator
    | zip,
    crs: str | tuple[str] | None = None,
    geometry: str = "geometry",
    copy: bool = True,
    **kwargs,
) -> GeoDataFrame:
    """Converts geometry objects to a GeoDataFrame.

    Constructs a GeoDataFrame from any geometry object, or an interable of geometry
    objects. Accepted types are wkt, wkb, coordinate tuples, shapely objects,
    GeoSeries, Series/DataFrame. The index/keys will be preserved if the input type is
    (Geo)Series or dictionary.

    Note:
        The name of the geometry-like column/key in DataFrame/dict can be specified with
        the "geometry" parameter. The geometry column in the resulting GeoDataFrame will,
        however, always be named 'geometry'.

    Args:
        geom: the object to be converted to a GeoDataFrame
        crs: if None (the default), it uses the crs of the GeoSeries if GeoSeries
            is the input type.
        geometry: name of resulting geometry column. Defaults to 'geometry'. If you
            have a DataFrame with geometry-like column, specify this column.
        copy: Applies to DataFrames.
        **kwargs: additional keyword arguments taken by the GeoDataFrame constructor

    Returns:
        A GeoDataFrame with one column, the geometry column.

    Raises:
        TypeError: If geom is a GeoDataFrame

    Examples
    --------
    >>> from gis_utils import to_gdf
    >>> coords = (10, 60)
    >>> to_gdf(coords, crs=4326)
                        geometry
    0  POINT (10.00000 60.00000)

    >>> wkt = "POINT (10 60)"
    >>> to_gdf(wkt, crs=4326)
                        geometry
    0  POINT (10.00000 60.00000)

    Multiple coordinates will be converted to Points, unless a line or polygon geometry
    is constructed beforehand.

    >>> coordslist = [(10, 60), (11, 59)]
    >>> to_gdf(coordslist, crs=4326)
                        geometry
    0  POINT (10.00000 60.00000)
    1  POINT (11.00000 59.00000)

    >>> from shapely.geometry import LineString
    >>> to_gdf(LineString(coordslist), crs=4326)
                                                geometry
    0  LINESTRING (10.00000 60.00000, 11.00000 59.00000)

    Dictionaries/Series will preserve the keys/index.

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

    DataFrame/dict with geometry-like column, will keep its column structure if
    the geometry column matches the "geometry" parameter, which is set to "geometry" by
    default.

    >>> df = pd.DataFrame({"col": [1, 2], "geometry": ["point (10 60)", (11, 59)]})
    >>> df
       col       geometry
    0    1  point (10 60)
    1    2       (11, 59)
    >>> gdf = to_gdf(df, geometry="geometry)
    >>> gdf
       col                   geometry
    0    1  POINT (10.00000 60.00000)
    1    2  POINT (11.00000 59.00000)

    From DataFrame with x, y (optionally z) coordinate columns.

    >>> df = pd.DataFrame({"x": [10, 11], "y": [60, 59]})
        x   y
    0  10  60
    1  11  59
    >>> gdf = to_gdf(df, geometry=["x", "y"])
    >>> gdf
        x   y                   geometry
    0  10  60  POINT (10.00000 60.00000)
    1  11  59  POINT (11.00000 59.00000)

    From 2/3 dimensional array.

    >>> to_gdf(np.random.randint(100, size=(5, 3)))
                            geometry
    0  POINT Z (82.000 88.000 82.000)
    1  POINT Z (70.000 92.000 20.000)
    2   POINT Z (91.000 34.000 3.000)
    3   POINT Z (1.000 50.000 77.000)
    4  POINT Z (58.000 49.000 46.000)
    """
    if isinstance(geom, GeoDataFrame):
        raise TypeError("'to_gdf' doesn't accept GeoDataFrames as input type.")

    if isinstance(geom, GeoSeries):
        if not crs:
            crs = geom.crs
        else:
            geom = geom.to_crs(crs) if geom.crs else geom.set_crs(crs)

        return GeoDataFrame({"geometry": geom}, geometry="geometry", crs=crs, **kwargs)

    # first the iterators that get consumed by 'all' statements
    if isinstance(geom, Iterator) and not isinstance(geom, Sized):
        # geom = GeoSeries(map(_make_shapely_geom, geom))
        geom = GeoSeries(_make_shapely_geom(x) for x in geom)
        return GeoDataFrame({"geometry": geom}, geometry="geometry", crs=crs, **kwargs)

    # dataframes and dicts with geometry/xyz key(s)
    if _is_df_like(geom, geometry):
        geom = geom.copy() if copy else geom
        geom = _make_geomcol_df_like(geom, geometry, index=kwargs.get("index"))
        if "geometry" in geom:
            return GeoDataFrame(geom, geometry="geometry", crs=crs, **kwargs)
        elif isinstance(geom, pd.DataFrame):
            raise ValueError(f"Cannot find 'geometry' column {geometry!r}")

    if is_dict_like(geom):
        if isinstance(geom, dict):
            geom = pd.Series(geom)
        if "index" not in kwargs:
            index_ = geom.index
        else:
            index_ = None
        geom = GeoSeries(_make_shapely_geom(x) for x in geom)
        #  geom = GeoSeries(map(_make_shapely_geom, geom))
        gdf = GeoDataFrame({"geometry": geom}, geometry="geometry", crs=crs, **kwargs)
        if index_ is not None:
            gdf.index = index_
        return gdf

    # single geometry objects like wkt, wkb, shapely geometry or iterable of numbers
    if _is_one_geometry(geom):
        geom = GeoSeries(_make_shapely_geom(geom))
        return GeoDataFrame({"geometry": geom}, geometry="geometry", crs=crs, **kwargs)

        # single geometry objects like wkt, wkb, shapely geometry or iterable of numbers
    if hasattr(geom, "__iter__"):
        geom = GeoSeries(_make_shapely_geom(x) for x in geom)
        # geom = GeoSeries(map(_make_shapely_geom, geom))
        return GeoDataFrame({"geometry": geom}, geometry="geometry", crs=crs, **kwargs)

    raise TypeError(f"Got unexpected type {type(geom)}")


def _is_one_geometry(geom) -> bool:
    if (
        isinstance(geom, (str, bytes, Geometry))
        or all(isinstance(i, numbers.Number) for i in geom)
        or not hasattr(geom, "__iter__")
    ):
        return True
    return False


def _is_df_like(geom, geometry) -> bool:

    if not is_dict_like(geom):
        return False

    if len(geometry) == 1 and geometry[0] in geom:
        return True

    if len(geometry) == 2:
        x, y = geometry
        if x in geom and y in geom:
            return True

    elif len(geometry) == 3:
        x, y, z = geometry
        if x in geom and y in geom and z in geom:
            return True

    if geometry in geom:
        return True

    if len(geom.keys()) == 1:
        return True


def _make_geomcol_df_like(
    geom: pd.DataFrame | dict,
    geometry: str | tuple[str],
    index: list | tuple | None,
) -> pd.DataFrame | dict:
    """Create GeoSeries column in DataFrame or dictionary that has a DataFrame structure
    rather than a Series structure.

    Use the same logic for DataFrame and dict, as long as the speficied 'geometry'
    value or values (if x, y, z coordinate columns) are keys in the dictionary. If
    not, nothing happens here, and the dict goes on to be treated as a Series where
    keys are index and values the geometries.
    """

    if not index and hasattr(geom, "index"):
        index = geom.index
    elif index and hasattr(geom, "index"):
        geom.index = index

    if len(geometry) == 1 and geometry[0] in geom:
        geometry = geometry[0]

    if isinstance(geometry, (tuple, list)):
        if len(geometry) == 2:
            x, y = geometry
            z = None
        elif len(geometry) == 3:
            x, y, z = geometry
            z = geom[z]
        else:
            raise ValueError(
                "geometry should be one geometry-like column or 2-3 x, y, z columns."
            )
        geom["geometry"] = gpd.GeoSeries.from_xy(x=geom[x], y=geom[y], z=z)
    elif geometry in geom:
        geom["geometry"] = GeoSeries(
            map(_make_shapely_geom, geom[geometry]), index=index
        )

    elif isinstance(geom, pd.DataFrame) and len(geom.columns) == 1:
        geometry = geom.columns[0]
        geom["geometry"] = GeoSeries(
            map(_make_shapely_geom, geom[geometry]), index=index
        )

    elif isinstance(geom, dict) and len(geom.keys()) == 1:
        geometry = next(iter(geom))
        geom["geometry"] = GeoSeries(
            map(_make_shapely_geom, geom[geometry]), index=index
        )

    return geom


def _make_shapely_geom(geom):
    """Create shapely geometry from wkt, wkb or coordinate tuple.

    Works recursively if the object is a nested iterable.
    """
    if isinstance(geom, str):
        return wkt.loads(geom)

    if isinstance(geom, bytes):
        return wkb.loads(geom)

    if isinstance(geom, Geometry):
        return geom

    if not hasattr(geom, "__iter__"):
        raise ValueError(
            f"Couldn't create shapely geometry from {geom} of type {type(geom)}"
        )

    if isinstance(geom, GeoSeries):
        raise TypeError(
            "to_gdf doesn't accept iterable of GeoSeries. Instead use: "
            "gdf_concat(to_gdf(geom) for geom in geoseries_list)"
        )

    if not any(isinstance(g, numbers.Number) for g in geom):
        # we're likely dealing with a nested iterable, so let's
        # recursively dig out way down to the coords/wkt/wkb
        return unary_union([_make_shapely_geom(g) for g in geom])
    elif len(geom) == 2 or len(geom) == 3:
        return Point(geom)
    else:
        raise ValueError(
            "If 'geom' is an iterable, each item should consist of "
            "wkt, wkb or 2/3 coordinates (x, y, z)."
        )


def gdf_concat(
    gdfs: list[GeoDataFrame] | tuple[GeoDataFrame],
    crs: str | int | None = None,
    ignore_index: bool = True,
    geometry: str = "geometry",
    **kwargs,
) -> GeoDataFrame:
    """Converts to common crs and concatinates GeoDataFrames rowwise while ignoring index.

    If no crs is given, chooses the first crs in the list of GeoDataFrames.

    Args:
        gdfs: list, tuple or other iterable of GeoDataFrames to be concatinated.
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

    Examples
    --------
    >>> from gis_utils import gdf_concat, to_gdf
    >>> points = to_gdf([(0, 0), (0.5, 0.5), (2, 2)])
    >>> points
                    geometry
    0  POINT (0.00000 0.00000)
    1  POINT (0.50000 0.50000)
    2  POINT (2.00000 2.00000)
    >>> gdf_concat([points, points])
    C:/Users/ort/git/ssb-gis-utils/src/gis_utils/geopandas_utils.py:828: UserWarning: None of your GeoDataFrames have crs.
    warnings.warn("None of your GeoDataFrames have crs.")
                    geometry
    0  POINT (0.00000 0.00000)
    1  POINT (0.50000 0.50000)
    2  POINT (2.00000 2.00000)
    3  POINT (0.00000 0.00000)
    4  POINT (0.50000 0.50000)
    5  POINT (2.00000 2.00000)

    We get a warning that we don't have any crs. Let's create the same point with
    different crs. gdf_concat will then convert to the first gdf, if crs is
    unspecified.

    >>> unprojected_point = to_gdf([(10, 60)], crs=4326)
    >>> utm_point = unprojected_point.to_crs(25833)
    >>> gdf_concat([utm_point, unprojected_point])
                            geometry
    0  POINT (221288.770 6661953.040)
    1  POINT (221288.770 6661953.040)
    """
    if not hasattr(gdfs, "__iter__"):
        raise TypeError("'gdfs' must be an iterable.")

    gdfs = [gdf for gdf in gdfs if len(gdf)]

    if not len(gdfs):
        raise ValueError("All GeoDataFrames have 0 rows")

    if not crs:
        crs = gdfs[0].crs

    try:
        gdfs = [gdf.to_crs(crs) for gdf in gdfs]
    except ValueError:
        if all(gdf.crs is None for gdf in gdfs):
            warnings.warn("None of your GeoDataFrames have crs.")
        else:
            warnings.warn(
                "Not all your GeoDataFrames have crs. If you are concatenating "
                "GeoDataFrames with different crs, the results will be wrong. First use "
                "set_crs to set the correct crs then the crs can be changed with to_crs()",
                stacklevel=2,
            )

    return GeoDataFrame(
        pd.concat(gdfs, ignore_index=ignore_index, **kwargs), geometry=geometry, crs=crs
    )


def coordinate_array(
    gdf: GeoDataFrame,
) -> np.ndarray[np.ndarray[float], np.ndarray[float]]:
    """Creates a 2d ndarray of coordinates from a GeoDataFrame of points.

    Args:
        gdf: GeoDataFrame of point geometries.

    Returns:
        np.ndarray of np.ndarrays of coordinates.

    Examples
    --------
    >>> from gis_utils import coordinate_array, random_points
    >>> points = random_points(5)
    >>> points
                    geometry
    0  POINT (0.59376 0.92577)
    1  POINT (0.34075 0.91650)
    2  POINT (0.74841 0.10627)
    3  POINT (0.00966 0.87868)
    4  POINT (0.38046 0.87879)
    >>> coordinate_array(points)
    array([[0.59376221, 0.92577159],
        [0.34074678, 0.91650446],
        [0.74840912, 0.10626954],
        [0.00965935, 0.87867915],
        [0.38045827, 0.87878816]])

    """
    return np.array([(geom.x, geom.y) for geom in gdf.geometry])


def push_geom_col(gdf: GeoDataFrame) -> GeoDataFrame:
    """Makes the geometry column the rightmost column in the GeoDataFrame.

    Args:
        gdf: GeoDataFrame.

    Returns:
        The GeoDataFrame with the geometry column pushed all the way to the right.
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

    Examples
    --------
    >>> from gis_utils import random_points
    >>> points = random_points(10_000)
    >>> points
                        geometry
    0     POINT (0.62044 0.22805)
    1     POINT (0.31885 0.38109)
    2     POINT (0.39632 0.61130)
    3     POINT (0.99401 0.35732)
    4     POINT (0.76403 0.73539)
    ...                       ...
    9995  POINT (0.90433 0.75080)
    9996  POINT (0.10959 0.59785)
    9997  POINT (0.00330 0.79168)
    9998  POINT (0.90926 0.96215)
    9999  POINT (0.01386 0.22935)

    [10000 rows x 1 columns]
    """
    if isinstance(n, (str, float)):
        n = int(n)

    x = _np_random(n)
    y = _np_random(n)

    return GeoDataFrame(
        (Point(x, y) for x, y in zip(x, y, strict=True)), columns=["geometry"]
    )
    return GeoDataFrame(map(Point, zip(x, y, strict=True)), columns=["geometry"])


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
