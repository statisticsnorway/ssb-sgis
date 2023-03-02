import warnings

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from numpy.random import random as np_random
from shapely import (
    Geometry,
    area,
    get_exterior_ring,
    get_interior_ring,
    get_num_interior_rings,
    get_parts,
    polygons,
)
from shapely.geometry import Point
from shapely.ops import nearest_points, snap, unary_union


def close_holes(
    polygons: GeoDataFrame | GeoSeries | Geometry,
    max_km2: int | None = None,
    copy: bool = True,
) -> GeoDataFrame | GeoSeries | Geometry:
    """Closes holes in polygons, either all holes or holes smaller than 'max_km2'

    Args:
      polygons: GeoDataFrame, GeoSeries or shapely Geometry.
      max_km2: if None (default), all holes are closed.
        Otherwise, closes holes with an area below the specified number in
        square kilometers if the crs unit is in meters.
      copy: if True (default), the input GeoDataFrame or GeoSeries is copied.
        Defaults to True

    Returns:
      A GeoDataFrame, GeoSeries or shapely Geometry with closed holes in the geometry column
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
    """

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

    return gdf


def snap_to(
    points: GeoDataFrame | GeoSeries,
    snap_to: GeoDataFrame | GeoSeries,
    *,
    max_dist: int | None = None,
    to_node: bool = False,
    snap_to_id: str | None = None,
    copy: bool = True,
) -> GeoDataFrame | GeoSeries:
    """Snaps a set of points to the nearest geometry

    It takes a GeoDataFrame or GeoSeries of points and snaps them to the nearest
    geometry in a second GeoDataFrame or GeoSeries.

    Args:
        points: The GeoDataFrame or GeoSeries of points to snap
        snap_to: The GeoDataFrame or GeoSeries to snap to
        max_dist: The maximum distance to snap to. Defaults to None.
        to_node: If False (the default), the points will snap to the nearest point on
            the snap_to geometry, which can be between two vertices if the snap_to
            geometry is line or polygon. If True, the points will snap to the nearest
            node of the snap_to geometry.
        snap_to_id: name of a column in the snap_to data to use as an identifier for
            the geometry it was snapped to. Defaults to None.
        copy: If True, a copy of the GeoDataFrame is returned. Otherwise, the original
            GeoDataFrame. Defaults to True

    Returns:
        A GeoDataFrame or GeoSeries with the points snapped to the nearest point in the
        'snap_to' GeoDataFrame or GeoSeries.
    """

    if copy:
        points = points.copy()

    unioned = snap_to.unary_union

    if to_node:
        unioned = to_multipoint(unioned)

    def func(point, snap_to):
        if not max_dist:
            return nearest_points(point, snap_to)[1]

        nearest = nearest_points(point, snap_to)[1]
        return snap(point, nearest, tolerance=max_dist)

    if isinstance(points, GeoDataFrame):
        points[points._geometry_column_name] = points[
            points._geometry_column_name
        ].apply(lambda point: func(point, unioned))

    if isinstance(points, gpd.GeoSeries):
        points = points.apply(lambda point: func(point, unioned))

    if snap_to_id:
        points = points.sjoin_nearest(snap_to[[snap_to_id, "geometry"]]).drop(
            "index_right", axis=1, errors="ignore"
        )

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
    """
    from shapely import force_2d
    from shapely.wkt import loads

    if copy:
        gdf = gdf.copy()

    def _to_multipoint(gdf):
        koordinater = "".join(
            [x for x in gdf.wkt if x.isdigit() or x.isspace() or x == "." or x == ","]
        ).strip()

        alle_punkter = [
            loads(f"POINT ({punkt.strip()})") for punkt in koordinater.split(",")
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
    """Returns a list of neigbours for a GeoDataFrame

    Finds all the geometries in 'possible_neighbours' that intersects with the first
    geometry.

    Args:
        gdf: GeoDataFrame or GeoSeries
        possible_neighbours: GeoDataFrame or GeoSeries
        id_col: The column in the GeoDataFrame that contains the unique identifier for
            each geometry.
        max_dist: The maximum distance between the two geometries. Defaults to 0

    Returns:
      A list of unique values from the id_col column in the joined dataframe.
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
      A GeoDataFrame with a single geometry type

    Raises:
        TypeError: If incorrect gdf type. ValueError if incorrect geom_type.
        ValueError: If 'geom_type' is neither 'polygon', 'line' or 'point'.
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
            f"Invalid geom_type '{geom_type}'. Should be 'polygon', 'line' or 'point'"
        )

    if ignore_index:
        gdf = gdf.reset_index(drop=True)

    return gdf


def is_single_geom_type(
    gdf: GeoDataFrame | GeoSeries,
) -> bool:
    """
    Returns True if all the geometries in the GeoDataFrame are of the same type,
    either polygon, line or point. Multipart and singlepart are considered the same
    type.

    Args:
      gdf: GeoDataFrame or GeoSeries

    Returns:
      True if all geometries are the same type, False if not
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
    """Returns a string of the geometry type in a GeoDataFrame or GeoSeries

    Args:
      gdf: GeoDataFrame or GeoSeries

    Returns:
      A string that is either "polygon", "line", "point", or "mixed"
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


def _close_holes_poly(poly, max_km2=None):
    """closes holes within one shapely geometry of polygons."""

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


def gdf_concat(
    gdfs: list[GeoDataFrame],
    crs: str | int | None = None,
    ignore_index: bool = True,
    geometry: str = "geometry",
    **kwargs,
) -> GeoDataFrame:
    """Sets common crs and concatinates GeoDataFrames rowwise while ignoring index

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

    """

    gdfs = [gdf for gdf in gdfs if len(gdf)]

    if not len(gdfs):
        raise ValueError("All GeoDataFrames have 0 rows")

    if not crs:
        crs = gdfs[0].crs

    try:
        gdfs = [gdf.to_crs(crs) for gdf in gdfs]
    except ValueError:
        print(
            "Not all your GeoDataFrames have crs. If you are concatenating "
            "GeoDataFrames with different crs, the results will be wrong. First use "
            "set_crs to set the correct crs then the crs can be changed with to_crs()"
        )

    return GeoDataFrame(
        pd.concat(gdfs, ignore_index=ignore_index, **kwargs), geometry=geometry, crs=crs
    )


def to_gdf(
    geom: GeoSeries | Geometry | str | bytes, crs=None, **kwargs
) -> GeoDataFrame:
    """Converts GeoSeries, shapely Geometry, wkt string or wkb bytes to a GeoDataFrame.

    Args:
        geom: the object to be converted to a GeoDataFrame
        crs: if None (the default), it uses the crs of the GeoSeries if GeoSeries
            is the input type. Otherwise, an exception is raised, saying that
            crs must be specified.
        **kwargs: additional keyword arguments taken by the GeoDataFrame constructor

    Returns:
        A GeoDataFrame with one column, the geometry column.

    Raises:
        ValueError if 'crs' is not specified and the input type is not GeoSeries

    """

    if not crs:
        if not isinstance(geom, GeoSeries):
            raise ValueError("'crs' have to be specified when the input is a string.")
        crs = geom.crs

    if isinstance(geom, str):
        from shapely.wkt import loads

        geom = loads(geom)
        gdf = GeoDataFrame({"geometry": GeoSeries(geom)}, crs=crs, **kwargs)

    elif isinstance(geom, bytes):
        from shapely.wkb import loads

        geom = loads(geom)
        gdf = GeoDataFrame({"geometry": GeoSeries(geom)}, crs=crs, **kwargs)

    else:
        gdf = GeoDataFrame({"geometry": GeoSeries(geom)}, crs=crs, **kwargs)

    return gdf


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
    """geopandas.sjoin that removes index columns before and after

    geopandas.sjoin returns the column 'index_right', which throws
    an error the next time.

    Args:
        left_df: GeoDataFrame
        right_df: GeoDataFrame
        drop_dupcol: optionally remove all columns in right_df that is in left_df.
            Defaults to False, meaning no columns are dropped.
        **kwargs: keyword arguments taken by geopandas.sjoin

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


def find_neighbors(
    gdf: GeoDataFrame | GeoSeries,
    possible_neighbors: GeoDataFrame | GeoSeries,
    id_col: str,
    max_dist: int = 0,
) -> list[str]:
    """American alias for find_neighbours"""
    return find_neighbours(gdf, possible_neighbors, id_col, max_dist)


def random_points(n: int) -> GeoDataFrame:
    """Creates a GeoDataFrame with 'n' random points.

    Args:
        n: number of points/rows to create

    Returns:
        A GeoDataFrame of points with n rows
    """

    if isinstance(n, (str, float)):
        n = int(n)

    x = np_random(n)
    y = np_random(n)

    return GeoDataFrame((Point(geom) for geom in zip(x, y)), columns=["geometry"])
