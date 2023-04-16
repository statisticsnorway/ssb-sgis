import numbers
import warnings
from collections.abc import Iterator, Sized

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryDtype
from numpy.random import random as _np_random
from pandas.api.types import is_dict_like
from shapely import (
    Geometry,
    force_2d,
    get_exterior_ring,
    get_interior_ring,
    get_num_interior_rings,
    get_parts,
    wkb,
    wkt,
)
from shapely.geometry import LineString, Point
from shapely.ops import unary_union


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
    >>> from sgis import coordinate_array, random_points
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


def _push_geom_col(gdf: GeoDataFrame) -> GeoDataFrame:
    """Makes the geometry column the rightmost column in the GeoDataFrame.

    Args:
        gdf: GeoDataFrame.

    Returns:
        The GeoDataFrame with the geometry column pushed all the way to the right.
    """
    geom_col = gdf._geometry_column_name
    return gdf.reindex(columns=[c for c in gdf.columns if c != geom_col] + [geom_col])


def drop_inactive_geometry_columns(gdf: GeoDataFrame) -> GeoDataFrame:
    for col in gdf.columns:
        if (
            isinstance(gdf[col].dtype, GeometryDtype)
            and col != gdf._geometry_column_name
        ):
            gdf = gdf.drop(col, axis=1)
    return gdf


def rename_geometry_if(gdf):
    geom_col = gdf._geometry_column_name
    if geom_col == "geometry":
        return gdf
    return gdf.rename_geometry("geometry")


def clean_geoms(
    gdf: GeoDataFrame | GeoSeries, ignore_index: bool = False
) -> GeoDataFrame | GeoSeries:
    """Fixes geometries and removes invalid, empty, NaN and None geometries.

    Args:
        gdf: GeoDataFrame or GeoSeries to be cleaned.
        ignore_index: If True, the resulting axis will be labeled 0, 1, …, n - 1.
            Defaults to False

    Returns:
        GeoDataFrame or GeoSeries with fixed geometries and only the rows with valid,
        non-empty and not-NaN/-None geometries.

    Examples
    --------
    >>> from sgis import clean_geoms, to_gdf
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

    Removing None and empty geometries.

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

    if ignore_index:
        gdf = gdf.reset_index(drop=True)

    return gdf


def random_points(n: int, loc: float | int = 0.5) -> GeoDataFrame:
    """Creates a GeoDataFrame with n random points.

    Args:
        n: Number of points/rows to create.
        loc: Mean ('centre') of the distribution.

    Returns:
        A GeoDataFrame of points with n rows.

    Examples
    --------
    >>> from sgis import random_points
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

    Values with a mean of 100.

    >>> points = random_points(10_000, loc=100)
    >>> points
                         geometry
    0      POINT (50.442 199.729)
    1       POINT (26.450 83.367)
    2     POINT (111.054 147.610)
    3      POINT (93.141 141.456)
    4       POINT (94.101 24.837)
    ...                       ...
    9995   POINT (174.344 91.772)
    9996    POINT (95.375 11.391)
    9997    POINT (45.694 60.843)
    9998   POINT (73.261 101.881)
    9999  POINT (134.503 168.155)
    [10000 rows x 1 columns]
    """
    if isinstance(n, (str, float)):
        n = int(n)

    x = np.random.rand(n) * float(loc) * 2
    y = np.random.rand(n) * float(loc) * 2

    return GeoDataFrame(
        (Point(x, y) for x, y in zip(x, y, strict=True)), columns=["geometry"]
    )


def random_points_in_polygons(
    gdf: GeoDataFrame, n: int, ignore_index=False
) -> GeoDataFrame:
    """Creates n random points inside each polygon of a GeoDataFrame.

    Args:
        gdf: GeoDataFrame to use as mask for the points.
        n: Number of points to create per polygon in 'gdf'.
        ignore_index: If True, the resulting axis will be labeled 0, 1, …, n - 1.
            Defaults to False, meaning the points will have the index of the polygon
            it is within.

    Returns:
        A GeoDataFrame of points with 'n' rows per row in 'gdf'. It uses the index
        values of 'gdf'.

    Examples
    --------
    First create and buffer 100 random points.

    >>> import sgis as sg
    >>> gdf = sg.random_points(100)
    >>> polygons = sg.buff(gdf, 1)
    >>> polygons
                                                 geometry
    0   POLYGON ((1.49436 0.36088, 1.49387 0.32947, 1....
    1   POLYGON ((1.38427 0.21069, 1.38378 0.17928, 1....
    2   POLYGON ((1.78894 0.94134, 1.78845 0.90992, 1....
    3   POLYGON ((1.47174 0.81259, 1.47125 0.78118, 1....
    4   POLYGON ((1.13941 0.20821, 1.13892 0.17680, 1....
    ..                                                ...
    95  POLYGON ((1.13462 0.18908, 1.13412 0.15767, 1....
    96  POLYGON ((1.96391 0.43191, 1.96342 0.40050, 1....
    97  POLYGON ((1.30569 0.46956, 1.30520 0.43815, 1....
    98  POLYGON ((1.18172 0.10944, 1.18122 0.07803, 1....
    99  POLYGON ((1.06156 0.99893, 1.06107 0.96752, 1....
    [100 rows x 1 columns]

    >>> points = sg.random_points_in_polygons(polygons, 3)
    >>> points
                        geometry
    0   POINT (0.74944 -0.41658)
    0    POINT (1.27490 0.54076)
    0    POINT (0.22523 0.49323)
    1   POINT (0.25302 -0.34825)
    1    POINT (0.21124 0.89223)
    ..                       ...
    98  POINT (-0.39865 0.87135)
    98   POINT (0.03573 0.50788)
    99  POINT (-0.79089 0.57835)
    99   POINT (0.39838 1.50881)
    99   POINT (0.98383 0.77298)
    [300 rows x 1 columns]
    """

    if not all(gdf.geom_type.isin(["Polygon", "MultiPolygon"])):
        raise ValueError("Geometry types must be polygon.")

    if gdf.index.is_unique:
        gdf["temp_idx____"] = gdf.index
    else:
        gdf["temp_idx____"] = range(len(gdf))

    all_points = pd.DataFrame()

    for _ in range(n):
        bounds = gdf.bounds
        temp_idx____ = gdf["temp_idx____"].values
        overlapping = pd.DataFrame()
        overlapping_indices = ()

        while len(bounds):
            xs = np.random.uniform(bounds.minx, bounds.maxx)
            ys = np.random.uniform(bounds.miny, bounds.maxy)

            points_df = pd.DataFrame({"x": xs, "y": ys}, index=temp_idx____)

            points = to_gdf(points_df, geometry=["x", "y"], crs=gdf.crs).drop(
                ["x", "y"], axis=1
            )

            overlapping = points.sjoin(gdf[["temp_idx____", "geometry"]], how="inner")

            overlapping = overlapping.loc[overlapping.index == overlapping.temp_idx____]

            all_points = pd.concat([all_points, overlapping], ignore_index=ignore_index)

            overlapping_indices = overlapping_indices + tuple(overlapping.index.values)

            gdf__ = gdf.loc[~gdf["temp_idx____"].isin(overlapping_indices)]
            temp_idx____ = gdf__["temp_idx____"].values
            bounds = gdf__.bounds

    all_points = all_points.sort_index().drop(["temp_idx____", "index_right"], axis=1)

    if gdf.index.is_unique:
        gdf = gdf.drop("temp_idx____", axis=1)
        return all_points

    original_index = {
        temp_idx: idx for temp_idx, idx in zip(gdf.temp_idx____, gdf.index)
    }

    all_points.index = all_points.index.map(original_index)
    all_points.index.name = None

    gdf = gdf.drop("temp_idx____", axis=1)

    return all_points


def points_in_bounds(gdf: GeoDataFrame, n2: int):
    minx, miny, maxx, maxy = gdf.total_bounds
    xs = np.linspace(minx, maxx, num=n2)
    ys = np.linspace(miny, maxy, num=n2)
    x_coords, y_coords = np.meshgrid(xs, ys, indexing="ij")
    coords = np.concatenate((x_coords.reshape(-1, 1), y_coords.reshape(-1, 1)), axis=1)
    return to_gdf(coords, crs=gdf.crs)


def to_lines(*gdfs: GeoDataFrame, copy: bool = True) -> GeoDataFrame:
    """Makes lines out of one or more GeoDataFrames and splits them at intersections.

    The GeoDataFrames' geometries are converted to LineStrings, then unioned together
    and made to singlepart. The lines are split at the intersections. Mimics
    'feature to line' in ArcGIS.

    Args:
        *gdfs: one or more GeoDataFrames.
        copy: whether to take a copy of the incoming GeoDataFrames. Defaults to True.

    Returns:
        A GeoDataFrame with singlepart line geometries and columns of all input
            GeoDataFrames.

    Note:
        The index is preserved if only one GeoDataFrame is given, but otherwise
        ignored. This is because the union overlay used if multiple GeoDataFrames
        always ignores the index.

    Examples
    --------
    Convert single polygon to linestring.

    >>> import sgis as sg
    >>> from shapely.geometry import Polygon
    >>> poly1 = sg.to_gdf(Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]))
    >>> poly1["poly1"] = 1
    >>> line = sg.to_lines(poly1)
    >>> line
                                                geometry  poly1
    0  LINESTRING (0.00000 0.00000, 0.00000 1.00000, ...      1

    Convert two overlapping polygons to linestrings.

    >>> poly2 = sg.to_gdf(Polygon([(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)]))
    >>> poly2["poly2"] = 1
    >>> lines = sg.to_lines(poly1, poly2)
    >>> lines
    poly1  poly2                                           geometry
    0    1.0    NaN  LINESTRING (0.00000 0.00000, 0.00000 1.00000, ...
    1    1.0    NaN  LINESTRING (0.50000 1.00000, 1.00000 1.00000, ...
    2    1.0    NaN  LINESTRING (1.00000 0.50000, 1.00000 0.00000, ...
    3    NaN    1.0      LINESTRING (0.50000 0.50000, 0.50000 1.00000)
    4    NaN    1.0  LINESTRING (0.50000 1.00000, 0.50000 1.50000, ...
    5    NaN    1.0      LINESTRING (1.00000 0.50000, 0.50000 0.50000)

    Plot before and after.

    >>> sg.qtm(poly1, poly2)
    >>> lines["l"] = lines.length
    >>> sg.qtm(lines, "l")
    """

    if any(any(gdf.geom_type.isin(["Point", "MultiPoint"])) for gdf in gdfs):
        raise ValueError("Cannot convert points to lines.")

    def _shapely_geometry_to_lines(geom):
        if geom.area == 0:
            return geom

        singlepart = get_parts(geom)
        lines = []
        for part in singlepart:
            exterior_ring = get_exterior_ring(part)
            lines.append(exterior_ring)

            n_interior_rings = get_num_interior_rings(part)
            if not (n_interior_rings):
                continue

            interior_rings = [
                LineString(get_interior_ring(part, n)) for n in range(n_interior_rings)
            ]

            lines = lines + interior_rings

        return unary_union(lines)

    lines = []
    for gdf in gdfs:
        if copy:
            gdf = gdf.copy()

        gdf[gdf._geometry_column_name] = gdf[gdf._geometry_column_name].map(
            _shapely_geometry_to_lines
        )

        lines.append(gdf)

    if len(lines) == 1:
        return lines[0]

    unioned = lines[0].overlay(lines[1], how="union", keep_geom_type=True)

    if len(lines) > 2:
        for line_gdf in lines[2:]:
            unioned = unioned.overlay(line_gdf, how="union", keep_geom_type=True)

    return unioned.explode(ignore_index=True)


def to_multipoint(
    gdf: GeoDataFrame | GeoSeries | Geometry, copy: bool = True
) -> GeoDataFrame | GeoSeries | Geometry:
    """Creates a multipoint geometry of any geometry object.

    Takes a GeoDataFrame, GeoSeries or Shapely geometry and turns it into a MultiPoint.
    If the input is a GeoDataFrame or GeoSeries, the rows and columns will be preserved,
    but with a geometry column of MultiPoints.

    Args:
        gdf: The geometry to be converted to MultiPoint. Can be a GeoDataFrame,
            GeoSeries or a shapely geometry.
        copy: If True, the geometry will be copied. Defaults to True.

    Returns:
        A GeoDataFrame with the geometry column as a MultiPoint, or Point if the
        original geometry was a point.

    Examples
    --------
    Let's create a GeoDataFrame with a point, a line and a polygon.

    >>> from sgis import to_multipoint, to_gdf
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
    if copy and not isinstance(gdf, Geometry):
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


def clean_clip(
    gdf: GeoDataFrame | GeoSeries,
    mask: GeoDataFrame | GeoSeries | Geometry,
    **kwargs,
) -> GeoDataFrame | GeoSeries:
    """Clips geometries to the mask extent and cleans the geometries.

    Geopandas.clip does a fast and durty clipping, with no guarantee for valid outputs.
    Here, the clipped geometries are made valid, and then empty, NaN and invalid
    geometries are removed.

    Args:
        gdf: GeoDataFrame or GeoSeries to be clipped
        mask: the geometry to clip gdf
        **kwargs: Additional keyword arguments passed to GeoDataFrame.clip

    Returns:
        The cleanly clipped GeoDataFrame.

    Raises:
        TypeError: If gdf is not of type GeoDataFrame or GeoSeries.
    """
    if not isinstance(gdf, (GeoDataFrame, GeoSeries)):
        raise TypeError(f"'gdf' should be GeoDataFrame or GeoSeries, got {type(gdf)}")

    try:
        return gdf.clip(mask, **kwargs).pipe(clean_geoms)
    except Exception:
        gdf = clean_geoms(gdf)
        mask = clean_geoms(mask)
        return gdf.clip(mask, **kwargs).pipe(clean_geoms)


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
    | Iterator,
    crs: str | tuple[str] | None = None,
    geometry: str | None = None,
    **kwargs,
) -> GeoDataFrame:
    """Converts geometry-like objects to a GeoDataFrame.

    Constructs a GeoDataFrame from any geometry-like object, or an interable of such.
    Accepted types are string (wkt), byte (wkb), coordinate tuples, shapely geometries,
    GeoSeries, Series/DataFrame. The index/keys will be preserved if the input type is
    Series, DataFrame or dictionary.

    Args:
        geom: the object to be converted to a GeoDataFrame
        crs: if None (default), it uses the crs of the GeoSeries if GeoSeries
            is the input type. Otherwise, no crs is used.
        geometry: name of column(s) containing the geometry-like values. Can be
            ['x', 'y', 'z'] if coordinate columns, or e.g. 'geometry' if one column.
            The resulting geometry column will always be named 'geometry'.
        **kwargs: additional keyword arguments taken by the GeoDataFrame constructor.

    Returns:
        A GeoDataFrame with one column, the geometry column.

    Raises:
        TypeError: If geom is a GeoDataFrame.

    Examples
    --------
    >>> from sgis import to_gdf
    >>> coords = (10, 60)
    >>> to_gdf(coords, crs=4326)
                        geometry
    0  POINT (10.00000 60.00000)

    From wkt.

    >>> wkt = "POINT (10 60)"
    >>> to_gdf(wkt, crs=4326)
                        geometry
    0  POINT (10.00000 60.00000)

    From DataFrame with x, y (optionally z) coordinate columns. Index and
    columns are preserved.

    >>> df = pd.DataFrame({"x": [10, 11], "y": [60, 59]}, index=[1,3])
        x   y
    1  10  60
    3  11  59
    >>> gdf = to_gdf(df, geometry=["x", "y"], crs=4326)
    >>> gdf
        x   y                   geometry
    1  10  60  POINT (10.00000 60.00000)
    3  11  59  POINT (11.00000 59.00000)

    For DataFrame/dict with a geometry-like column, the geometry column can be
    speficied with the geometry parameter, which is set to "geometry" by default.

    >>> df = pd.DataFrame({"col": [1, 2], "geometry": ["point (10 60)", (11, 59)]})
    >>> df
       col       geometry
    0    1  point (10 60)
    1    2       (11, 59)
    >>> gdf = to_gdf(df, crs=4326)
    >>> gdf
       col                   geometry
    0    1  POINT (10.00000 60.00000)
    1    2  POINT (11.00000 59.00000)

    From Series or Series-like dictionary.

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

    Multiple coordinates will be converted to points, unless a line or polygon geometry
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

    From 2 or 3 dimensional array.

    >>> arr = np.random.randint(100, size=(5, 3))
    >>> to_gdf(arr)
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
        geom = GeoSeries(_make_shapely_geom(x) for x in geom)
        return GeoDataFrame({"geometry": geom}, geometry="geometry", crs=crs, **kwargs)

    # dataframes and dicts with geometry/xyz key(s)
    geometry = "geometry" if not geometry else geometry
    if _is_df_like(geom, geometry):
        geom = geom.copy()
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


def _is_df_like(geom, geometry: str | None) -> bool:
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
        if not hasattr(geom[geometry], "__iter__"):
            geom[geometry] = [geom[geometry]]
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
            "pd.concat(to_gdf(geom) for geom in geoseries_iterable)"
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
