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
from shapely import Geometry, wkb, wkt
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
from shapely import (
    get_exterior_ring,
    get_interior_ring,
    get_num_interior_rings,
    get_parts,
)

from .geometry_types import to_single_geom_type
from .buffer_dissolve_explode import exp


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
    >>> from sgis import clean_geoms, to_gdf, gdf_concat
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


def random_points(n: int) -> GeoDataFrame:
    """Creates a GeoDataFrame with n random points.

    Args:
        n: number of points/rows to create.

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
    """
    if isinstance(n, (str, float)):
        n = int(n)

    x = _np_random(n)
    y = _np_random(n)

    return GeoDataFrame(
        (Point(x, y) for x, y in zip(x, y, strict=True)), columns=["geometry"]
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

    Examples
    --------
    >>> from sgis import gdf_concat, to_gdf
    >>> points = to_gdf([(0, 0), (0.5, 0.5), (2, 2)])
    >>> points
                    geometry
    0  POINT (0.00000 0.00000)
    1  POINT (0.50000 0.50000)
    2  POINT (2.00000 2.00000)
    >>> gdf_concat([points, points])
    C:/Users/ort/git/ssb-gis-utils/src/sgis/geopandas_utils.py:828: UserWarning: None of your GeoDataFrames have crs.
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

    # to list
    gdfs = [gdf for gdf in gdfs]

    columns = list(set(col for gdf in gdfs for col in gdf.columns))
    gdfs = [gdf for gdf in gdfs if len(gdf)]

    if not gdfs:
        return pd.DataFrame(columns=columns)

    gdfs = [rename_geometry_if(gdf) for gdf in gdfs]

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

    # ignoring erronous warning
    warnings.filterwarnings(action="ignore", category=UserWarning)
    return GeoDataFrame(
        pd.concat(gdfs, ignore_index=ignore_index, **kwargs), geometry=geometry, crs=crs
    )


def to_lines(
    *gdfs: GeoDataFrame, ignore_index: bool = True, copy: bool = True
) -> GeoDataFrame:
    """Makes lines out of one or more GeoDataFrames.

    The GeoDataFrames' geometries are converted to LineStrings, then unioned and
    exploded. The lines are split at the intersections. Mimics 'feature to line' in
    ArcGIS.

    Args:
        *gdfs: one or more GeoDataFrames.
        ignore_index: If True, the resulting axis will be labeled 0, 1, …, n - 1.
            Defaults to True
        copy: whether to take a copy of the incoming GeoDataFrames. Defaults to True.

    Returns:
        A GeoDataFrame with singlepart line geometries and columns of all input
            GeoDataFrames.

    Examples
    --------

    Convert single polygon to linestring.

    >>> import sgis as sg
    >>> from shapely.geometry import Polygon
    >>> poly1 = sg.to_gdf(Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]))
    >>> poly1["poly1"] = 1
    >>> line = sg.to_lines(poly1)
    >>> line
        poly1                                           geometry
    0      1  LINESTRING (0.00000 0.00000, 0.00000 1.00000, ...

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

    Plot before and after (plots not showing in terminal).

    >>> sg.qtm(poly1, poly2)
    <Axes: >
    >>> lines["l"] = lines.length
    >>> sg.qtm(lines, "l")
    <Axes: >
    """

    def _to_lines(geom):
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

        gdf[gdf._geometry_column_name] = gdf[gdf._geometry_column_name].map(_to_lines)

        lines.append(gdf)

    if len(lines) == 1:
        return exp(lines[0], ignore_index=ignore_index)

    unioned = lines[0].overlay(lines[1], how="union", keep_geom_type=True)

    if len(lines) > 2:
        for line_gdf in lines[2:]:
            unioned = unioned.overlay(line_gdf, how="union", keep_geom_type=True)

    return exp(unioned, ignore_index=ignore_index)


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
    >>> from sgis import to_gdf
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
