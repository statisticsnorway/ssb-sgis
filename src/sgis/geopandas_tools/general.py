import numbers
import warnings
from typing import Any, Iterable

import geocoder
import numpy as np
import pandas as pd
import pyproj
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryDtype
from shapely import (
    Geometry,
    box,
    get_exterior_ring,
    get_interior_ring,
    get_num_interior_rings,
    get_parts,
)
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from .geometry_types import get_geom_type, make_all_singlepart, to_single_geom_type
from .to_geodataframe import to_gdf


def get_utm33(lon, lat, crs=25833):
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", f"EPSG:{crs}", always_xy=True
    )
    return transformer.transform(lon, lat)


def get_lonlat(lon, lat, crs=25833):
    transformer = pyproj.Transformer.from_crs(
        f"EPSG:{crs}", "EPSG:4326", always_xy=True
    )
    return transformer.transform(lon, lat)


def get_common_crs(iterable: Iterable[Any], strict: bool = False) -> pyproj.CRS | None:
    """Returns the common not-None crs or raises a ValueError if more than one.

    Args:
        iterable: Iterable of objects with the attribute "crs" or a list
            of CRS-like (pyproj.CRS-accepted) objects.
        strict: If False (default), falsy CRS-es will be ignored and None
            will be returned if all CRS-es are falsy. If strict is True,

    Returns:
        pyproj.CRS object or None (if all crs are None).

    Raises:
        ValueError if there are more than one crs. If strict is True,
        None is included.
    """
    try:
        crs = list({x.crs for x in iterable})
    except AttributeError:
        crs = list(set(iterable))

    if not crs:
        return None

    truthy_crs = list({x for x in crs if x})

    if strict and len(truthy_crs) != len(crs):
        raise ValueError("Mix of falsy and truthy CRS-es found.")

    if len(truthy_crs) > 1:
        raise ValueError("'crs' mismatch.", truthy_crs)
    return pyproj.CRS(truthy_crs[0])


def is_bbox_like(obj) -> bool:
    if (
        hasattr(obj, "__iter__")
        and len(obj) == 4
        and all(isinstance(x, numbers.Number) for x in obj)
    ):
        return True
    return False


def to_shapely(obj) -> Geometry:
    if isinstance(obj, Geometry):
        return obj
    if not hasattr(obj, "__iter__"):
        raise TypeError(type(obj))
    if hasattr(obj, "unary_union"):
        return obj.unary_union
    if is_bbox_like(obj):
        return box(*obj)
    try:
        return Point(*obj)
    except TypeError as e:
        raise TypeError(obj) from e


def address_to_gdf(address: str, crs=4326) -> GeoDataFrame:
    """Takes an address and returns a point GeoDataFrame."""
    g = geocoder.osm(address).json
    coords = g["lng"], g["lat"]
    return to_gdf(coords, crs=4326).to_crs(crs)


def address_to_coords(address: str, crs=4326) -> tuple[float, float]:
    """Takes an address and returns a tuple of xy coordinates."""
    g = geocoder.osm(address).json
    coords = g["lng"], g["lat"]
    point = to_gdf(coords, crs=4326).to_crs(crs)
    x, y = point.geometry.iloc[0].x, point.geometry.iloc[0].y
    return x, y


def is_wkt(text: str) -> bool:
    gemetry_types = ["point", "polygon", "line", "geometrycollection"]
    return any(x in text.lower() for x in gemetry_types)


def coordinate_array(
    gdf: GeoDataFrame | GeoSeries,
) -> np.ndarray[np.ndarray[float], np.ndarray[float]]:
    """Creates a 2d ndarray of coordinates from point geometries.

    Args:
        gdf: GeoDataFrame or GeoSeries of point geometries.

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
    >>> coordinate_array(points.geometry)
    array([[0.59376221, 0.92577159],
        [0.34074678, 0.91650446],
        [0.74840912, 0.10626954],
        [0.00965935, 0.87867915],
        [0.38045827, 0.87878816]])
    """
    if isinstance(gdf, GeoDataFrame):
        return np.array([(geom.x, geom.y) for geom in gdf.geometry])
    else:
        return np.array([(geom.x, geom.y) for geom in gdf])


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


def rename_geometry_if(gdf: GeoDataFrame) -> GeoDataFrame:
    geom_col = gdf._geometry_column_name
    if geom_col == "geometry" and geom_col in gdf.columns:
        return gdf
    elif geom_col in gdf.columns:
        return gdf.rename_geometry("geometry")

    geom_cols = list(
        {col for col in gdf.columns if isinstance(gdf[col].dtype, GeometryDtype)}
    )
    if len(geom_cols) == 1:
        gdf._geometry_column_name = geom_cols[0]
        return gdf.rename_geometry("geometry")

    raise ValueError(
        "There are multiple geometry columns and none are the active geometry"
    )


def clean_geoms(
    gdf: GeoDataFrame | GeoSeries,
    ignore_index: bool = False,
) -> GeoDataFrame | GeoSeries:
    """Fixes geometries, then removes empty, NaN and None geometries.

    Args:
        gdf: GeoDataFrame or GeoSeries to be cleaned.
        ignore_index: If True, the resulting axis will be labeled 0, 1, …, n - 1.
            Defaults to False

    Returns:
        GeoDataFrame or GeoSeries with fixed geometries and only the rows with valid,
        non-empty and not-NaN/-None geometries.

    Examples
    --------
    >>> import sgis as sg
    >>> import pandas as pd
    >>> from shapely import wkt
    >>> gdf = sg.to_gdf([
    ...         "POINT (0 0)",
    ...         "LINESTRING (1 1, 2 2)",
    ...         "POLYGON ((3 3, 4 4, 3 4, 3 3))"
    ...         ])
    >>> gdf
                                                geometry
    0                            POINT (0.00000 0.00000)
    1      LINESTRING (1.00000 1.00000, 2.00000 2.00000)
    2  POLYGON ((3.00000 3.00000, 4.00000 4.00000, 3....

    Add None and empty geometries.

    >>> missing = pd.DataFrame({"geometry": [None]})
    >>> empty = sg.to_gdf(wkt.loads("POINT (0 0)").buffer(0))
    >>> gdf = pd.concat([gdf, missing, empty])
    >>> gdf
                                                geometry
    0                            POINT (0.00000 0.00000)
    1      LINESTRING (1.00000 1.00000, 2.00000 2.00000)
    2  POLYGON ((3.00000 3.00000, 4.00000 4.00000, 3....
    0                                               None
    0                                      POLYGON EMPTY

    Clean.

    >>> sg.clean_geoms(gdf)
                                                geometry
    0                            POINT (0.00000 0.00000)
    1      LINESTRING (1.00000 1.00000, 2.00000 2.00000)
    2  POLYGON ((3.00000 3.00000, 4.00000 4.00000, 3....
    """
    warnings.filterwarnings("ignore", "GeoSeries.notna", UserWarning)

    if isinstance(gdf, GeoDataFrame):
        geom_col = gdf._geometry_column_name

        # only repair if necessary
        if not gdf[geom_col].is_valid.all():
            gdf[geom_col] = gdf.make_valid()

        notna = gdf.geometry.notna()
        if not notna.all():
            gdf = gdf.loc[notna]

        nonempty = gdf.geometry.map(bool)
        if not nonempty.all():
            gdf = gdf.loc[nonempty]

    elif isinstance(gdf, GeoSeries):
        if not gdf.is_valid.all():
            gdf = gdf.make_valid()

        notna = gdf.notna()
        if not notna.all():
            gdf = gdf.loc[notna]

        nonempty = gdf.map(bool)
        if not nonempty.all():
            gdf = gdf.loc[nonempty]

    else:
        raise TypeError(f"'gdf' should be GeoDataFrame or GeoSeries, got {type(gdf)}")

    if ignore_index:
        gdf = gdf.reset_index(drop=True)

    return gdf


def get_grouped_centroids(
    gdf: GeoDataFrame, groupby: str, as_string: bool = True
) -> pd.Series:
    centerpoints = gdf.assign(geometry=lambda x: x.centroid)

    grouped_centerpoints = centerpoints.dissolve(by=groupby).assign(
        geometry=lambda x: x.centroid
    )
    xs = grouped_centerpoints.geometry.x
    ys = grouped_centerpoints.geometry.y

    if as_string:
        grouped_centerpoints["wkt"] = [f"{int(x)}_{int(y)}" for x, y in zip(xs, ys)]
    else:
        grouped_centerpoints["wkt"] = [Point(x, y) for x, y in zip(xs, ys)]

    return gdf[groupby].map(grouped_centerpoints["wkt"])


def sort_large_first(gdf: GeoDataFrame) -> GeoDataFrame:
    """Sort GeoDataFrame by area in decending order.

    Examples
    --------
    Create GeoDataFrame with NaN values.

    >>> import sgis as sg
    >>> df = sg.random_points(5)
    >>> df.geometry = df.buffer([4, 1, 2, 3, 5])
    >>> df["col"] = [None, 1, 2, None, 1]
    >>> df["col2"] = [None, 1, 2, 3, None]
    >>> df["area"] = df.area
    >>> df
                                                geometry  col  col2       area
    0  POLYGON ((4.56136 0.53436, 4.54210 0.14229, 4....  NaN   NaN  50.184776
    1  POLYGON ((1.40111 0.71798, 1.39630 0.61996, 1....  1.0   1.0   3.136548
    2  POLYGON ((2.33302 0.49287, 2.32339 0.29683, 2....  2.0   2.0  12.546194
    3  POLYGON ((3.68381 0.46299, 3.66936 0.16894, 3....  NaN   3.0  28.228936
    4  POLYGON ((5.63590 0.16005, 5.61182 -0.33004, 5...  1.0   NaN  78.413712

    >>> sg.sort_large_first(df)
                                                geometry  col  col2       area
    4  POLYGON ((5.63590 0.16005, 5.61182 -0.33004, 5...  1.0   NaN  78.413712
    0  POLYGON ((4.56136 0.53436, 4.54210 0.14229, 4....  NaN   NaN  50.184776
    3  POLYGON ((3.68381 0.46299, 3.66936 0.16894, 3....  NaN   3.0  28.228936
    2  POLYGON ((2.33302 0.49287, 2.32339 0.29683, 2....  2.0   2.0  12.546194
    1  POLYGON ((1.40111 0.71798, 1.39630 0.61996, 1....  1.0   1.0   3.136548

    >>> sg.sort_nans_last(sg.sort_large_first(df))
                                                geometry  col  col2       area
    2  POLYGON ((2.33302 0.49287, 2.32339 0.29683, 2....  2.0   2.0  12.546194
    1  POLYGON ((1.40111 0.71798, 1.39630 0.61996, 1....  1.0   1.0   3.136548
    4  POLYGON ((5.63590 0.16005, 5.61182 -0.33004, 5...  1.0   NaN  78.413712
    3  POLYGON ((3.68381 0.46299, 3.66936 0.16894, 3....  NaN   3.0  28.228936
    0  POLYGON ((4.56136 0.53436, 4.54210 0.14229, 4....  NaN   NaN  50.184776
    """
    return (
        gdf.assign(area_=gdf.area)
        .sort_values("area_", ascending=False)
        .drop(columns="area_")
    )


def sort_long_first(gdf: GeoDataFrame) -> GeoDataFrame:
    return (
        gdf.assign(length_=gdf.length)
        .sort_values("length_", ascending=False)
        .drop(columns="length_")
    )


def random_points(n: int, loc: float | int = 0.5) -> GeoDataFrame:
    """Creates a GeoDataFrame with n random points.

    Args:
        n: Number of points/rows to create.
        loc: Mean ('centre') of the distribution.

    Returns:
        A GeoDataFrame of points with n rows.

    Examples
    --------
    >>> import sgis as sg
    >>> points = sg.random_points(10_000)
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

    >>> points = sg.random_points(10_000, loc=100)
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

    if any(gdf.geom_type.isin(["Point", "MultiPoint"]).any() for gdf in gdfs):
        raise ValueError("Cannot convert points to lines.")

    def _shapely_geometry_to_lines(geom):
        """Get all lines from the exterior and interiors of a Polygon."""

        # if lines (points are not allowed in this function)
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

            lines += interior_rings

        return unary_union(lines)

    lines = []
    for gdf in gdfs:
        if copy:
            gdf = gdf.copy()

        gdf[gdf._geometry_column_name] = gdf[gdf._geometry_column_name].map(
            _shapely_geometry_to_lines
        )

        gdf = to_single_geom_type(gdf, "line")

        lines.append(gdf)

    if len(lines) == 1:
        return lines[0]

    if len(lines[0]) and len(lines[1]):
        unioned = lines[0].overlay(lines[1], how="union", keep_geom_type=True)
    else:
        unioned = pd.concat([lines[0], lines[1]], ignore_index=True)

    if len(lines) > 2:
        for line_gdf in lines[2:]:
            if len(line_gdf):
                unioned = unioned.overlay(line_gdf, how="union", keep_geom_type=True)
            else:
                unioned = pd.concat([unioned, line_gdf], ignore_index=True)

    return make_all_singlepart(unioned, ignore_index=True)


def clean_clip(
    gdf: GeoDataFrame | GeoSeries,
    mask: GeoDataFrame | GeoSeries | Geometry,
    **kwargs,
) -> GeoDataFrame | GeoSeries:
    """Clips and clean geometries.

    Geopandas.clip does a "fast and dirty clipping, with no guarantee for valid
    outputs". Here, the clipped geometries are made valid, and empty and NaN
    geometries are removed.

    Args:
        gdf: GeoDataFrame or GeoSeries to be clipped
        mask: the geometry to clip gdf
        **kwargs: Keyword arguments passed to geopandas.GeoDataFrame.clip

    Returns:
        The cleanly clipped GeoDataFrame.

    Raises:
        TypeError: If gdf is not of type GeoDataFrame or GeoSeries.
    """
    if not isinstance(gdf, (GeoDataFrame, GeoSeries)):
        raise TypeError(f"'gdf' should be GeoDataFrame or GeoSeries, got {type(gdf)}")

    if kwargs.get("keep_geom_type"):
        geom_type = get_geom_type(gdf)

    try:
        gdf = gdf.clip(mask, **kwargs).pipe(clean_geoms)
    except Exception:
        gdf = clean_geoms(gdf)
        try:
            mask = clean_geoms(mask)
        except TypeError:
            mask = clean_geoms(to_gdf(mask, crs=gdf.crs))

        return gdf.clip(mask, **kwargs).pipe(clean_geoms)

    if kwargs.get("keep_geom_type"):
        gdf = to_single_geom_type(gdf, geom_type)

    return gdf
