import numbers
import warnings
from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
import pandas as pd
import pyproj
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryArray, GeometryDtype
from numpy.typing import NDArray
from shapely import (
    Geometry,
    get_coordinates,
    get_exterior_ring,
    get_interior_ring,
    get_num_interior_rings,
    get_parts,
    linestrings,
    make_valid,
)
from shapely import points as shapely_points
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from .geometry_types import get_geom_type, make_all_singlepart, to_single_geom_type


def split_geom_types(gdf: GeoDataFrame | GeoSeries) -> tuple[GeoDataFrame | GeoSeries]:
    return tuple(
        gdf.loc[gdf.geom_type == geom_type] for geom_type in gdf.geom_type.unique()
    )


def get_common_crs(
    iterable: Iterable[Hashable], strict: bool = False
) -> pyproj.CRS | None:
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
    crs = set()
    for obj in iterable:
        try:
            crs.add(obj.crs)
        except AttributeError:
            pass

    if not crs:
        try:
            crs = list(set(iterable))
        except TypeError:
            return None

    truthy_crs = list({x for x in crs if x})

    if strict and len(truthy_crs) != len(crs):
        raise ValueError("Mix of falsy and truthy CRS-es found.")

    if len(truthy_crs) > 1:
        # sometimes the bbox is slightly different, resulting in different
        # hash values for same crs. Therefore, trying to
        actually_different = set()
        for x in truthy_crs:
            if x.to_string() in {j.to_string() for j in actually_different}:
                continue
            actually_different.add(x)

        if len(actually_different) == 1:
            return list(actually_different)[0]
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


def is_wkt(text: str) -> bool:
    gemetry_types = ["point", "polygon", "line", "geometrycollection"]
    return any(x in text.lower() for x in gemetry_types)


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
        # only repair if necessary
        if not gdf.geometry.is_valid.all():
            gdf.geometry = gdf.make_valid()

        notna = gdf.geometry.notna()
        if not notna.all():
            gdf = gdf.loc[notna]

        is_empty = gdf.geometry.is_empty
        if is_empty.any():
            gdf = gdf.loc[~is_empty]

    elif isinstance(gdf, GeoSeries):
        if not gdf.is_valid.all():
            gdf = gdf.make_valid()

        notna = gdf.notna()
        if not notna.all():
            gdf = gdf.loc[notna]

        is_empty = gdf.is_empty
        if is_empty.any():
            gdf = gdf.loc[~is_empty]

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


def sort_large_first(gdf: GeoDataFrame | GeoSeries) -> GeoDataFrame | GeoSeries:
    """Sort GeoDataFrame by area in decending order.

    Args:
        gdf: A GeoDataFrame or GeoSeries.

    Returns:
        A GeoDataFrame or GeoSeries sorted from large to small in area.

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
    # using enumerate, then iloc on the sorted dict keys.
    # to avoid creating a temporary area column (which doesn't work for GeoSeries).
    area_mapper = dict(enumerate(gdf.area.values))
    sorted_areas = dict(reversed(sorted(area_mapper.items(), key=lambda item: item[1])))
    return gdf.iloc[list(sorted_areas)]


def sort_long_first(gdf: GeoDataFrame | GeoSeries) -> GeoDataFrame | GeoSeries:
    """Sort GeoDataFrame by length in decending order.

    Args:
        gdf: A GeoDataFrame or GeoSeries.

    Returns:
        A GeoDataFrame or GeoSeries sorted from long to short in length.
    """
    # using enumerate, then iloc on the sorted dict keys.
    # to avoid creating a temporary area column (which doesn't work for GeoSeries).
    length_mapper = dict(enumerate(gdf.length.values))
    sorted_lengths = dict(
        reversed(sorted(length_mapper.items(), key=lambda item: item[1]))
    )
    return gdf.iloc[list(sorted_lengths)]


def sort_short_first(gdf: GeoDataFrame | GeoSeries) -> GeoDataFrame | GeoSeries:
    """Sort GeoDataFrame by length in ascending order.

    Args:
        gdf: A GeoDataFrame or GeoSeries.

    Returns:
        A GeoDataFrame or GeoSeries sorted from short to long in length.
    """
    # using enumerate, then iloc on the sorted dict keys.
    # to avoid creating a temporary area column (which doesn't work for GeoSeries).
    length_mapper = dict(enumerate(gdf.length.values))
    sorted_lengths = dict(sorted(length_mapper.items(), key=lambda item: item[1]))
    return gdf.iloc[list(sorted_lengths)]


def sort_small_first(gdf: GeoDataFrame | GeoSeries) -> GeoDataFrame | GeoSeries:
    """Sort GeoDataFrame by area in ascending order.

    Args:
        gdf: A GeoDataFrame or GeoSeries.

    Returns:
        A GeoDataFrame or GeoSeries sorted from small to large in area.

    """
    # using enumerate, then iloc on the sorted dict keys.
    # to avoid creating a temporary area column (which doesn't work for GeoSeries).
    area_mapper = dict(enumerate(gdf.area.values))
    sorted_areas = dict(sorted(area_mapper.items(), key=lambda item: item[1]))
    return gdf.iloc[list(sorted_areas)]


def make_lines_between_points(
    arr1: NDArray[Point] | GeometryArray | GeoSeries,
    arr2: NDArray[Point] | GeometryArray | GeoSeries,
) -> NDArray[LineString]:
    """Creates an array of linestrings from two arrays of points.

    The operation is done rowwise.

    Args:
        arr1: GeometryArray og GeoSeries of points.
        arr2: GeometryArray og GeoSeries of points of same length as arr1.

    Returns:
        A numpy array of linestrings.

    Raises:
        ValueError: If the arrays have unequal shape.

    """
    if arr1.shape != arr2.shape:
        raise ValueError("Arrays must have equal shape.")

    coords: pd.DataFrame = pd.concat(
        [
            pd.DataFrame(get_coordinates(arr1), columns=["x", "y"]),
            pd.DataFrame(get_coordinates(arr2), columns=["x", "y"]),
        ]
    ).sort_index()

    return linestrings(coords.values, indices=coords.index)


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


def random_points_in_polygons(gdf: GeoDataFrame, n: int, seed=None) -> GeoDataFrame:
    all_points = []

    rng = np.random.default_rng(seed)

    for i, geom in enumerate(gdf.geometry):
        minx, miny, maxx, maxy = geom.bounds

        xs = rng.uniform(minx, maxx, size=n * 500)
        ys = rng.uniform(miny, maxy, size=n * 500)

        points = GeoSeries(shapely_points(xs, y=ys), index=[i] * len(xs))
        all_points.append(points)

    return (
        pd.concat(all_points)
        .loc[lambda x: x.intersects(gdf.geometry)]
        .groupby(level=0)
        .head(n)
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

    if not all(isinstance(gdf, (GeoSeries, GeoDataFrame)) for gdf in gdfs):
        raise TypeError("gdf must be GeoDataFrame or GeoSeries")

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

        mapped = gdf.geometry.map(_shapely_geometry_to_lines)
        try:
            gdf.geometry = mapped
        except AttributeError:
            # geoseries
            gdf.loc[:] = mapped

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
    keep_geom_type: bool | None = None,
    geom_type: str | None = None,
    **kwargs,
) -> GeoDataFrame | GeoSeries:
    """Clips and clean geometries.

    Geopandas.clip does a "fast and dirty clipping, with no guarantee for valid
    outputs". Here, the clipped geometries are made valid, and empty and NaN
    geometries are removed.

    Args:
        gdf: GeoDataFrame or GeoSeries to be clipped
        mask: the geometry to clip gdf
        geom_type: Optionally specify what geometry type to keep.,
            if there are mixed geometry types. Must be either "polygon",
            "line" or "point".
        keep_geom_type: Defaults to None, meaning True if 'geom_type' is given
            and True if the geometries are single-typed and False if the geometries
            are mixed.
        **kwargs: Keyword arguments passed to geopandas.GeoDataFrame.clip

    Returns:
        The cleanly clipped GeoDataFrame.

    Raises:
        TypeError: If gdf is not of type GeoDataFrame or GeoSeries.
    """
    if not isinstance(gdf, (GeoDataFrame, GeoSeries)):
        raise TypeError(f"'gdf' should be GeoDataFrame or GeoSeries, got {type(gdf)}")

    gdf, geom_type, keep_geom_type = _determine_geom_type_args(
        gdf, geom_type, keep_geom_type
    )

    try:
        gdf = gdf.clip(mask, **kwargs).pipe(clean_geoms)
    except Exception:
        gdf = clean_geoms(gdf)
        try:
            mask = clean_geoms(mask)
        except TypeError:
            mask = make_valid(mask)

        return gdf.clip(mask, **kwargs).pipe(clean_geoms)

    if keep_geom_type:
        gdf = to_single_geom_type(gdf, geom_type)

    return gdf


def _determine_geom_type_args(
    gdf: GeoDataFrame, geom_type: str | None, keep_geom_type: bool | None
) -> tuple[GeoDataFrame, str, bool]:
    if geom_type:
        gdf = to_single_geom_type(gdf, geom_type)
        keep_geom_type = True
    elif keep_geom_type is None:
        geom_type = get_geom_type(gdf)
        if geom_type == "mixed":
            keep_geom_type = False
        else:
            keep_geom_type = True
    elif keep_geom_type:
        geom_type = get_geom_type(gdf)
        if geom_type == "mixed":
            raise ValueError("Cannot set keep_geom_type=True with mixed geometries")
    return gdf, geom_type, keep_geom_type
