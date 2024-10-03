import functools
import itertools
import numbers
import warnings
from collections.abc import Hashable
from collections.abc import Iterable
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyproj
import shapely
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from geopandas.array import GeometryArray
from geopandas.array import GeometryDtype
from numpy.typing import NDArray
from shapely import Geometry
from shapely import extract_unique_points
from shapely import get_coordinates
from shapely import get_exterior_ring
from shapely import get_interior_ring
from shapely import get_num_interior_rings
from shapely import get_parts
from shapely import linestrings
from shapely import make_valid
from shapely import points as shapely_points
from shapely import union_all
from shapely.geometry import LineString
from shapely.geometry import MultiPoint
from shapely.geometry import Point
from shapely.geometry import Polygon

from .conversion import coordinate_array
from .conversion import to_bbox
from .conversion import to_gdf
from .conversion import to_geoseries
from .geometry_types import get_geom_type
from .geometry_types import make_all_singlepart
from .geometry_types import to_single_geom_type


def split_geom_types(gdf: GeoDataFrame | GeoSeries) -> tuple[GeoDataFrame | GeoSeries]:
    return tuple(
        gdf[gdf.geom_type == geom_type] for geom_type in gdf.geom_type.unique()
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
            return next(iter(actually_different))
        raise ValueError("'crs' mismatch.", truthy_crs)

    return pyproj.CRS(truthy_crs[0])


def is_bbox_like(obj: Any) -> bool:
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
    """Removes geometry columns in a GeoDataFrame if they are not active."""
    for col in gdf.columns:
        if (
            isinstance(gdf[col].dtype, GeometryDtype)
            and col != gdf._geometry_column_name
        ):
            gdf = gdf.drop(col, axis=1)
    return gdf


def _rename_geometry_if(gdf: GeoDataFrame) -> GeoDataFrame:
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

    Examples:
    ---------
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
    gdf: GeoDataFrame, groupby: str | list[str], as_string: bool = True
) -> pd.Series:
    """Get the centerpoint of the geometries within a group.

    Args:
        gdf: GeoDataFrame.
        groupby: column to group by.
        as_string: If True (default), coordinates are returned in
            the format "{x}_{y}". If False, coordinates are returned
            as Points.

    Returns:
        A pandas.Series of grouped centroids with the index of 'gdf'.
    """
    centerpoints = gdf.assign(geometry=lambda x: x.centroid)

    grouped_centerpoints = centerpoints.dissolve(by=groupby).assign(
        geometry=lambda x: x.centroid
    )
    xs = grouped_centerpoints.geometry.x
    ys = grouped_centerpoints.geometry.y

    if as_string:
        grouped_centerpoints["wkt"] = [
            f"{int(x)}_{int(y)}" for x, y in zip(xs, ys, strict=False)
        ]
    else:
        grouped_centerpoints["wkt"] = [
            Point(x, y) for x, y in zip(xs, ys, strict=False)
        ]

    return gdf[groupby].map(grouped_centerpoints["wkt"])


def sort_large_first(gdf: GeoDataFrame | GeoSeries) -> GeoDataFrame | GeoSeries:
    """Sort GeoDataFrame by area in decending order.

    Args:
        gdf: A GeoDataFrame or GeoSeries.

    Returns:
        A GeoDataFrame or GeoSeries sorted from large to small in area.

    Examples:
    ---------
    Create GeoDataFrame with NaN values.

    >>> import sgis as sg
    >>> df = sg.to_gdf(
    ...     [
    ...         (0, 1),
    ...         (1, 0),
    ...         (1, 1),
    ...         (0, 0),
    ...         (0.5, 0.5),
    ...     ]
    ... )
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
    *arrs: NDArray[Point] | GeometryArray | GeoSeries,
) -> NDArray[LineString]:
    """Creates an array of linestrings from two or more arrays of points.

    The lines are created rowwise, meaning from arr0[0] to arr1[0], from arr0[1] to arr1[1]...
    If more than two arrays are passed, e.g. three arrays,
    the lines will go from arr0[0] via arr1[0] to arr2[0].

    Args:
        arrs: 1 dimensional arrays of point geometries.
            All arrays must have the same shape.
            Must be at least two arrays.

    Returns:
        A numpy array of linestrings.

    """
    coords = [get_coordinates(arr, return_index=False) for arr in arrs]
    return linestrings(
        np.concatenate([coords_arr[:, None, :] for coords_arr in coords], axis=1)
    )


def random_points(n: int, loc: float | int = 0.5) -> GeoDataFrame:
    """Creates a GeoDataFrame with n random points.

    Args:
        n: Number of points/rows to create.
        loc: Mean ('centre') of the distribution.

    Returns:
        A GeoDataFrame of points with n rows.

    Examples:
    ---------
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
    """Creates a GeoDataFrame with n random points within the geometries of 'gdf'.

    Args:
        gdf: A GeoDataFrame.
        n: Number of points/rows to create.
        seed: Optional random seet.

    Returns:
        A GeoDataFrame of points with n rows.
    """
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

    Examples:
    ---------
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
        raise ValueError(
            f"Cannot convert points to lines. {[gdf.geom_type.value_counts() for gdf in gdfs]}"
        )

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

        return union_all(lines)

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


def extend_lines(arr1, arr2, distance) -> NDArray[LineString]:
    if len(arr1) != len(arr2):
        raise ValueError
    if not len(arr1):
        return arr1

    arr1, arr2 = arr2, arr1  # TODO fix

    coords1 = coordinate_array(arr1)
    coords2 = coordinate_array(arr2)

    dx = coords2[:, 0] - coords1[:, 0]
    dy = coords2[:, 1] - coords1[:, 1]
    len_xy = np.sqrt((dx**2.0) + (dy**2.0))
    x = coords1[:, 0] + (coords1[:, 0] - coords2[:, 0]) / len_xy * distance
    y = coords1[:, 1] + (coords1[:, 1] - coords2[:, 1]) / len_xy * distance

    new_points = np.array([None for _ in range(len(arr1))])
    new_points[~np.isnan(x)] = shapely.points(x[~np.isnan(x)], y[~np.isnan(x)])

    new_points[~np.isnan(x)] = make_lines_between_points(
        arr2[~np.isnan(x)], new_points[~np.isnan(x)]
    )
    return new_points


def multipoints_to_line_segments_numpy(
    points: GeoSeries | NDArray[MultiPoint] | MultiPoint,
    cycle: bool = False,
) -> list[LineString]:
    try:
        arr = get_parts(points.geometry.values)
    except AttributeError:
        arr = get_parts(points)

    line_between_last_and_first = [LineString([arr[-1], arr[0]])] if cycle else []
    return [
        LineString([p0, p1]) for p0, p1 in itertools.pairwise(arr)
    ] + line_between_last_and_first


def multipoints_to_line_segments(
    multipoints: GeoSeries | GeoDataFrame, cycle: bool = True  # to_next: bool = True,
) -> GeoSeries | GeoDataFrame:

    if not len(multipoints):
        return multipoints

    if isinstance(multipoints, GeoDataFrame):
        df = multipoints.drop(columns=multipoints.geometry.name)
        multipoints = multipoints.geometry
        was_gdf = True
    else:
        multipoints = to_geoseries(multipoints)
        was_gdf = False

    multipoints = to_geoseries(multipoints)

    segs = pd.Series(
        [
            multipoints_to_line_segments_numpy(geoms, cycle=cycle)
            for geoms in multipoints
        ],
        index=multipoints.index,
    ).explode()

    segs = GeoSeries(segs, crs=multipoints.crs, name=multipoints.name)

    if was_gdf:
        return GeoDataFrame(df.join(segs), geometry=segs.name, crs=segs.crs)
    else:
        return segs


def get_line_segments(
    lines: GeoDataFrame | GeoSeries, extract_unique: bool = False, cycle=False
) -> GeoDataFrame:
    try:
        assert lines.index.is_unique
    except AttributeError:
        pass

    if isinstance(lines, GeoDataFrame):
        df = lines.drop(columns=lines.geometry.name)
        lines = lines.geometry
        was_gdf = True
    else:
        lines = to_geoseries(lines)
        was_gdf = False

    partial_segs_func = functools.partial(
        multipoints_to_line_segments_numpy, cycle=cycle
    )
    if extract_unique:
        points = extract_unique_points(lines.geometry.values)
        segs = pd.Series(
            [partial_segs_func(geoms) for geoms in points],
            index=lines.index,
        ).explode()
    else:
        coords, indices = shapely.get_coordinates(lines, return_index=True)
        points = GeoSeries(shapely.points(coords), index=indices)
        index_mapper = {
            i: idx
            for i, idx in zip(
                np.unique(indices), lines.index.drop_duplicates(), strict=True
            )
        }
        points.index = points.index.map(index_mapper)

        segs = points.groupby(level=0).agg(partial_segs_func).explode()
    segs = GeoSeries(segs, crs=lines.crs, name=lines.name)

    if was_gdf:
        return GeoDataFrame(df.join(segs), geometry=segs.name, crs=lines.crs)
    else:
        return segs


def get_index_right_columns(gdf: pd.DataFrame | pd.Series) -> list[str]:
    """Get a list of what will be the resulting columns in an sjoin."""
    if gdf.index.name is None and all(name is None for name in gdf.index.names):
        if gdf.index.nlevels == 1:
            return ["index_right"]
        else:
            return [f"index_right{i}" for i in range(gdf.index.nlevels)]
    else:
        return gdf.index.names


def points_in_bounds(
    gdf: GeoDataFrame | GeoSeries, gridsize: int | float
) -> GeoDataFrame:
    """Get a GeoDataFrame of points within the bounds of the GeoDataFrame."""
    minx, miny, maxx, maxy = to_bbox(gdf)
    try:
        crs = gdf.crs
    except AttributeError:
        crs = None

    xs = np.linspace(minx, maxx, num=int((maxx - minx) / gridsize))
    ys = np.linspace(miny, maxy, num=int((maxy - miny) / gridsize))
    x_coords, y_coords = np.meshgrid(xs, ys, indexing="ij")
    coords = np.concatenate((x_coords.reshape(-1, 1), y_coords.reshape(-1, 1)), axis=1)
    return to_gdf(coords, crs=crs)


def points_in_polygons(
    gdf: GeoDataFrame | GeoSeries, gridsize: int | float
) -> GeoDataFrame:
    index_right_col = get_index_right_columns(gdf)
    out = points_in_bounds(gdf, gridsize).sjoin(gdf).set_index(index_right_col)
    out.index.name = gdf.index.name
    return out.sort_index()


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


def _unary_union_for_notna(geoms, **kwargs):
    try:
        return make_valid(union_all(geoms, **kwargs))
    except TypeError:
        return union_all([geom for geom in geoms.dropna().values], **kwargs)


def _grouped_unary_union(
    df: GeoDataFrame | GeoSeries | pd.DataFrame | pd.Series,
    by: str | list[str] | None = None,
    level: int | None = None,
    as_index: bool = True,
    grid_size: float | int | None = None,
    dropna: bool = False,
    **kwargs,
) -> GeoSeries | GeoDataFrame:
    """Vectorized unary_union for groups.

    Experimental. Messy code.
    """
    df = df.copy()
    df_orig = df.copy()

    try:
        geom_col = df._geometry_column_name
    except AttributeError:
        try:
            geom_col = df.name
            if geom_col is None:
                geom_col = "geometry"
        except AttributeError:
            geom_col = "geometry"

    if not len(df):
        return GeoSeries(name=geom_col)

    if isinstance(df, pd.Series):
        df.name = geom_col
        original_index = df.index
        df = df.reset_index()
        df.index = original_index

    if isinstance(by, str):
        by = [by]
    elif by is None and level is None:
        raise TypeError("You have to supply one of 'by' and 'level'")
    elif by is None:
        by = df.index.get_level_values(level)

    cumcount = df.groupby(by, dropna=dropna).cumcount().values

    def get_col_or_index(df, col: str) -> pd.Series | pd.Index:
        try:
            return df[col]
        except KeyError:
            for i, name in enumerate(df.index.names):
                if name == col:
                    return df.index.get_level_values(i)
        raise KeyError(col)

    try:
        df.index = pd.MultiIndex.from_arrays(
            [cumcount, *[get_col_or_index(df, col) for col in by]]
        )
    except KeyError:
        df.index = pd.MultiIndex.from_arrays([cumcount, by])

    # to wide format: each row will be one group to be merged to one geometry
    try:
        geoms_wide: pd.DataFrame = df[geom_col].unstack(level=0)
    except Exception as e:
        bb = [*by, geom_col]
        raise e.__class__(e, f"by={by}", df_orig[bb], df[geom_col]) from e
    geometries_2d: NDArray[Polygon | None] = geoms_wide.values
    try:
        geometries_2d = make_valid(geometries_2d)
    except TypeError:
        # make_valid doesn't like nan, so converting to None
        # np.isnan doesn't accept geometry type, so using isinstance
        np_isinstance = np.vectorize(isinstance)
        geometries_2d[np_isinstance(geometries_2d, Geometry) == False] = None

    unioned = make_valid(union_all(geometries_2d, axis=1, **kwargs))

    geoms = GeoSeries(unioned, name=geom_col, index=geoms_wide.index)

    return geoms if as_index else geoms.reset_index()


def _parallel_unary_union(
    gdf: GeoDataFrame, n_jobs: int = 1, by=None, grid_size=None, **kwargs
) -> list[Geometry]:
    try:
        geom_col = gdf._geometry_column_name
    except AttributeError:
        geom_col = "geometry"

    with joblib.Parallel(n_jobs=n_jobs, backend="threading") as parallel:
        delayed_operations = []
        for _, geoms in gdf.groupby(by, **kwargs)[geom_col]:
            delayed_operations.append(
                joblib.delayed(_unary_union_for_notna)(geoms, grid_size=grid_size)
            )

        return parallel(delayed_operations)


def _parallel_unary_union_geoseries(
    ser: GeoSeries, n_jobs: int = 1, grid_size=None, **kwargs
) -> list[Geometry]:

    with joblib.Parallel(n_jobs=n_jobs, backend="threading") as parallel:
        delayed_operations = []
        for _, geoms in ser.groupby(**kwargs):
            delayed_operations.append(
                joblib.delayed(_unary_union_for_notna)(geoms, grid_size=grid_size)
            )

        return parallel(delayed_operations)
