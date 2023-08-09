"""Overlay function that avoids a GEOSException from geopandas.overlay.

This module includes the function 'clean_overlay', which bypasses a
GEOSException from the regular geopandas.overlay. The function is a generalized
version of the solution from GH 2792.

'clean_overlay' also includes the overlay type "update", which can be specified in the
"how" parameter, in addition to the five native geopandas how-s.
"""
from typing import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from pandas import DataFrame
from pyproj import CRS
from shapely import STRtree, difference, intersection, make_valid, unary_union, union
from shapely.errors import GEOSException
from shapely.geometry import Polygon

from .general import clean_geoms
from .geometry_types import get_geom_type, make_all_singlepart, to_single_geom_type


def update_geometries(
    gdf: GeoDataFrame, keep_geom_type: bool = True, copy: bool = True
) -> GeoDataFrame:
    """Puts geometries on top of each other rowwise.

    Since this operation is done rowwise, it's important to
    first sort the GeoDataFrame approriately. See example below.

    Example
    ------
    Create two circles and get the overlap.

    >>> import sgis as sg
    >>> circles = sg.to_gdf([(0, 0), (1, 1)]).pipe(sg.buff, 1)
    >>> duplicates = sg.get_intersections(circles)
    >>> duplicates
       idx                                           geometry
    0    1  POLYGON ((0.03141 0.99951, 0.06279 0.99803, 0....
    1    2  POLYGON ((1.00000 0.00000, 0.96859 0.00049, 0....

    The polygons are identical except for the order of the coordinates.

    >>> poly1, poly2 = duplicates.geometry
    >>> poly1.equals(poly2)
    True

    'update_geometries' gives different results based on the order
    of the GeoDataFrame.

    >>> sg.update_geometries(duplicates)
        idx                                           geometry
    0    1  POLYGON ((0.03141 0.99951, 0.06279 0.99803, 0....

    >>> dups_rev = duplicates.iloc[::-1]
    >>> sg.update_geometries(dups_rev)
        idx                                           geometry
    1    2  POLYGON ((1.00000 0.00000, 0.96859 0.00049, 0....

    It might be appropriate to put the largest polygons on top
    and sort all NaNs to the bottom.

    >>> updated = (
    ...     sg.sort_large_first(duplicates)
    ...     .pipe(sg.sort_nans_last)
    ...     .pipe(sg.update_geometries)
    >>> updated
        idx                                           geometry
    0    1  POLYGON ((0.03141 0.99951, 0.06279 0.99803, 0....

    """
    if len(gdf) <= 1:
        return gdf

    df = pd.DataFrame(gdf, copy=copy)

    unioned = Polygon()
    out_rows, indices, geometries = [], [], []

    if keep_geom_type:
        geom_type = get_geom_type(gdf)
        if geom_type == "mixed":
            raise ValueError("Cannot have mixed geometries when keep_geom_type is True")

    for i, row in df.iterrows():
        geom = row.pop("geometry")

        if any(geom.equals(geom2) for geom2 in geometries):
            continue

        new = try_shapely_func_pair(geom, unioned, func=difference)

        if not new:
            continue

        unioned = try_shapely_func_pair(new, unioned, func=union)

        out_rows.append(row)
        geometries.append(new)
        indices.append(i)

    out = GeoDataFrame(out_rows, geometry=geometries, index=indices, crs=gdf.crs)

    if keep_geom_type:
        out = to_single_geom_type(out, geom_type)

    return out


def try_shapely_func_pair(geom1, geom2, func: Callable):
    try:
        return func(geom1, geom2)
    except GEOSException:
        try:
            geom1 = make_valid(geom1)
            return func(geom1, geom2)
        except GEOSException:
            try:
                return func(geom1, geom2, grid_size=0.01)
            except GEOSException as e:
                raise ValueError(geom1, geom2) from e


def get_intersections(gdf: GeoDataFrame) -> GeoDataFrame:
    """Find geometries that intersect in a GeoDataFrame.

    Does an intersection with itself and keeps only the geometries that appear
    more than once. This means each intersection gives at least two rows. The
    duplicates should then be handled appropriately. See example
    below.

    Args:
        gdf: GeoDataFrame of polygons.

    Returns:
        A GeoDataFrame of the overlapping polygons.

    Examples
    --------
    Create two fully overlapping polygons.

    >>> import sgis as sg
    >>> circles = sg.to_gdf([(0, 0), (0, 0)])
    >>> circles["geometry"] = circles["geometry"].buffer([1, 2])
    >>> circles["idx"] = [1, 3]
    >>> circles.area
    0     3.141076
    1    12.564304
    dtype: float64

    Get the duplicates.

    >>> duplicates = sg.get_intersections(circles)
    >>> duplicates
       idx                                           geometry
    0    1  POLYGON ((0.99951 -0.03141, 0.99803 -0.06279, ...
    1    3  POLYGON ((0.99951 -0.03141, 0.99803 -0.06279, ...
    >>> duplicates.area
    0    3.141076
    1    3.141076
    dtype: float64

    We get two rows for each intersection pair.

    The function sgis.update_geometries can be used to put geometries
    on top of the other rowwise.

    >>> updated = sg.update_geometries(duplicates)
    >>> updated
        idx                                           geometry
    0    1  POLYGON ((0.99518 -0.09802, 0.98079 -0.19509, ...

    Reversing the rows means the bottom polygon is put on top.

    >>> updated = sg.update_geometries(duplicates.iloc[::-1]))
    >>> updated
        idx                                           geometry
    1    3  POLYGON ((0.99518 -0.09802, 0.98079 -0.19509, ...

    It might be appropriate to sort the dataframe by columns.
    Or put large polygons first and NaN values last.

    >>> updated = (
    ...     sg.sort_large_first(duplicates)
    ...     .pipe(sg.sort_nans_last)
    ...     .pipe(sg.update_geometries)
    ... )
    >>> updated
       idx                                           geometry
    0    1  POLYGON ((0.99518 -0.09802, 0.98079 -0.19509, ...
    """

    idx_name = gdf.index.name
    duplicated_geoms = _get_intersecting_geometries(gdf).pipe(clean_geoms)

    duplicated_geoms.index = duplicated_geoms["orig_idx"].values
    duplicated_geoms.index.name = idx_name
    return duplicated_geoms.drop(columns="orig_idx")


def _get_intersecting_geometries(gdf: GeoDataFrame) -> GeoDataFrame:
    gdf = gdf.assign(orig_idx=gdf.index).reset_index(drop=True)

    right = gdf[[gdf._geometry_column_name]]
    right["idx_right"] = right.index
    left = gdf
    left["idx_left"] = left.index

    intersected = clean_overlay(left, right, how="intersection")

    # these are identical as the input geometries
    not_from_same_poly = intersected.loc[lambda x: x["idx_left"] != x["idx_right"]]

    points_joined = (
        not_from_same_poly.representative_point().to_frame().sjoin(not_from_same_poly)
    )

    duplicated_points = points_joined.loc[points_joined.index.duplicated(keep=False)]

    return intersected.loc[intersected.index.isin(duplicated_points.index)].drop(
        columns=["idx_left", "idx_right"]
    )


def clean_overlay(
    df1: GeoDataFrame,
    df2: GeoDataFrame,
    how: str = "intersection",
    keep_geom_type: bool = True,
    geom_type: str | None = None,
    grid_size: float | None = None,
) -> GeoDataFrame:
    """Fixes and explodes geometries before doing a shapely overlay, then cleans up.

    Fixes geometries, then does a shapely overlay operation that dodges a GEOSException
    raised in the regular geopandas.overlay. The function is a generalised version of a
    solution from GH 2792.

    Args:
        df1: GeoDataFrame
        df2: GeoDataFrame
        how: Method of spatial overlay. Includes the 'update' method, plus the five
            native geopandas methods 'intersection', 'union', 'identity',
            'symmetric_difference' and 'difference'.
        keep_geom_type: If True (default), return only geometries of the same
            geometry type as df1 has, if False, return all resulting geometries.
        geom_type: optionally specify what geometry type to keep before the overlay,
            if there are mixed geometry types. Must be either "polygon", "line" or
            "point".

    Returns:
        GeoDataFrame with overlayed and fixed geometries and columns from both
        GeoDataFrames.

    Raises:
        ValueError: If 'how' is not one of 'intersection', 'union', 'identity',
            'symmetric_difference', 'difference' or 'update'.
    """
    # Allowed operations
    allowed_hows = [
        "intersection",
        "union",
        "identity",
        "symmetric_difference",
        "difference",
        "update",
    ]
    if how not in allowed_hows:
        raise ValueError(
            f"`how` was {how!r} but is expected to be in {', '.join(allowed_hows)}"
        )

    if df1.crs != df2.crs:
        raise ValueError(f"'crs' mismatch. Got {df1.crs} and {df2.crs}")

    crs = df1.crs

    if not geom_type:
        geom_type = get_geom_type(df1)
        if geom_type == "mixed":
            raise ValueError("mixed geometries are not allowed.")

    df1 = clean_geoms(df1)
    df2 = clean_geoms(df2)

    # df1.geometry = df1.geometry.buffer(0)
    # df2.geometry = df2.geometry.buffer(0)

    if geom_type:
        df1 = to_single_geom_type(df1, geom_type)

    df1 = to_single_geom_type(df1, geom_type)

    if keep_geom_type:
        df2 = to_single_geom_type(df2, geom_type)

    df1 = make_all_singlepart(df1, ignore_index=True)
    df2 = make_all_singlepart(df2, ignore_index=True)

    overlayed = _shapely_overlay(df1, df2, how=how, crs=crs, grid_size=grid_size).pipe(
        clean_geoms
    )

    # overlayed.geometry = overlayed.geometry.buffer(0)

    if geom_type:
        overlayed = to_single_geom_type(overlayed, geom_type)

    return overlayed.reset_index(drop=True)


def _shapely_overlay(
    df1: GeoDataFrame,
    df2: GeoDataFrame,
    how: str,
    crs: int | str | None | CRS,
    grid_size: float,
) -> GeoDataFrame:
    if df1._geometry_column_name != "geometry":
        df1 = df1.rename_geometry("geometry")

    if df2._geometry_column_name != "geometry":
        df2 = df2.rename_geometry("geometry")

    tree = STRtree(df2.geometry.values)
    left, right = tree.query(df1.geometry.values, predicate="intersects")
    # GeoDataFrame constructor is expensive, so doing it only once in the end
    df1 = DataFrame(df1)
    df2 = DataFrame(df2)

    pairs = _get_intersects_pairs(df1, df2, left, right)

    if how == "intersection":
        overlayed = [_intersection(pairs, grid_size=grid_size)]

    elif how == "difference":
        overlayed = _difference(pairs, df1, left, grid_size=grid_size)

    elif how == "symmetric_difference":
        overlayed = _symmetric_difference(
            pairs, df1, df2, left, right, grid_size=grid_size
        )

    elif how == "identity":
        overlayed = _identity(pairs, df1, left, grid_size=grid_size)

    elif how == "union":
        overlayed = _union(pairs, df1, df2, left, right, grid_size=grid_size)

    elif how == "update":
        overlayed = _update(pairs, df1, df2, left=left, grid_size=grid_size)

    assert isinstance(overlayed, list)

    overlayed = pd.concat(overlayed, ignore_index=True).drop(
        columns="index_right", errors="ignore"
    )

    # push geometry column to the end
    overlayed = overlayed.reindex(
        columns=[c for c in overlayed.columns if c != "geometry"] + ["geometry"]
    )

    if how not in ["difference", "update"]:
        overlayed = _add_suffix_left(overlayed, df1, df2)

    overlayed["geometry"] = make_valid(overlayed["geometry"])
    # None and empty are falsy
    overlayed = overlayed.loc[lambda x: x["geometry"].map(bool)]

    return gpd.GeoDataFrame(overlayed, geometry="geometry", crs=crs)


def _update(pairs, df1, df2, left, grid_size) -> GeoDataFrame:
    overlayed = _difference(pairs, df1, left, grid_size=grid_size)

    return overlayed + [df2]


def _intersection(pairs, grid_size) -> GeoDataFrame:
    if not len(pairs):
        return pairs.drop(columns="geom_right")

    intersections = pairs.copy()
    intersections["geometry"] = intersection(
        intersections["geometry"].to_numpy(),
        intersections["geom_right"].to_numpy(),
        grid_size=grid_size,
    )

    return intersections.drop(columns="geom_right")


def _union(pairs, df1, df2, left, right, grid_size):
    merged = []
    if len(left):
        intersections = _intersection(pairs, grid_size=grid_size)
        merged.append(intersections)
    symmdiff = _symmetric_difference(pairs, df1, df2, left, right, grid_size=grid_size)
    merged += symmdiff
    return merged


def _identity(pairs, df1, left, grid_size):
    merged = []
    if len(left):
        intersections = _intersection(pairs, grid_size=grid_size)
        merged.append(intersections)
    diff = _difference(pairs, df1, left, grid_size=grid_size)
    merged += diff
    return merged


def _symmetric_difference(pairs, df1, df2, left, right, grid_size) -> list:
    merged = []

    difference_left = _difference(pairs, df1, left, grid_size=grid_size)
    merged += difference_left

    if len(left):
        clip_right = _shapely_diffclip_right(pairs, df1, df2, grid_size=grid_size)
        merged.append(clip_right)

    diff_right = _add_from_right(df1, df2, right)
    merged.append(diff_right)

    return merged


def _difference(pairs, df1, left, grid_size=None) -> list:
    merged = []
    if len(left):
        clip_left = _shapely_diffclip_left(pairs, df1, grid_size=grid_size)
        merged.append(clip_left)
    diff_left = _add_from_left(df1, left)
    merged.append(diff_left)
    return merged


def _get_intersects_pairs(
    df1: GeoDataFrame, df2: GeoDataFrame, left: np.ndarray, right: np.ndarray
) -> DataFrame:
    return pd.concat(
        [
            df1.take(left),
            (DataFrame({"index_right": right}, index=df1.index.values.take(left))),
        ],
        axis=1,
    ).join(
        df2.rename(columns={"geometry": "geom_right"}, errors="raise"),
        on="index_right",
        rsuffix="_2",
    )


def _add_suffix_left(overlayed, df1, df2):
    """Separating this from _add_from_left, since this suffix is not needed in difference."""
    return overlayed.rename(
        columns={
            c: f"{c}_1"
            if c in df1.columns and c in df2.columns and c != "geometry"
            else c
            for c in overlayed.columns
        }
    )


def _add_from_left(df1, left):
    return df1.take(np.setdiff1d(np.arange(len(df1)), left))


def _add_from_right(
    df1: GeoDataFrame, df2: GeoDataFrame, right: np.ndarray
) -> GeoDataFrame:
    return df2.take(np.setdiff1d(np.arange(len(df2)), right)).rename(
        columns={
            c: f"{c}_2" if c in df1.columns and c != "geometry" else c
            for c in df2.columns
        }
    )


def _shapely_diffclip_left(pairs, df1, grid_size):
    """Aggregate areas in right by unique values of left, then use those to clip
    areas out of left"""
    clip_left = pairs.groupby(level=0).agg(
        {
            "geom_right": lambda g: unary_union(g) if len(g) > 1 else g,
            **{
                c: "first"
                for c in df1.columns
                if c not in ["index_right", "geom_right"]
            },
        }
    )
    clip_left["geometry"] = difference(
        clip_left["geometry"].to_numpy(),
        clip_left["geom_right"].to_numpy(),
        grid_size=grid_size,
    )

    return clip_left.drop(columns="geom_right")


def _shapely_diffclip_right(pairs, df1, df2, grid_size):
    clip_right = (
        pairs.rename(columns={"geometry": "geom_left", "geom_right": "geometry"})
        .groupby(by="index_right")
        .agg(
            {
                "geom_left": lambda g: unary_union(g) if len(g) > 1 else g,
                "geometry": "first",
            }
        )
        .join(df2.drop(columns=["geometry"]))
        .rename(
            columns={
                c: f"{c}_2" if c in df1.columns and c != "geometry" else c
                for c in df2.columns
            }
        )
    )

    clip_right["geometry"] = difference(
        clip_right["geometry"].to_numpy(),
        clip_right["geom_left"].to_numpy(),
        grid_size=grid_size,
    )

    return clip_right.drop(columns="geom_left")
