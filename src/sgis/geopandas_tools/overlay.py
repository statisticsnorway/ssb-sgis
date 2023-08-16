"""Overlay function that avoids a GEOSException from geopandas.overlay.

This module includes the function 'clean_overlay', which bypasses a
GEOSException from the regular geopandas.overlay. The function is a generalized
version of the solution from GH 2792.

'clean_overlay' also includes the overlay type "update", which can be specified in the
"how" parameter, in addition to the five native geopandas how-s.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from pandas import DataFrame
from pyproj import CRS
from shapely import STRtree, box, difference, intersection, make_valid, unary_union

from .general import clean_geoms
from .geometry_types import get_geom_type, make_all_singlepart, to_single_geom_type


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
        grid_size: Precision grid size to round the geometries. Will use the highest
            precision of the inputs by default.

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
            raise ValueError("mixed geometries are not allowed.", df1.geometry)

    df1 = clean_geoms(df1)
    df2 = clean_geoms(df2)

    df1 = to_single_geom_type(df1, geom_type)

    df1 = make_all_singlepart(df1, ignore_index=True)
    df2 = make_all_singlepart(df2, ignore_index=True)

    overlayed = _shapely_overlay(df1, df2, how=how, crs=crs, grid_size=grid_size).pipe(
        clean_geoms
    )

    if keep_geom_type:
        overlayed = to_single_geom_type(overlayed, geom_type)

    return overlayed.reset_index(drop=True)


def _join_and_get_no_rows(df1, df2):
    geom_col = df1._geometry_column_name
    df1_cols = df1.columns.difference({geom_col})
    df2_cols = df2.columns.difference({df2._geometry_column_name})
    cols_with_suffix = [f"{col}_1" if col in df2_cols else col for col in df1_cols] + [
        f"{col}_2" if col in df1_cols else col for col in df2_cols
    ]

    return GeoDataFrame(
        pd.DataFrame(columns=cols_with_suffix + [geom_col]),
        geometry=geom_col,
        crs=df1.crs,
    )


def _no_intersections_return(df1, df2, how):
    """Return with no overlay if no intersecting bounding box"""

    if how == "intersection":
        return _join_and_get_no_rows(df1, df2)

    if how == "difference":
        return df1.reset_index(drop=True)

    if how == "identity":
        # add suffixes and return df1
        df_template = _join_and_get_no_rows(df1, df2)
        df2_cols = df2.columns.difference({df2._geometry_column_name})
        df1.columns = [f"{col}_1" if col in df2_cols else col for col in df1]
        return pd.concat([df_template, df1], ignore_index=True)

    if how == "update":
        return pd.concat([df1, df2], ignore_index=True)

    assert how in ["union", "symmetric_difference"]

    # add suffixes and return both concatted
    df_template = _join_and_get_no_rows(df1, df2)
    if not len(df1) and not len(df2):
        return df_template

    df_template = _join_and_get_no_rows(df1, df2)
    return pd.concat([df_template, df1, df2], ignore_index=True)


def _shapely_overlay(
    df1: GeoDataFrame,
    df2: GeoDataFrame,
    how: str,
    crs: int | str | None | CRS,
    grid_size: float,
) -> GeoDataFrame:
    if not len(df1) or not len(df2):
        return _no_intersections_return(df1, df2, how)

    box1 = box(*df1.total_bounds)
    box2 = box(*df2.total_bounds)
    if not len(df1) or not len(df1) or not box1.intersects(box2):
        return _no_intersections_return(df1, df2, how)

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
