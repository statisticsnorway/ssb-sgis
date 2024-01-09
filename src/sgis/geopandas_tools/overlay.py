"""Overlay function that avoids a GEOSException from geopandas.overlay.

This module includes the function 'clean_overlay', which bypasses a
GEOSException from the regular geopandas.overlay. The function is a generalized
version of the solution from GH 2792.

'clean_overlay' also includes the overlay type "update", which can be specified in the
"how" parameter, in addition to the five native geopandas how-s.
"""
import functools

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame
from pyproj import CRS
from shapely import (
    STRtree,
    box,
    difference,
    intersection,
    is_valid,
    make_valid,
    unary_union,
)
from shapely.errors import GEOSException

from .general import clean_geoms
from .geometry_types import get_geom_type, make_all_singlepart, to_single_geom_type


DEFAULT_GRID_SIZE = None
DEFAULT_LSUFFIX = "_1"
DEFAULT_RSUFFIX = "_2"


def clean_overlay(
    df1: GeoDataFrame,
    df2: GeoDataFrame,
    how: str = "intersection",
    keep_geom_type: bool = True,
    geom_type: str | None = None,
    grid_size: float | None = None,
    lsuffix: str = DEFAULT_LSUFFIX,
    rsuffix: str = DEFAULT_RSUFFIX,
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

    original_geom_type = geom_type

    if not geom_type:
        geom_type = get_geom_type(df1)
        if geom_type == "mixed":
            raise ValueError(
                "mixed geometries are not allowed when geom_type isn't specified.",
                df1.geometry,
            )
        if get_geom_type(df2) == "mixed":
            raise ValueError(
                "mixed geometries are not allowed when geom_type isn't specified.",
                df2.geometry,
            )

    df1 = clean_geoms(df1)
    df2 = clean_geoms(df2)

    df1 = make_all_singlepart(df1, ignore_index=True)
    df2 = make_all_singlepart(df2, ignore_index=True)

    df1 = to_single_geom_type(df1, geom_type)

    if original_geom_type:
        df2 = to_single_geom_type(df2, geom_type)

    assert df1.is_valid.all()
    assert df2.is_valid.all()
    assert df1.geometry.notna().all()
    assert df2.geometry.notna().all()

    box1 = box(*df1.total_bounds)
    box2 = box(*df2.total_bounds)

    if not len(df1) or not len(df1) or not box1.intersects(box2):
        return _no_intersections_return(df1, df2, how, lsuffix, rsuffix)

    if df1._geometry_column_name != "geometry":
        df1 = df1.rename_geometry("geometry")

    if df2._geometry_column_name != "geometry":
        df2 = df2.rename_geometry("geometry")

    # to pandas because GeoDataFrame constructor is expensive
    df1 = DataFrame(df1).reset_index(drop=True)
    df2 = DataFrame(df2).reset_index(drop=True)

    overlayed = (
        gpd.GeoDataFrame(
            _shapely_pd_overlay(
                df1,
                df2,
                how=how,
                grid_size=grid_size,
                lsuffix=lsuffix,
                rsuffix=rsuffix,
            ),
            geometry="geometry",
            crs=crs,
        )
        .pipe(clean_geoms)
        .pipe(make_all_singlepart, ignore_index=True)
    )

    if keep_geom_type:
        overlayed = to_single_geom_type(overlayed, geom_type)

    return overlayed.reset_index(drop=True)


def _join_and_get_no_rows(df1, df2, lsuffix, rsuffix):
    geom_col = df1._geometry_column_name
    df1_cols = df1.columns.difference({geom_col})
    df2_cols = df2.columns.difference({df2._geometry_column_name})
    cols_with_suffix = [
        f"{col}{lsuffix}" if col in df2_cols else col for col in df1_cols
    ] + [f"{col}{rsuffix}" if col in df1_cols else col for col in df2_cols]

    return GeoDataFrame(
        pd.DataFrame(columns=cols_with_suffix + [geom_col]),
        geometry=geom_col,
        crs=df1.crs,
    )


def _no_intersections_return(df1, df2, how, lsuffix, rsuffix):
    """Return with no overlay if no intersecting bounding box"""

    if how == "intersection":
        return _join_and_get_no_rows(df1, df2, lsuffix, rsuffix)

    if how == "difference":
        return df1.reset_index(drop=True)

    if how == "identity":
        # add suffixes and return df1
        df_template = _join_and_get_no_rows(df1, df2, lsuffix, rsuffix)
        df2_cols = df2.columns.difference({df2._geometry_column_name})
        df1.columns = [f"{col}{lsuffix}" if col in df2_cols else col for col in df1]
        return pd.concat([df_template, df1], ignore_index=True)

    if how == "update":
        return pd.concat([df1, df2], ignore_index=True)

    assert how in ["union", "symmetric_difference"]

    # add suffixes and return both concatted
    df_template = _join_and_get_no_rows(df1, df2, lsuffix, rsuffix)
    if not len(df1) and not len(df2):
        return df_template

    df_template = _join_and_get_no_rows(df1, df2, lsuffix, rsuffix)
    return pd.concat([df_template, df1, df2], ignore_index=True)


def _shapely_pd_overlay(
    df1: DataFrame,
    df2: DataFrame,
    how: str,
    grid_size: float = DEFAULT_GRID_SIZE,
    lsuffix=DEFAULT_LSUFFIX,
    rsuffix=DEFAULT_RSUFFIX,
) -> DataFrame:
    if not grid_size and not len(df1) or not len(df2):
        return _no_intersections_return(df1, df2, how, lsuffix, rsuffix)

    tree = STRtree(df2.geometry.values)
    left, right = tree.query(df1.geometry.values, predicate="intersects")

    pairs = _get_intersects_pairs(df1, df2, left, right, rsuffix)
    assert pairs.geometry.notna().all()
    assert pairs.geom_right.notna().all()

    if how == "intersection":
        overlayed = [_intersection(pairs, grid_size=grid_size)]

    elif how == "difference":
        overlayed = _difference(pairs, df1, left, grid_size=grid_size)

    elif how == "symmetric_difference":
        overlayed = _symmetric_difference(
            pairs, df1, df2, left, right, grid_size=grid_size, rsuffix=rsuffix
        )

    elif how == "identity":
        overlayed = _identity(pairs, df1, left, grid_size=grid_size)

    elif how == "union":
        overlayed = _union(
            pairs, df1, df2, left, right, grid_size=grid_size, rsuffix=rsuffix
        )

    elif how == "update":
        overlayed = _update(pairs, df1, df2, left=left, grid_size=grid_size)

    assert isinstance(overlayed, list)

    overlayed = pd.concat(overlayed, ignore_index=True).drop(
        columns="_overlay_index_right", errors="ignore"
    )

    # push geometry column to the end
    overlayed = overlayed.reindex(
        columns=[c for c in overlayed.columns if c != "geometry"] + ["geometry"]
    )

    if how not in ["difference", "update"]:
        overlayed = _add_suffix_left(overlayed, df1, df2, lsuffix)

    overlayed["geometry"] = make_valid(overlayed["geometry"])
    # None and empty are falsy
    overlayed = overlayed.loc[lambda x: x["geometry"].notna()]

    return overlayed


def _update(pairs, df1, df2, left, grid_size) -> GeoDataFrame:
    overlayed = _difference(pairs, df1, left, grid_size=grid_size)

    return overlayed + [df2]


def _intersection(pairs, grid_size) -> GeoDataFrame:
    if not len(pairs):
        return pairs.drop(columns="geom_right")

    # assert all(is_valid(pairs["geometry"].to_numpy())), "\nnot all valid\n"
    # assert all(is_valid(pairs["geom_right"].to_numpy())), "\nnot all valid\n"

    intersections = pairs.copy()
    """
    geoms_left = make_valid(intersections["geometry"].to_numpy())
    geoms_right = make_valid(intersections["geom_right"].to_numpy())
    intersections["geometry"] = intersection(
        geoms_left,
        geoms_right,
        grid_size=grid_size,
    )
    try:
        intersections["geometry"] = intersection(
            intersections["geometry"].to_numpy(),
            intersections["geom_right"].to_numpy(),
            grid_size=grid_size,
        )
    except GEOSException:"""
    intersections["geometry"] = intersection(
        intersections["geometry"].to_numpy(),
        intersections["geom_right"].to_numpy(),
        grid_size=grid_size,
    )

    return intersections.drop(columns="geom_right")


def _union(pairs, df1, df2, left, right, grid_size, rsuffix):
    merged = []
    if len(left):
        intersections = _intersection(pairs, grid_size=grid_size)
        merged.append(intersections)
    symmdiff = _symmetric_difference(
        pairs, df1, df2, left, right, grid_size=grid_size, rsuffix=rsuffix
    )
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


def _symmetric_difference(pairs, df1, df2, left, right, grid_size, rsuffix) -> list:
    merged = []

    difference_left = _difference(pairs, df1, left, grid_size=grid_size)
    merged += difference_left

    if len(left):
        clip_right = _shapely_diffclip_right(
            pairs, df1, df2, grid_size=grid_size, rsuffix=rsuffix
        )
        merged.append(clip_right)

    diff_right = _add_from_right(df1, df2, right, rsuffix)
    merged.append(diff_right)

    return merged


def _difference(pairs, df1, left, grid_size=None) -> list:
    merged = []
    if len(left):
        clip_left = _shapely_diffclip_left(pairs=pairs, df1=df1, grid_size=grid_size)
        merged.append(clip_left)
    diff_left = _add_indices_from_left(df1, left)
    merged.append(diff_left)
    return merged


def _get_intersects_pairs(
    df1: GeoDataFrame,
    df2: GeoDataFrame,
    left: np.ndarray,
    right: np.ndarray,
    rsuffix,
) -> DataFrame:
    return pd.concat(
        [
            df1.take(left),
            (
                DataFrame(
                    {"_overlay_index_right": right}, index=df1.index.values.take(left)
                )
            ),
        ],
        axis=1,
    ).join(
        df2.rename(columns={"geometry": "geom_right"}, errors="raise"),
        on="_overlay_index_right",
        rsuffix=rsuffix,
    )


def _add_suffix_left(overlayed, df1, df2, lsuffix):
    """Separating this from _add_indices_from_left, since this suffix is not needed in difference."""
    return overlayed.rename(
        columns={
            c: f"{c}{lsuffix}"
            if c in df1.columns and c in df2.columns and c != "geometry"
            else c
            for c in overlayed.columns
        }
    )


def _add_indices_from_left(df1, left):
    return df1.take(np.setdiff1d(np.arange(len(df1)), left))


def _add_from_right(
    df1: GeoDataFrame, df2: GeoDataFrame, right: np.ndarray, rsuffix
) -> GeoDataFrame:
    return df2.take(np.setdiff1d(np.arange(len(df2)), right)).rename(
        columns={
            c: f"{c}{rsuffix}" if c in df1.columns and c != "geometry" else c
            for c in df2.columns
        }
    )


def _shapely_diffclip_left(pairs, df1, grid_size):
    """Aggregate areas in right by unique values of left, then use those to clip
    areas out of left"""

    agg_geoms_partial = functools.partial(agg_geoms, grid_size=grid_size)

    clip_left = pairs.groupby(level=0).agg(
        {
            "geom_right": agg_geoms_partial,
            **{
                c: "first"
                for c in df1.columns
                if c not in ["_overlay_index_right", "geom_right"]
            },
        }
    )

    assert clip_left["geometry"].notna().all()
    assert clip_left["geom_right"].notna().all()

    clip_left["geometry"] = _try_difference(
        clip_left["geometry"].to_numpy(),
        clip_left["geom_right"].to_numpy(),
        grid_size=grid_size,
    )

    return clip_left.drop(columns="geom_right")


def _shapely_diffclip_right(pairs, df1, df2, grid_size, rsuffix):
    agg_geoms_partial = functools.partial(agg_geoms, grid_size=grid_size)

    clip_right = (
        pairs.rename(columns={"geometry": "geom_left", "geom_right": "geometry"})
        .groupby(by="_overlay_index_right")
        .agg(
            {
                "geom_left": agg_geoms_partial,
                "geometry": "first",
            }
        )
        .join(df2.drop(columns=["geometry"]))
        .rename(
            columns={
                c: f"{c}{rsuffix}" if c in df1.columns and c != "geometry" else c
                for c in df2.columns
            }
        )
    )

    assert clip_right["geometry"].notna().all()
    assert clip_right["geom_left"].notna().all()

    clip_right["geometry"] = _try_difference(
        clip_right["geometry"].to_numpy(),
        clip_right["geom_left"].to_numpy(),
        grid_size=grid_size,
    )

    return clip_right.drop(columns="geom_left")


def _try_difference(left, right, grid_size):
    """Try difference overlay, then make_valid and retry."""
    try:
        return difference(
            left,
            right,
            grid_size=grid_size,
        )
    except GEOSException:
        return difference(
            make_valid(left),
            make_valid(right),
            grid_size=grid_size,
        )


def agg_geoms(g, grid_size=None):
    return (
        make_valid(unary_union(g, grid_size=grid_size)) if len(g) > 1 else make_valid(g)
    )
