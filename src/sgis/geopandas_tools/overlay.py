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
from pyproj import CRS
from shapely import STRtree, difference, intersection, union_all

from .general import _push_geom_col, clean_geoms
from .geometry_types import get_geom_type, to_single_geom_type


def overlay_update(
    df1: GeoDataFrame,
    df2: GeoDataFrame,
    keep_geom_type: bool = True,
    geom_type: str | tuple[str, str] | None = None,
    **kwargs,
) -> GeoDataFrame:
    """Updates df1 by putting df2 on top.

    First runs a difference overlay of df1 and df2 to erase the parts of df1 that is in
    df2. Then concatinates the overlayed geometries with df2.

    Args:
        df1: GeoDataFrame
        df2: GeoDataFrame
        keep_geom_type: If True (default), return only geometries of the same
            geometry type as df1 has, if False, return all resulting geometries.
        geom_type: optionally specify what geometry type to keep before the overlay,
            if there may be mixed geometry types. Either a string with one geom_type
            or a tuple/list with geom_type for df1 and df2 respectfully.
        **kwargs: Additional keyword arguments passed to geopandas.overlay

    Returns:
        GeoDataFrame with overlayed geometries and columns from both GeoDataFrames.

    """
    geom_type_left, geom_type_right = _get_geom_type_left_right(geom_type)

    if keep_geom_type and not geom_type_left:
        geom_type_left = get_geom_type(df1)

    df1 = clean_geoms(df1)
    if geom_type_left:
        df1 = to_single_geom_type(df1, geom_type_left)
    df2 = clean_geoms(df2)
    if geom_type_right:
        df2 = to_single_geom_type(df2, geom_type_right)

    overlayed = df1.overlay(df2[["geometry"]], how="difference", **kwargs)
    overlayed = overlayed.loc[:, ~overlayed.columns.str.contains("index|level_")]
    return pd.concat([overlayed, df2], ignore_index=True)


def clean_overlay(
    df1: GeoDataFrame,
    df2: GeoDataFrame,
    how: str = "intersection",
    keep_geom_type: bool = True,
    geom_type: str | tuple[str, str] | None = None,
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
            if there may be mixed geometry types. Either a string with one geom_type
            or a tuple/list with geom_type for df1 and df2 respectfully.

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
        "difference",  # aka erase
        "update",
    ]
    # Error messages
    if how not in allowed_hows:
        raise ValueError(
            f"`how` was {how!r} but is expected to be in {', '.join(allowed_hows)}"
        )

    if df1.crs != df2.crs:
        raise ValueError(f"'crs' mismatch. Got {df1.crs} and {df2.crs}")

    crs = df1.crs

    geom_type_left, geom_type_right = _get_geom_type_left_right(geom_type)

    if keep_geom_type and not geom_type_left:
        geom_type_left = get_geom_type(df1)

    df1 = clean_geoms(df1)
    df2 = clean_geoms(df2)

    if geom_type_left:
        df1 = to_single_geom_type(df1, geom_type_left)
    if geom_type_right:
        df2 = to_single_geom_type(df2, geom_type_right)

    df1 = df1.explode(ignore_index=True)
    df2 = df2.explode(ignore_index=True)

    overlayed = (
        _shapely_overlay(df1, df2, how=how, crs=crs)
        .pipe(clean_geoms)
        .pipe(_push_geom_col)
    )

    if geom_type_left:
        overlayed = to_single_geom_type(overlayed, geom_type_left)

    return overlayed.reset_index(drop=True)


def _get_geom_type_left_right(
    geom_type: str | tuple | list | None,
) -> tuple[str | None, str | None]:
    if isinstance(geom_type, str):
        return geom_type, geom_type
    elif geom_type is None:
        return None, None

    elif hasattr(geom_type, "__iter__"):
        if len(geom_type) == 1:
            return geom_type[0], geom_type[0]
        elif len(geom_type) == 2:
            return geom_type
        else:
            raise ValueError(
                "'geom_type' should be one or two strings for the left and right gdf"
            )
    else:
        raise ValueError(
            "'geom_type' should be one or two strings for the left and right gdf"
        )


def _shapely_overlay(
    df1: GeoDataFrame, df2: GeoDataFrame, how: str, crs: int | str | None | CRS
) -> GeoDataFrame:
    tree = STRtree(df2.geometry.values)
    left, right = tree.query(df1.geometry.values, predicate="intersects")

    pairs = _get_intersects_pairs(df1, df2, left, right)

    if how == "intersection":
        overlayed = _intersection(pairs, crs=crs)

    elif how == "difference":
        # don't add suffix on difference
        return _difference(pairs, df1, left, crs=crs)

    elif how == "symmetric_difference":
        overlayed = _symmetric_difference(pairs, df1, df2, left, right, crs=crs)

    elif how == "identity":
        overlayed = _identity(pairs, df1, df2, left, crs=crs)

    elif how == "union":
        overlayed = _union(pairs, df1, df2, left, right, crs=crs)

    elif how == "update":
        # don't add suffix on update
        return _update(pairs, df1, df2, left=left, crs=crs)

    overlayed = _add_suffix_left(overlayed, df1, df2)
    return overlayed


def _update(pairs, df1, df2, left, crs) -> GeoDataFrame:
    overlayed = _difference(pairs, df1, left, crs=crs)
    overlayed = overlayed.loc[:, ~overlayed.columns.str.contains("index|level_")]
    return pd.concat([overlayed, df2], ignore_index=True)


def _intersection(pairs, crs) -> GeoDataFrame:
    intersections = pairs.copy()
    intersections["geometry"] = intersection(
        intersections.geometry.values, intersections.geom_right.values
    )
    intersections = intersections.drop(["index_right", "geom_right"], axis=1)

    if crs:
        intersections = intersections.to_crs(crs)
    return intersections


def _union(pairs, df1, df2, left, right, crs):
    merged = []
    if len(left):
        intersections = _intersection(pairs, crs=crs)
        merged.append(intersections)
    symmdiff = _symmetric_difference(pairs, df1, df2, left, right, crs=crs)
    merged.append(symmdiff)
    if crs:
        merged = [gdf.to_crs(crs) for gdf in merged]
    return pd.concat(merged, ignore_index=True).pipe(_push_geom_col)


def _identity(pairs, df1, df2, left, crs):
    merged = []
    if len(left):
        intersections = _intersection(pairs, crs=crs)
        merged.append(intersections)
    diff = _difference(pairs, df1, left, crs=crs)
    merged.append(diff)
    if crs:
        merged = [gdf.to_crs(crs) for gdf in merged]
    return pd.concat(merged, ignore_index=True).pipe(_push_geom_col)


def _symmetric_difference(pairs, df1, df2, left, right, crs):
    merged = []
    difference_left = _difference(pairs, df1, left, crs=crs)
    merged.append(difference_left)
    if len(left):
        clip_right = _shapely_diffclip_right(pairs, df1, df2)
        merged.append(clip_right)
    diff_right = _add_from_right(df1, df2, right)
    merged.append(diff_right)
    if crs:
        merged = [gdf.to_crs(crs) for gdf in merged]
    return pd.concat(merged, ignore_index=True).pipe(_push_geom_col)


def _difference(pairs, df1, left, crs):
    merged = []
    if len(left):
        clip_left = _shapely_diffclip_left(pairs, df1)
        merged.append(clip_left)
    diff_left = _add_from_left(df1, left)
    merged.append(diff_left)
    if crs:
        merged = [gdf.to_crs(crs) for gdf in merged]
    return pd.concat(merged, ignore_index=True).pipe(_push_geom_col)


def _get_intersects_pairs(
    df1: GeoDataFrame, df2: GeoDataFrame, left: np.ndarray, right: np.ndarray
) -> GeoDataFrame:
    return pd.concat(
        [
            df1.take(left),
            (pd.DataFrame({"index_right": right}, index=df1.index.values.take(left))),
        ],
        axis=1,
    ).join(
        df2.rename(columns={"geometry": "geom_right"}),
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


def _shapely_diffclip_left(pairs, df1):
    """Aggregate areas in right by unique values of left, then use those to clip
    areas out of left"""
    clip_left = gpd.GeoDataFrame(
        pairs.groupby(level=0).agg(
            {
                "geom_right": lambda g: union_all(g) if len(g) > 1 else g,
                **{
                    c: "first"
                    for c in df1.columns
                    if c not in ["index_right", "geom_right"]
                },
            }
        ),
        geometry="geometry",
        crs=df1.crs,
    )
    clip_left["geometry"] = difference(
        clip_left.geometry.values, clip_left.geom_right.values
    )
    clip_left = clip_left.drop(columns=["geom_right"])

    return clip_left


def _shapely_diffclip_right(pairs, df1, df2):
    clip_right = (
        gpd.GeoDataFrame(
            pairs.rename(columns={"geometry": "geom_left", "geom_right": "geometry"})
            .groupby(by="index_right")
            .agg(
                {
                    "geom_left": lambda g: union_all(g) if len(g) > 1 else g,
                    "geometry": "first",
                }
            ),
            geometry="geometry",
            crs=df2.crs,
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
        clip_right.geometry.values, clip_right.geom_left.values
    )
    clip_right = clip_right.drop(columns=["geom_left"])
    return clip_right
