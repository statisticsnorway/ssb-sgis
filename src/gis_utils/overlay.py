from typing import Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from shapely import STRtree, difference, intersection, union_all
from shapely.wkt import dumps, loads

from .geopandas_utils import (
    clean_geoms,
    gdf_concat,
    get_geom_type,
    is_single_geom_type,
    push_geom_col,
    to_single_geom_type,
)


def overlay(
    left_gdf: GeoDataFrame,
    right_gdf: GeoDataFrame,
    how: str = "intersection",
    drop_dupcol: bool = True,
    keep_geom_type: bool = True,
    geom_type: str | None = None,
    geom_type_left: str | None = None,
    geom_type_right: str | None = None,
    **kwargs,
) -> GeoDataFrame:
    """
    Try to do a geopandas overlay, then try to set common crs and clean geometries
    before retrying the overlay. Then, last resort is to do the overlay in shapely,
    as suggested here: https://github.com/geopandas/geopandas/issues/2792

    """

    if geom_type:
        if geom_type_left or geom_type_right:
            raise ValueError(
                "Specify only geom_type or geom_type_left and geom_type_right"
            )

        geom_type_left, geom_type_right = geom_type, geom_type

        left_gdf = to_single_geom_type(left_gdf, geom_type=geom_type)
        right_gdf = to_single_geom_type(right_gdf, geom_type=geom_type)

    elif geom_type_left or geom_type_right:
        if not geom_type_left:
            geom_type_left = geom_type_right
        if not geom_type_right:
            geom_type_right = geom_type_left
        left_gdf = to_single_geom_type(left_gdf, geom_type=geom_type_left)
        right_gdf = to_single_geom_type(right_gdf, geom_type=geom_type_right)
    else:
        geom_type_left, geom_type_right = None, None

    if not is_single_geom_type(left_gdf):
        raise ValueError(
            "mixed geometry types in 'left_gdf'. Specify 'geom_type' to keep"
        )
    if not is_single_geom_type(right_gdf):
        raise ValueError(
            "mixed geometry types in 'right_gdf'. Specify 'geom_type' to keep"
        )

    if keep_geom_type:
        geom_type_left = get_geom_type(left_gdf)

    # Allowed operations
    allowed_hows = [
        "intersection",
        "union",
        "identity",
        "symmetric_difference",
        "difference",  # aka erase
        "update",
    ]
    # Error Messages
    if how not in allowed_hows:
        raise ValueError(
            f"`how` was '{how}' but is expected to be in {', '.join(allowed_hows)}"
        )

    left_gdf = left_gdf.loc[:, ~left_gdf.columns.str.contains("index|level_")]
    right_gdf = right_gdf.loc[:, ~right_gdf.columns.str.contains("index|level_")]

    # remove columns in right_gdf that are in left_gdf (except for geometry column)
    if drop_dupcol:
        right_gdf = right_gdf.loc[
            :,
            right_gdf.columns.difference(
                left_gdf.columns.difference([left_gdf._geometry_column_name])
            ),
        ]

    if how == "update":
        overlayfunc = overlay_update
    else:
        overlayfunc = gpd.overlay
        kwargs = kwargs | {"how": how}

    try:
        joined = overlayfunc(left_gdf, right_gdf, **kwargs)
    except Exception:
        try:
            right_gdf = right_gdf.to_crs(left_gdf.crs)
            left_gdf = clean_geoms(left_gdf, geom_type=geom_type_left)
            right_gdf = clean_geoms(right_gdf, geom_type=geom_type_right)
            joined = overlayfunc(left_gdf, right_gdf, **kwargs)
        except Exception as e:
            if how == "update":
                raise e
            joined = clean_shapely_overlay(left_gdf, right_gdf, how=how)

    joined = clean_geoms(joined)

    if keep_geom_type:
        joined = to_single_geom_type(joined, geom_type_left)

    return joined.loc[:, ~joined.columns.str.contains("index|level_")]


def overlay_update(
    left_gdf: GeoDataFrame, right_gdf: GeoDataFrame, **kwargs
) -> GeoDataFrame:
    """En overlay-variant som ikke finnes i geopandas."""

    try:
        out = left_gdf.overlay(right_gdf, how="difference", **kwargs)
    except Exception:
        out = clean_shapely_overlay(left_gdf, right_gdf, how="difference")
    out = out.loc[:, ~out.columns.str.contains("index|level_")]
    out = gdf_concat([out, right_gdf])
    return out


def clean_shapely_overlay(
    left_gdf: GeoDataFrame,
    right_gdf: GeoDataFrame,
    how: str = "intersection",
) -> GeoDataFrame:
    # Allowed operations
    allowed_hows = [
        "intersection",
        "union",
        "identity",
        "symmetric_difference",
        "difference",  # aka erase
    ]
    # Error messages
    if how not in allowed_hows:
        raise ValueError(
            f"`how` was '{how}' but is expected to be in {', '.join(allowed_hows)}"
        )

    left_gdf = clean_geoms(left_gdf)
    right_gdf = clean_geoms(right_gdf)

    left_gdf = left_gdf.explode(ignore_index=True)
    right_gdf = right_gdf.explode(ignore_index=True)

    unioned = (
        _shapely_overlay(left_gdf, right_gdf, how=how)
        .pipe(clean_geoms)
        .reset_index(drop=True)
    )

    return unioned


def _shapely_overlay(df1: GeoDataFrame, df2: GeoDataFrame, how: str) -> GeoDataFrame:
    merged = []

    tree = STRtree(df2.geometry.values)
    left, right = tree.query(df1.geometry.values, predicate="intersects")

    if len(left):
        pairs = pd.concat(
            [
                df1.take(left),
                (
                    pd.DataFrame(
                        {"index_right": right}, index=df1.index.values.take(left)
                    )
                ),
            ],
            axis=1,
        ).join(
            df2.rename(columns={"geometry": "geom_right"}),
            on="index_right",
            rsuffix="_2",
        )

        if how == "intersection":
            return _shapely_intersection(pairs)

        if how == "difference":
            clip_left = _shapely_difference_left(pairs, df1)
            return push_geom_col(clip_left)

        if how == "union" or how == "identity":
            intersections = _shapely_intersection(pairs)
            merged.append(intersections)

        clip_left = _shapely_difference_left(pairs, df1)
        merged.append(clip_left)

        if how == "union" or how == "symmetric_difference":
            clip_right = _shapely_difference_right(pairs, df1, df2)
            merged.append(clip_right)

    # add any from left or right data frames that did not intersect
    diff_left = df1.take(np.setdiff1d(np.arange(len(df1)), left))
    merged.append(diff_left)

    if how == "union" or how == "symmetric_difference":
        diff_right = df2.take(np.setdiff1d(np.arange(len(df2)), right)).rename(
            columns={
                c: f"{c}_2" if c in df1.columns and c != "geometry" else c
                for c in df2.columns
            }
        )
        merged.append(diff_right)

    # merge all data frames
    merged = gdf_concat(merged, ignore_index=True)

    # push geometry column to the end
    merged = push_geom_col(merged)

    return merged


def _shapely_intersection(pairs: GeoDataFrame) -> GeoDataFrame:
    intersections = pairs.copy()
    intersections["geometry"] = intersection(
        intersections.geometry.values, intersections.geom_right.values
    )
    intersections = intersections.drop(columns=["index_right", "geom_right"])
    return intersections


def _shapely_difference_left(pairs, df1):
    """Aggregate areas in right by unique values of left, then use those to clip
    areas out of left"""
    clip_left = gpd.GeoDataFrame(
        pairs.groupby(level=0).agg(
            {
                "geom_right": lambda g: union_all(g) if len(g) > 1 else g,
                **{
                    c: "first"
                    for c in df1.columns
                    if not c in ["index_right", "geom_right"]
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


def _shapely_difference_right(pairs, df1, df2):
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


def try_overlay(
    gdf1: GeoDataFrame,
    gdf2: GeoDataFrame,
    presicion_col: bool = True,
    max_rounding: int = 3,
    single_geom_type: bool = True,
    **kwargs,
) -> GeoDataFrame:
    """
    Overlay, i hvert fall union, har gitt TopologyException: found non-noded
    intersection error from overlay. https://github.com/geopandas/geopandas/issues/1724
    En løsning er å avrunde koordinatene for å få valid polygon. Prøver først uten
    avrunding, så runder av til 10 koordinatdesimaler, så 9, 8, ..., og så gir opp på 0

    Args:
      presisjonskolonne: om man skal inkludere en kolonne som angir hvilken avrunding
        som måtte til.
      max_avrunding: hvilken avrunding man stopper på. 0 betyr at man fortsetter fram
        til 0 desimaler.
    """

    try:
        gdf1 = clean_geoms(gdf1, single_geom_type=single_geom_type)
        gdf2 = clean_geoms(gdf2, single_geom_type=single_geom_type)
        return gdf1.overlay(gdf2, **kwargs)

    except Exception:
        # loop through list from 10 to 'max_rounding'

        roundings = list(range(max_rounding, 11))
        roundings.reverse()

        for rounding in roundings:
            try:
                gdf1.geometry = [
                    loads(dumps(gdf1, rounding_precision=rounding))
                    for geom in gdf1.geometry
                ]
                gdf2.geometry = [
                    loads(dumps(gdf2, rounding_precision=rounding))
                    for geom in gdf2.geometry
                ]

                gdf1 = clean_geoms(gdf1, single_geom_type=single_geom_type)
                gdf2 = clean_geoms(gdf2, single_geom_type=single_geom_type)

                overlayet = gdf1.overlay(gdf2, **kwargs)

                if presicion_col:
                    overlayet["avrunding"] = rounding

                return overlayet

            except Exception:
                rounding -= 1

        # returnerer feilmeldingen hvis det fortsatt ikke funker
        gdf1.overlay(gdf2, **kwargs)


def make_valid_with_equal_precision(gdf, precision: int = 10):
    while True:
        gdf.geometry = [
            loads(dumps(geom, rounding_precision=precision)) for geom in gdf.geometry
        ]

        if all(gdf.geometry.is_valid):
            gdf = behold_vanligste_geomtype(gdf)
            break

        gdf = fiks_geometrier(gdf, to_single_geomtype=False)

    return gdf
