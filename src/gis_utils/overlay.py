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
    drop_dupcol: bool = False,
    keep_geom_type: bool = True,
    geom_type: str | tuple[str, str] | list[str, str] | None = None,
    **kwargs,
) -> GeoDataFrame:
    """Try geopandas.overlay, then try alternative function from geopandas issue #2792

    Fixes geometries, then tries a regular geopandas.overlay. If it doesn't succeed,
    tries a solution inspired by: https://github.com/geopandas/geopandas/issues/2792.
    This solution avoids the reduce function and bypasses GEOSExceptions raised in the
    regular geopandas.overlay.

    Like in geopandas, the default is to keep only the geometry type of left_gdf. If
    either 'left_gdf' or 'right_gdf' has mixed geometries, it will raise an
    Exception if not 'geom_type' is specified. 'geom_type' can be a string or a tuple
    of two strings for the geometry type of the left and right GeoDataFrame
    respectfully.

    Args:
        left_gdf:


    """

    # Allowed operations (includes 'update')
    allowed_hows = [
        "intersection",
        "union",
        "identity",
        "symmetric_difference",
        "difference",  # aka erase
        "update",  # not in geopandas
    ]
    # Error Messages
    if how not in allowed_hows:
        raise ValueError(
            f"`how` was '{how}' but is expected to be in {', '.join(allowed_hows)}"
        )

    left_gdf = clean_geoms(left_gdf)
    right_gdf = clean_geoms(right_gdf)

    geom_type_left, geom_type_right = _get_geom_type_left_right(geom_type)

    if geom_type_left:
        left_gdf = to_single_geom_type(left_gdf, geom_type=geom_type_left)
    if geom_type_right:
        right_gdf = to_single_geom_type(right_gdf, geom_type=geom_type_right)

    if not is_single_geom_type(left_gdf):
        raise ValueError(
            "mixed geometry types in 'left_gdf'. Specify 'geom_type' as 'polygon', 'line' or 'point'."
        )
    if not is_single_geom_type(right_gdf):
        raise ValueError(
            "mixed geometry types in 'right_gdf'. Specify 'geom_type' as 'polygon', 'line' or 'point'."
        )

    if keep_geom_type and not geom_type_left:
        geom_type_left = get_geom_type(left_gdf)

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

    # determine what function to use
    if how == "update":
        overlayfunc = overlay_update
    else:
        overlayfunc = gpd.overlay
        kwargs = kwargs | {"how": how}

    try:
        overlayed = overlayfunc(left_gdf, right_gdf, **kwargs)
    except Exception as e:
        if how == "update":
            raise e
        overlayed = clean_shapely_overlay(left_gdf, right_gdf, how=how)

    overlayed = clean_geoms(overlayed)

    if keep_geom_type:
        overlayed = to_single_geom_type(overlayed, geom_type_left)

    return overlayed.loc[:, ~overlayed.columns.str.contains("index|level_")]


def overlay_update(
    left_gdf: GeoDataFrame, right_gdf: GeoDataFrame, **kwargs
) -> GeoDataFrame:
    """Put left_gdf on top of right_gdf"""

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
    keep_geom_type: bool = True,
    geom_type: str | tuple[str, str] | list[str, str] | None = None,
) -> GeoDataFrame:
    """
    It takes two GeoDataFrames, cleans their geometries, explodes them, and then performs a shapely
    overlay operation on them

    Args:
        left_gdf (GeoDataFrame): GeoDataFrame
        right_gdf (GeoDataFrame): GeoDataFrame
        how: Defaults to intersection

    Returns:
      The updated GeoDataFrame
    """

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

    geom_type_left, geom_type_right = _get_geom_type_left_right(geom_type)

    if keep_geom_type and not geom_type_left:
        geom_type_left = get_geom_type(left_gdf)

    left_gdf = clean_geoms(left_gdf, geom_type=geom_type_left)
    right_gdf = clean_geoms(right_gdf, geom_type=geom_type_right)

    left_gdf = left_gdf.explode(ignore_index=True)
    right_gdf = right_gdf.explode(ignore_index=True)

    overlayed = (
        _shapely_overlay(left_gdf, right_gdf, how=how)
        .pipe(clean_geoms, geom_type=geom_type_left)
        .reset_index(drop=True)
    )

    return overlayed


def _get_geom_type_left_right(
    geom_type: str | tuple | list | None,
) -> tuple[str | None, str | None]:
    if isinstance(geom_type, (tuple, list)):
        if len(geom_type) == 1:
            return geom_type[0], geom_type[0]
        elif len(geom_type) == 2:
            return geom_type
        else:
            raise ValueError(
                "'geom_type' should be one or two strings for the left and right gdf"
            )
    elif isinstance(geom_type, str):
        return geom_type, geom_type
    elif geom_type is None:
        return None, None
    else:
        raise ValueError(
            "'geom_type' should be one or two strings for the left and right gdf"
        )


def _shapely_overlay(df1: GeoDataFrame, df2: GeoDataFrame, how: str) -> GeoDataFrame:
    tree = STRtree(df2.geometry.values)
    left, right = tree.query(df1.geometry.values, predicate="intersects")

    pairs = _get_intersects_pairs(df1, df2, left, right)

    if how == "intersection":
        return _intersection(pairs)

    if how == "difference":
        return _difference(pairs, df1, left)

    if how == "symmetric_difference":
        return _symmetric_difference(pairs, df1, df2, left, right)

    if how == "identity":
        return _identity(pairs, df1, left)

    if how == "union":
        return _union(pairs, df1, df2, left, right)


def _intersection(pairs: GeoDataFrame) -> GeoDataFrame:
    intersections = pairs.copy()
    intersections["geometry"] = intersection(
        intersections.geometry.values, intersections.geom_right.values
    )
    intersections = intersections.drop(columns=["index_right", "geom_right"])
    return intersections


def _union(pairs, df1, df2, left, right):
    merged = []
    if len(left):
        intersections = _intersection(pairs)
        merged.append(intersections)
    symmdiff = _symmetric_difference(pairs, df1, df2, left, right)
    merged.append(symmdiff)
    return gdf_concat(merged).pipe(push_geom_col)


def _identity(pairs, df1, left):
    merged = []
    if len(left):
        intersections = _intersection(pairs)
        merged.append(intersections)
    diff = _difference(pairs, df1, left)
    merged.append(diff)
    return gdf_concat(merged).pipe(push_geom_col)


def _symmetric_difference(pairs, df1, df2, left, right):
    merged = []
    difference_left = _difference(pairs, df1, left)
    merged.append(difference_left)
    if len(left):
        clip_right = _shapely_diffclip_right(pairs, df1, df2)
        merged.append(clip_right)
    diff_right = _add_from_right(df1, df2, right)
    merged.append(diff_right)
    return gdf_concat(merged).pipe(push_geom_col)


def _difference(pairs, df1, left):
    merged = []
    if len(left):
        clip_left = _shapely_diffclip_left(pairs, df1)
        merged.append(clip_left)
    diff_left = _add_from_left(df1, left)
    merged.append(diff_left)
    return gdf_concat(merged).pipe(push_geom_col)


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


def try_overlay(
    gdf1: GeoDataFrame,
    gdf2: GeoDataFrame,
    presicion_col: bool = True,
    max_rounding: int = 3,
    geom_type: bool = True,
    **kwargs,
) -> GeoDataFrame:
    try:
        gdf1 = clean_geoms(gdf1, geom_type=geom_type)
        gdf2 = clean_geoms(gdf2, geom_type=geom_type)
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

                gdf1 = clean_geoms(gdf1, geom_type=geom_type)
                gdf2 = clean_geoms(gdf2, geom_type=geom_type)

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
