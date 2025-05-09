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
from geopandas import GeoDataFrame
from pandas import DataFrame
from shapely import Geometry
from shapely import box
from shapely import difference
from shapely import intersection
from shapely import is_empty
from shapely import make_valid
from shapely import union_all

from ..conf import _get_instance
from ..conf import config
from .general import _determine_geom_type_args
from .general import clean_geoms
from .geometry_types import get_geom_type
from .geometry_types import make_all_singlepart
from .geometry_types import to_single_geom_type
from .runners import OverlayRunner
from .runners import RTreeQueryRunner
from .runners import UnionRunner

DEFAULT_GRID_SIZE = None
DEFAULT_LSUFFIX = "_1"
DEFAULT_RSUFFIX = "_2"


def clean_overlay(
    df1: GeoDataFrame,
    df2: GeoDataFrame,
    how: str = "intersection",
    keep_geom_type: bool | None = None,
    geom_type: str | None = None,
    predicate: str | None = "intersects",
    grid_size: float | None = None,
    lsuffix: str = DEFAULT_LSUFFIX,
    rsuffix: str = DEFAULT_RSUFFIX,
    n_jobs: int = 1,
    rtree_runner: RTreeQueryRunner | None = None,
    union_runner: UnionRunner | None = None,
    overlay_runner: OverlayRunner | None = None,
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
        predicate: Spatial predicate in the spatial tree.
        lsuffix: Suffix of columns in df1 that are also in df2.
        rsuffix: Suffix of columns in df2 that are also in df1.
        n_jobs: number of jobs. Defaults to 1.
        union_runner: Optionally debug/manipulate the spatial union operations.
            See the 'runners' module for example implementations.
        rtree_runner: Optionally debug/manipulate the spatial indexing operations.
            See the 'runners' module for example implementations.
        overlay_runner: Optionally debug/manipulate the spatial overlay operations.
            See the 'runners' module for example implementations.

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

    if rtree_runner is None:
        rtree_runner = _get_instance(config, "rtree_runner", n_jobs=n_jobs)
    if union_runner is None:
        union_runner = _get_instance(config, "union_runner", n_jobs=n_jobs)
    if overlay_runner is None:
        overlay_runner = _get_instance(config, "overlay_runner", n_jobs=n_jobs)

    crs = df1.crs

    # original_geom_type = geom_type

    df1, geom_type, keep_geom_type = _determine_geom_type_args(
        df1, geom_type, keep_geom_type
    )

    if not geom_type or geom_type == "mixed":
        if keep_geom_type and geom_type == "mixed":
            raise ValueError(
                "mixed geometries are not allowed when geom_type isn't specified.",
                df1.geometry.geom_type.value_counts(),
            )

    if geom_type == "polygon" or get_geom_type(df1) == "polygon":
        df1.geometry = df1.buffer(0)
    if geom_type == "polygon" or get_geom_type(df2) == "polygon":
        df2.geometry = df2.buffer(0)

    df1 = clean_geoms(df1)
    df2 = clean_geoms(df2)

    df1 = make_all_singlepart(df1, ignore_index=True)
    df2 = make_all_singlepart(df2, ignore_index=True)

    if keep_geom_type:
        df1 = to_single_geom_type(df1, geom_type)

    if geom_type and get_geom_type(df1) == get_geom_type(df2):
        df2 = to_single_geom_type(df2, geom_type)

    assert df1.is_valid.all(), [
        geom.wkt for geom in df1[lambda x: x.is_valid == False].geometry
    ]
    assert df2.is_valid.all(), [
        geom.wkt for geom in df2[lambda x: x.is_valid == False].geometry
    ]
    assert df1.geometry.notna().all(), df1[lambda x: x.isna()]
    assert df2.geometry.notna().all(), df2[lambda x: x.isna()]

    box1 = box(*df1.total_bounds)
    box2 = box(*df2.total_bounds)

    if not grid_size and (
        (not len(df1) or not len(df2))
        or (not box1.intersects(box2) and how == "intersection")
    ):
        return _no_intersections_return(df1, df2, how, lsuffix, rsuffix)

    if df1.geometry.name != "geometry":
        df1 = df1.rename_geometry("geometry")

    if df2.geometry.name != "geometry":
        df2 = df2.rename_geometry("geometry")

    # to pandas because GeoDataFrame constructor is slow
    df1 = DataFrame(df1).reset_index(drop=True)
    df2 = DataFrame(df2).reset_index(drop=True)
    df1.geometry.values.crs = None
    df2.geometry.values.crs = None

    overlayed = (
        gpd.GeoDataFrame(
            _shapely_pd_overlay(
                df1,
                df2,
                how=how,
                grid_size=grid_size,
                lsuffix=lsuffix,
                rsuffix=rsuffix,
                geom_type=geom_type,
                predicate=predicate,
                rtree_runner=rtree_runner,
                overlay_runner=overlay_runner,
                union_runner=union_runner,
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
    geom_col = df1.geometry.name
    df1_cols = df1.columns.difference({geom_col})
    df2_cols = df2.columns.difference({df2.geometry.name})
    cols_with_suffix = [
        f"{col}{lsuffix}" if col in df2_cols else col for col in df1_cols
    ] + [f"{col}{rsuffix}" if col in df1_cols else col for col in df2_cols]

    return GeoDataFrame(
        pd.DataFrame(columns=cols_with_suffix + [geom_col]),
        geometry=geom_col,
        crs=df1.crs,
    )


def _no_intersections_return(
    df1: GeoDataFrame, df2: GeoDataFrame, how: str, lsuffix, rsuffix: str
) -> GeoDataFrame:
    """Return with no overlay if no intersecting bounding box."""
    if how == "intersection":
        return _join_and_get_no_rows(df1, df2, lsuffix, rsuffix)

    if how == "difference":
        return df1.reset_index(drop=True)

    if how == "identity":
        # add suffixes and return df1
        df_template = _join_and_get_no_rows(df1, df2, lsuffix, rsuffix)
        df2_cols = df2.columns.difference({df2.geometry.name})
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
    grid_size: float,
    predicate: str,
    lsuffix: str,
    rsuffix: str,
    geom_type: str | None,
    rtree_runner: RTreeQueryRunner,
    overlay_runner: OverlayRunner,
    union_runner: UnionRunner,
) -> DataFrame:
    left, right = rtree_runner.run(
        df1.geometry.values, df2.geometry.values, predicate=predicate
    )
    pairs = _get_intersects_pairs(df1, df2, left, right, rsuffix)
    assert pairs["geometry"].notna().all(), pairs.geometry[lambda x: x.isna()]
    assert pairs["geom_right"].notna().all(), pairs.geom_right[lambda x: x.isna()]

    if how == "intersection":
        overlayed = [
            _intersection(
                pairs,
                grid_size=grid_size,
                geom_type=geom_type,
                overlay_runner=overlay_runner,
            )
        ]

    elif how == "difference":
        overlayed = _difference(
            pairs,
            df1,
            left,
            grid_size=grid_size,
            geom_type=geom_type,
            overlay_runner=overlay_runner,
            union_runner=union_runner,
        )

    elif how == "symmetric_difference":
        overlayed = _symmetric_difference(
            pairs,
            df1,
            df2,
            left,
            right,
            grid_size=grid_size,
            rsuffix=rsuffix,
            geom_type=geom_type,
            overlay_runner=overlay_runner,
            union_runner=union_runner,
        )

    elif how == "identity":
        overlayed = _identity(
            pairs,
            df1,
            left,
            grid_size=grid_size,
            geom_type=geom_type,
            overlay_runner=overlay_runner,
            union_runner=union_runner,
        )

    elif how == "union":
        overlayed = _union(
            pairs,
            df1,
            df2,
            left,
            right,
            grid_size=grid_size,
            rsuffix=rsuffix,
            geom_type=geom_type,
            overlay_runner=overlay_runner,
            union_runner=union_runner,
        )

    elif how == "update":
        overlayed = _update(
            pairs,
            df1,
            df2,
            left=left,
            grid_size=grid_size,
            geom_type=geom_type,
            overlay_runner=overlay_runner,
            union_runner=union_runner,
        )

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
    overlayed = overlayed.loc[
        lambda x: (x["geometry"].notna().values) & (~is_empty(x["geometry"].values))
    ]

    return overlayed


def _update(
    pairs: pd.DataFrame,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    left: np.ndarray,
    grid_size: float | None | int,
    geom_type: str | None,
    overlay_runner: OverlayRunner,
    union_runner: UnionRunner,
) -> GeoDataFrame:
    overlayed = _difference(
        pairs,
        df1,
        left,
        grid_size=grid_size,
        geom_type=geom_type,
        overlay_runner=overlay_runner,
        union_runner=union_runner,
    )

    return overlayed + [df2]


def _intersection(
    pairs: pd.DataFrame,
    grid_size: None | float | int,
    geom_type: str | None,
    overlay_runner: OverlayRunner,
) -> GeoDataFrame:
    if not len(pairs):
        return pairs.drop(columns="geom_right")
    intersections = pairs.copy()
    intersections["geometry"] = overlay_runner.run(
        intersection,
        intersections["geometry"].to_numpy(),
        intersections["geom_right"].to_numpy(),
        grid_size=grid_size,
        geom_type=geom_type,
    )
    return intersections.drop(columns="geom_right")


def _union(
    pairs: pd.DataFrame,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    left: np.ndarray,
    right: np.ndarray,
    grid_size: int | float | None,
    rsuffix: str,
    geom_type: str | None,
    overlay_runner: OverlayRunner,
    union_runner: UnionRunner,
) -> list[GeoDataFrame]:
    merged = []
    if len(left):
        intersections = _intersection(
            pairs,
            grid_size=grid_size,
            geom_type=geom_type,
            overlay_runner=overlay_runner,
        )
        merged.append(intersections)
    symmdiff = _symmetric_difference(
        pairs,
        df1,
        df2,
        left,
        right,
        grid_size=grid_size,
        rsuffix=rsuffix,
        geom_type=geom_type,
        overlay_runner=overlay_runner,
        union_runner=union_runner,
    )
    merged += symmdiff
    return merged


def _identity(
    pairs: pd.DataFrame,
    df1: pd.DataFrame,
    left: np.ndarray,
    grid_size: int | float | None,
    geom_type: str | None,
    overlay_runner: OverlayRunner,
    union_runner: UnionRunner,
) -> list[GeoDataFrame]:
    merged = []
    if len(left):
        intersections = _intersection(
            pairs,
            grid_size=grid_size,
            geom_type=geom_type,
            overlay_runner=overlay_runner,
        )
        merged.append(intersections)
    diff = _difference(
        pairs,
        df1,
        left,
        geom_type=geom_type,
        grid_size=grid_size,
        overlay_runner=overlay_runner,
        union_runner=union_runner,
    )
    merged += diff
    return merged


def _symmetric_difference(
    pairs: pd.DataFrame,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    left: np.ndarray,
    right: np.ndarray,
    grid_size: int | float | None,
    rsuffix: str,
    geom_type: str | None,
    overlay_runner: OverlayRunner,
    union_runner: UnionRunner,
) -> list[GeoDataFrame]:
    merged = []

    difference_left = _difference(
        pairs,
        df1,
        left,
        grid_size=grid_size,
        geom_type=geom_type,
        overlay_runner=overlay_runner,
        union_runner=union_runner,
    )
    merged += difference_left

    if len(left):
        clip_right = _shapely_diffclip_right(
            pairs,
            df1,
            df2,
            grid_size=grid_size,
            rsuffix=rsuffix,
            geom_type=geom_type,
            overlay_runner=overlay_runner,
            union_runner=union_runner,
        )
        merged.append(clip_right)

    diff_right = _add_from_right(df1, df2, right, rsuffix)
    merged.append(diff_right)

    return merged


def _difference(
    pairs: pd.DataFrame,
    df1: pd.DataFrame,
    left: np.ndarray,
    grid_size: int | float | None,
    geom_type: str | None,
    overlay_runner: OverlayRunner,
    union_runner: UnionRunner,
) -> list[GeoDataFrame]:
    merged = []
    if len(left):
        clip_left = _shapely_diffclip_left(
            pairs=pairs,
            df1=df1,
            grid_size=grid_size,
            geom_type=geom_type,
            overlay_runner=overlay_runner,
            union_runner=union_runner,
        )
        merged.append(clip_left)
    diff_left = _add_indices_from_left(df1, left)
    merged.append(diff_left)
    return merged


def _get_intersects_pairs(
    df1: GeoDataFrame,
    df2: GeoDataFrame,
    left: np.ndarray,
    right: np.ndarray,
    rsuffix: str,
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


def _add_suffix_left(
    overlayed: pd.DataFrame, df1: pd.DataFrame, df2: pd.DataFrame, lsuffix: str
):
    """Separating this from _add_indices_from_left, since this suffix is not needed in difference."""
    return overlayed.rename(
        columns={
            c: (
                f"{c}{lsuffix}"
                if c in df1.columns and c in df2.columns and c != "geometry"
                else c
            )
            for c in overlayed.columns
        }
    )


def _add_indices_from_left(df1: pd.DataFrame, left: np.ndarray) -> pd.DataFrame:
    return df1.take(np.setdiff1d(np.arange(len(df1)), left))


def _add_from_right(
    df1: GeoDataFrame, df2: GeoDataFrame, right: np.ndarray, rsuffix: str
) -> GeoDataFrame:
    return df2.take(np.setdiff1d(np.arange(len(df2)), right)).rename(
        columns={
            c: f"{c}{rsuffix}" if c in df1.columns and c != "geometry" else c
            for c in df2.columns
        }
    )


def _shapely_diffclip_left(
    pairs: pd.DataFrame,
    df1: pd.DataFrame,
    grid_size: int | float | None,
    geom_type: str | None,
    overlay_runner: OverlayRunner,
    union_runner: UnionRunner,
) -> pd.DataFrame:
    """Aggregate areas in right by unique values from left, then erases those from left."""
    keep_cols = list(df1.columns.difference({"_overlay_index_right"})) + ["geom_right"]

    agg_geoms_partial = functools.partial(_agg_geoms, grid_size=grid_size)

    try:
        only_one = pairs.groupby(level=0).transform("size") == 1
        one_hit = pairs.loc[only_one, list(keep_cols)]
        many_hits = pairs.loc[~only_one, list(keep_cols) + ["_overlay_index_right"]]
        # keep first in non-geom-cols, agg only geom col bacause of speed
        many_hits_agged = many_hits.loc[
            lambda x: ~x.index.duplicated(),
            lambda x: x.columns.difference({"geom_right"}),
        ]

        index_mapper = {
            i: x
            for i, x in many_hits.groupby(level=0)["_overlay_index_right"]
            .unique()
            .apply(lambda j: tuple(sorted(j)))
            .items()
        }

        many_hits_agged["_right_indices"] = index_mapper

        inverse_index_mapper = pd.Series(
            {
                x[0]: x
                for x in many_hits_agged.reset_index()
                .groupby("_right_indices")["index"]
                .unique()
                .apply(tuple)
            }
        ).explode()
        inverse_index_mapper = pd.Series(
            inverse_index_mapper.index, index=inverse_index_mapper.values
        )

        agger = (
            pd.Series(index_mapper.values(), index=index_mapper.keys())
            .drop_duplicates()
            .explode()
            .to_frame("_overlay_index_right")
        )
        agger["geom_right"] = agger["_overlay_index_right"].map(
            {
                i: g
                for i, g in zip(
                    many_hits["_overlay_index_right"],
                    many_hits["geom_right"],
                    strict=False,
                )
            }
        )

        agged = union_runner.run(agger["geom_right"], level=0)
        # agged = pd.Series(

        #     {
        #         i: agg_geoms_partial(geoms)
        #         for i, geoms in agger.groupby(level=0)["geom_right"]
        #     }
        # )
        many_hits_agged["geom_right"] = inverse_index_mapper.map(agged)
        many_hits_agged = many_hits_agged.drop(columns=["_right_indices"])

        clip_left = pd.concat([one_hit, many_hits_agged])
    except IndexError:
        clip_left = pairs.loc[:, list(keep_cols)]

    assert clip_left["geometry"].notna().all(), clip_left["geometry"][
        lambda x: x.isna()
    ]
    assert clip_left["geom_right"].notna().all(), clip_left["geom_right"][
        lambda x: x.isna()
    ]

    clip_left["geometry"] = overlay_runner.run(
        difference,
        clip_left["geometry"].to_numpy(),
        clip_left["geom_right"].to_numpy(),
        grid_size=grid_size,
        geom_type=geom_type,
    )

    return clip_left.drop(columns="geom_right")


def _shapely_diffclip_right(
    pairs: pd.DataFrame,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    grid_size: int | float | None,
    rsuffix: str,
    geom_type: str | None,
    overlay_runner: OverlayRunner,
    union_runner: UnionRunner,
) -> pd.DataFrame:
    agg_geoms_partial = functools.partial(_agg_geoms, grid_size=grid_size)

    pairs = pairs.rename(columns={"geometry": "geom_left", "geom_right": "geometry"})

    try:
        only_one = pairs.groupby("_overlay_index_right").transform("size") == 1
        one_hit = pairs[only_one].set_index("_overlay_index_right")[
            ["geom_left", "geometry"]
        ]
        many_hits_ungrouped = pairs[~only_one].set_index("_overlay_index_right")
        many_hits = pd.DataFrame(index=many_hits_ungrouped.index.unique())
        many_hits["geometry"] = many_hits_ungrouped.groupby(level=0)["geometry"].first()
        many_hits["geom_left"] = union_runner.run(
            many_hits_ungrouped["geom_left"], level=0
        )
        # many_hits = (
        #     pairs[~only_one]
        #     .groupby("_overlay_index_right")
        #     .agg(
        #         {
        #             "geom_left": agg_geoms_partial,
        #             "geometry": "first",
        #         }
        #     )
        # )
        clip_right = (
            pd.concat([one_hit, many_hits])
            .join(df2.drop(columns=["geometry"]))
            .rename(
                columns={
                    c: f"{c}{rsuffix}" if c in df1.columns and c != "geometry" else c
                    for c in df2.columns
                }
            )
        )
    except IndexError:
        clip_right = pairs.join(df2.drop(columns=["geometry"])).rename(
            columns={
                c: f"{c}{rsuffix}" if c in df1.columns and c != "geometry" else c
                for c in df2.columns
            }
        )

    assert clip_right["geometry"].notna().all(), clip_right["geometry"][
        lambda x: x.isna()
    ]
    assert clip_right["geom_left"].notna().all(), clip_right["geom_left"][
        lambda x: x.isna()
    ]

    clip_right["geometry"] = overlay_runner.run(
        difference,
        clip_right["geometry"].to_numpy(),
        clip_right["geom_left"].to_numpy(),
        grid_size=grid_size,
        geom_type=geom_type,
    )

    return clip_right.drop(columns="geom_left")


def _agg_geoms(g: np.ndarray, grid_size: int | float | None = None) -> Geometry:
    return make_valid(union_all(g, grid_size=grid_size))
