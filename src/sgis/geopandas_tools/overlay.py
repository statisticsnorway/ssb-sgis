"""Overlay function that avoids a GEOSException from geopandas.overlay.

This module includes the function 'clean_overlay', which bypasses a
GEOSException from the regular geopandas.overlay. The function is a generalized
version of the solution from GH 2792.

'clean_overlay' also includes the overlay type "update", which can be specified in the
"how" parameter, in addition to the five native geopandas how-s.
"""

import functools
from collections.abc import Callable

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from pandas import DataFrame
from shapely import Geometry
from shapely import STRtree
from shapely import box
from shapely import difference
from shapely import intersection
from shapely import make_valid
from shapely import unary_union
from shapely.errors import GEOSException

try:
    import dask.array as da
except ImportError:
    pass

from .general import _determine_geom_type_args
from .general import clean_geoms
from .geometry_types import get_geom_type
from .geometry_types import make_all_singlepart
from .geometry_types import to_single_geom_type

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
    n_jobs: int = 1,
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
        n_jobs: number of threads.
        predicate: Spatial predicate in the spatial tree.
        lsuffix: Suffix of columns in df1 that are also in df2.
        rsuffix: Suffix of columns in df2 that are also in df1.

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

    df1 = clean_geoms(df1)
    df2 = clean_geoms(df2)

    df1 = make_all_singlepart(df1, ignore_index=True)
    df2 = make_all_singlepart(df2, ignore_index=True)

    if keep_geom_type:
        df1 = to_single_geom_type(df1, geom_type)

    if geom_type and get_geom_type(df1) == get_geom_type(df2):
        df2 = to_single_geom_type(df2, geom_type)

    assert df1.is_valid.all(), df1.is_valid.value_counts()
    assert df2.is_valid.all(), df2.is_valid.value_counts()
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
                geom_type=geom_type,
                n_jobs=n_jobs,
                predicate=predicate,
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
    predicate: str = "intersects",
    lsuffix: str = DEFAULT_LSUFFIX,
    rsuffix: str = DEFAULT_RSUFFIX,
    geom_type: str | None = None,
    n_jobs: int = 1,
) -> DataFrame:
    if not grid_size and not len(df1) or not len(df2):
        return _no_intersections_return(df1, df2, how, lsuffix, rsuffix)

    tree = STRtree(df2.geometry.values)
    left, right = tree.query(df1.geometry.values, predicate=predicate)

    pairs = _get_intersects_pairs(df1, df2, left, right, rsuffix)
    assert pairs.geometry.notna().all(), pairs.geometry
    assert pairs.geom_right.notna().all(), pairs.geom_right

    if how == "intersection":
        overlayed = [
            _intersection(
                pairs, grid_size=grid_size, geom_type=geom_type, n_jobs=n_jobs
            )
        ]

    elif how == "difference":
        overlayed = _difference(
            pairs, df1, left, grid_size=grid_size, geom_type=geom_type, n_jobs=n_jobs
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
            n_jobs=n_jobs,
        )

    elif how == "identity":
        overlayed = _identity(
            pairs, df1, left, grid_size=grid_size, geom_type=geom_type, n_jobs=n_jobs
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
            n_jobs=n_jobs,
        )

    elif how == "update":
        overlayed = _update(
            pairs,
            df1,
            df2,
            left=left,
            grid_size=grid_size,
            n_jobs=n_jobs,
            geom_type=geom_type,
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
    # None and empty are falsy
    overlayed = overlayed.loc[lambda x: x["geometry"].notna()]

    return overlayed


def _update(
    pairs: pd.DataFrame,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    left: np.ndarray,
    grid_size: float | None | int,
    geom_type: str | None,
    n_jobs: int,
) -> GeoDataFrame:
    overlayed = _difference(
        pairs, df1, left, grid_size=grid_size, geom_type=geom_type, n_jobs=n_jobs
    )

    return overlayed + [df2]


def _run_overlay_dask(
    arr1: np.ndarray,
    arr2: np.ndarray,
    func: Callable,
    n_jobs: int,
    grid_size: float | int | None,
) -> np.ndarray:
    if len(arr1) // n_jobs <= 1:
        try:
            return func(arr1, arr2, grid_size=grid_size)
        except TypeError as e:
            raise TypeError(e, {type(x) for x in arr1}, {type(x) for x in arr2}) from e
    arr1 = da.from_array(arr1, chunks=len(arr1) // n_jobs)
    arr2 = da.from_array(arr2, chunks=len(arr2) // n_jobs)
    res = arr1.map_blocks(func, arr2, grid_size=grid_size, dtype=float)
    return res.compute(scheduler="threads", optimize_graph=False, num_workers=n_jobs)


def _run_overlay_joblib_threading(
    arr1: np.ndarray,
    arr2: np.ndarray,
    func: Callable,
    n_jobs: int,
    grid_size: int | float | None,
) -> list[Geometry]:
    if len(arr1) // n_jobs <= 1:
        try:
            return func(arr1, arr2, grid_size=grid_size)
        except TypeError as e:
            raise TypeError(e, {type(x) for x in arr1}, {type(x) for x in arr2}) from e
    with joblib.Parallel(n_jobs=n_jobs, backend="threading") as parallel:
        return parallel(
            joblib.delayed(func)(g1, g2, grid_size=grid_size)
            for g1, g2 in zip(arr1, arr2, strict=True)
        )


def _intersection(
    pairs: pd.DataFrame,
    grid_size: None | float | int,
    geom_type: str | None,
    n_jobs: int = 1,
) -> GeoDataFrame:
    if not len(pairs):
        return pairs.drop(columns="geom_right")

    intersections = pairs.copy()

    arr1 = intersections["geometry"].to_numpy()
    arr2 = intersections["geom_right"].to_numpy()

    if n_jobs > 1 and len(arr1) / n_jobs > 10:
        try:
            res = _run_overlay_joblib_threading(
                arr1,
                arr2,
                func=intersection,
                n_jobs=n_jobs,
                grid_size=grid_size,
            )
        except GEOSException:
            arr1 = make_valid_and_keep_geom_type(
                arr1, geom_type=geom_type, n_jobs=n_jobs
            )
            arr2 = make_valid_and_keep_geom_type(
                arr2, geom_type=geom_type, n_jobs=n_jobs
            )
            arr1 = arr1.loc[lambda x: x.index.isin(arr2.index)]
            arr2 = arr2.loc[lambda x: x.index.isin(arr1.index)]

            res = _run_overlay_joblib_threading(
                arr1.to_numpy(),
                arr2.to_numpy(),
                func=intersection,
                n_jobs=n_jobs,
                grid_size=grid_size,
            )
        intersections["geometry"] = res
        return intersections.drop(columns="geom_right")

    try:
        intersections["geometry"] = intersection(
            intersections["geometry"].to_numpy(),
            intersections["geom_right"].to_numpy(),
            grid_size=grid_size,
        )
    except GEOSException:
        left = make_valid_and_keep_geom_type(
            intersections["geometry"].to_numpy(), geom_type, n_jobs=n_jobs
        )
        right = make_valid_and_keep_geom_type(
            intersections["geom_right"].to_numpy(), geom_type, n_jobs=n_jobs
        )
        left = left.loc[lambda x: x.index.isin(right.index)]
        right = right.loc[lambda x: x.index.isin(left.index)]

        intersections["geometry"] = intersection(
            left.to_numpy(), right.to_numpy(), grid_size=grid_size
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
    n_jobs: int = 1,
) -> list[GeoDataFrame]:
    merged = []
    if len(left):
        intersections = _intersection(
            pairs, grid_size=grid_size, geom_type=geom_type, n_jobs=n_jobs
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
        n_jobs=n_jobs,
    )
    merged += symmdiff
    return merged


def _identity(
    pairs: pd.DataFrame,
    df1: pd.DataFrame,
    left: np.ndarray,
    grid_size: int | float | None,
    geom_type: str | None,
    n_jobs: int = 1,
) -> list[GeoDataFrame]:
    merged = []
    if len(left):
        intersections = _intersection(
            pairs, grid_size=grid_size, geom_type=geom_type, n_jobs=n_jobs
        )
        merged.append(intersections)
    diff = _difference(pairs, df1, left, grid_size=grid_size, n_jobs=n_jobs)
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
    n_jobs: int = 1,
) -> list[GeoDataFrame]:
    merged = []

    difference_left = _difference(
        pairs, df1, left, grid_size=grid_size, geom_type=geom_type, n_jobs=n_jobs
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
            n_jobs=n_jobs,
        )
        merged.append(clip_right)

    diff_right = _add_from_right(df1, df2, right, rsuffix)
    merged.append(diff_right)

    return merged


def _difference(
    pairs: pd.DataFrame,
    df1: pd.DataFrame,
    left: np.ndarray,
    grid_size: int | float | None = None,
    geom_type: str | None = None,
    n_jobs: int = 1,
) -> list[GeoDataFrame]:
    merged = []
    if len(left):
        clip_left = _shapely_diffclip_left(
            pairs=pairs,
            df1=df1,
            grid_size=grid_size,
            geom_type=geom_type,
            n_jobs=n_jobs,
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
    n_jobs: int,
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

        agged = pd.Series(
            {
                i: agg_geoms_partial(geoms)
                for i, geoms in agger.groupby(level=0)["geom_right"]
            }
        )
        many_hits_agged["geom_right"] = inverse_index_mapper.map(agged)
        many_hits_agged = many_hits_agged.drop(columns=["_right_indices"])

        clip_left = pd.concat([one_hit, many_hits_agged])
    except IndexError:
        clip_left = pairs.loc[:, list(keep_cols)]

    assert clip_left["geometry"].notna().all()
    assert clip_left["geom_right"].notna().all()

    clip_left["geometry"] = _try_difference(
        clip_left["geometry"].to_numpy(),
        clip_left["geom_right"].to_numpy(),
        grid_size=grid_size,
        geom_type=geom_type,
        n_jobs=n_jobs,
    )

    return clip_left.drop(columns="geom_right")


def _shapely_diffclip_right(
    pairs: pd.DataFrame,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    grid_size: int | float | None,
    rsuffix: str,
    geom_type: str | None,
    n_jobs: int,
) -> pd.DataFrame:
    agg_geoms_partial = functools.partial(_agg_geoms, grid_size=grid_size)

    pairs = pairs.rename(columns={"geometry": "geom_left", "geom_right": "geometry"})

    try:
        only_one = pairs.groupby("_overlay_index_right").transform("size") == 1
        one_hit = pairs[only_one].set_index("_overlay_index_right")[
            ["geom_left", "geometry"]
        ]
        many_hits = (
            pairs[~only_one]
            .groupby("_overlay_index_right")
            .agg(
                {
                    "geom_left": agg_geoms_partial,
                    "geometry": "first",
                }
            )
        )
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

    assert clip_right["geometry"].notna().all()
    assert clip_right["geom_left"].notna().all()

    clip_right["geometry"] = _try_difference(
        clip_right["geometry"].to_numpy(),
        clip_right["geom_left"].to_numpy(),
        grid_size=grid_size,
        geom_type=geom_type,
    )

    return clip_right.drop(columns="geom_left")


def _try_difference(
    left: np.ndarray,
    right: np.ndarray,
    grid_size: int | float | None,
    geom_type: str | None,
    n_jobs: int = 1,
) -> np.ndarray:
    """Try difference overlay, then make_valid and retry."""
    if n_jobs > 1 and len(left) / n_jobs > 10:
        try:
            return _run_overlay_joblib_threading(
                left,
                right,
                func=difference,
                n_jobs=n_jobs,
                grid_size=grid_size,
            )
        except GEOSException:
            left = make_valid_and_keep_geom_type(
                left, geom_type=geom_type, n_jobs=n_jobs
            )
            right = make_valid_and_keep_geom_type(
                right, geom_type=geom_type, n_jobs=n_jobs
            )
            left = left.loc[lambda x: x.index.isin(right.index)]
            right = right.loc[lambda x: x.index.isin(left.index)]

            return _run_overlay_joblib_threading(
                left.to_numpy(),
                right.to_numpy(),
                func=difference,
                n_jobs=n_jobs,
                grid_size=grid_size,
            )

    try:
        return difference(
            left,
            right,
            grid_size=grid_size,
        )
    except GEOSException:
        left = make_valid_and_keep_geom_type(left, geom_type, n_jobs=n_jobs)
        right = make_valid_and_keep_geom_type(right, geom_type, n_jobs=n_jobs)
        left = left.loc[lambda x: x.index.isin(right.index)]
        right = right.loc[lambda x: x.index.isin(left.index)]
        try:
            return difference(
                left.to_numpy(),
                right.to_numpy(),
                grid_size=grid_size,
            )
        except GEOSException as e:
            raise e.__class__(e, f"{grid_size=}", f"{left=}", f"{right=}") from e


def make_valid_and_keep_geom_type(
    geoms: np.ndarray, geom_type: str, n_jobs: int
) -> GeoSeries:
    """Make GeometryCollections into (Multi)Polygons, (Multi)LineStrings or (Multi)Points.

    Because GeometryCollections might appear after dissolving (unary_union).
    And this makes shapely difference/intersection fail.

    Args:
        geoms: Array of geometries.
        geom_type: geometry type to be kept.
        n_jobs: Number of treads.
    """
    geoms = GeoSeries(geoms)
    geoms.index = range(len(geoms))
    geoms.loc[:] = make_valid(geoms.to_numpy())
    geoms = geoms.explode(index_parts=False).pipe(to_single_geom_type, geom_type)
    only_one = geoms.groupby(level=0).transform("size") == 1
    one_hit = geoms[only_one]
    many_hits = geoms[~only_one].groupby(level=0).agg(unary_union)
    return pd.concat([one_hit, many_hits]).sort_index()


def _agg_geoms(g: np.ndarray, grid_size: int | float | None = None) -> Geometry:
    return make_valid(unary_union(g, grid_size=grid_size))
