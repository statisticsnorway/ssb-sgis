# %%
import re
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from geopandas.array import GeometryArray
from numpy.typing import NDArray
from shapely import extract_unique_points
from shapely import get_parts
from shapely.errors import GEOSException
from shapely.geometry import LineString

from .buffer_dissolve_explode import buff
from .buffer_dissolve_explode import dissexp
from .conversion import coordinate_array
from .conversion import to_gdf
from .duplicates import get_intersections
from .duplicates import update_geometries
from .general import clean_geoms
from .general import make_lines_between_points
from .general import sort_large_first
from .general import sort_small_first
from .general import to_lines
from .geometry_types import make_all_singlepart
from .geometry_types import to_single_geom_type
from .overlay import clean_overlay
from .polygon_operations import eliminate_by_longest
from .polygon_operations import get_cluster_mapper
from .polygon_operations import get_gaps
from .sfilter import sfilter_inverse
from .sfilter import sfilter_split

warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


PRECISION = 1e-3
BUFFER_RES = 50


def coverage_clean(
    gdf: GeoDataFrame,
    tolerance: int | float,
    duplicate_action: str = "fix",
    grid_sizes: tuple[None | int] = (None,),
    n_jobs: int = 1,
) -> GeoDataFrame:
    """Fix thin gaps, holes, slivers and double surfaces.

    The operations might raise GEOSExceptions, so it might be nessecary to set
    the 'grid_sizes' argument, it might also be a good idea to run coverage_clean
    twice to fill gaps resulting from these GEOSExceptions.

    Rules:
    - Holes (interiors) thinner than the tolerance are closed.
    - Gaps between polygons are filled if thinner than the tolerance.
    - Sliver polygons thinner than the tolerance are eliminated
    into the neighbor polygon with the longest shared border.
    - Double surfaces thinner than the tolerance are eliminated.
    If duplicate_action is "fix", thicker double surfaces will
    be updated.
    - Line and point geometries are removed with no warning.
    - MultiPolygons and GeometryCollections are exploded to Polygons.
    - Index is reset.

    Args:
        gdf: GeoDataFrame to be cleaned.
        tolerance: distance (usually meters) used as the minimum thickness
            for polygons to be eliminated. Any gap, hole, sliver or double
            surface that are empty after a negative buffer of tolerance / 2
            are eliminated into the neighbor with the longest shared border.
        duplicate_action: Either "fix", "error" or "ignore".
            If "fix" (default), double surfaces thicker than the
            tolerance will be updated from top to bottom (function update_geometries)
            and then dissolved into the neighbor polygon with the longest shared border.
            If "error", an Exception is raised if there are any double surfaces thicker
            than the tolerance. If "ignore", double surfaces are kept as is.
        grid_sizes: One or more grid_sizes used in overlay and dissolve operations that
            might raise a GEOSException. Defaults to (None,), meaning no grid_sizes.
        n_jobs: Number of threads.

    Returns:
        A GeoDataFrame with cleaned polygons.
    """
    if not len(gdf):
        return gdf

    _cleaning_checks(gdf, tolerance, duplicate_action)

    if not gdf.index.is_unique:
        gdf = gdf.reset_index(drop=True)

    gdf = make_all_singlepart(gdf).loc[
        lambda x: x.geom_type.isin(["Polygon", "MultiPolygon"])
    ]

    gdf = (
        clean_geoms(gdf)
        .pipe(make_all_singlepart)
        .loc[lambda x: x.geom_type.isin(["Polygon", "MultiPolygon"])]
    )

    try:
        gaps = get_gaps(gdf, include_interiors=True)
    except GEOSException:
        for i, grid_size in enumerate(grid_sizes):
            try:
                gaps = get_gaps(gdf, include_interiors=True, grid_size=grid_size)
                if grid_size:
                    # in order to not get more gaps
                    gaps.geometry = gaps.buffer(grid_size)
                break
            except GEOSException as e:
                if i == len(grid_sizes) - 1:
                    explore_geosexception(e, gdf)
                    raise e

    gaps["_was_gap"] = 1

    if duplicate_action == "ignore":
        double = GeoDataFrame({"geometry": []}, crs=gdf.crs)
        double["_double_idx"] = None
    else:
        double = get_intersections(gdf, n_jobs=n_jobs)
        double["_double_idx"] = range(len(double))

    gdf, slivers = split_out_slivers(gdf, tolerance)

    gdf["_poly_idx"] = range(len(gdf))

    thin_gaps_and_double = pd.concat([gaps, double]).loc[
        lambda x: (x.buffer(-tolerance / 2).is_empty)
    ]

    all_are_thin = double["_double_idx"].isin(thin_gaps_and_double["_double_idx"]).all()

    if not all_are_thin and duplicate_action == "fix":
        gdf, thin_gaps_and_double, slivers = _properly_fix_duplicates(
            gdf,
            double,
            slivers,
            thin_gaps_and_double,
            tolerance,
            n_jobs=n_jobs,
        )

    elif not all_are_thin and duplicate_action == "error":
        raise ValueError("Large double surfaces.")

    to_eliminate = pd.concat([thin_gaps_and_double, slivers], ignore_index=True)

    to_eliminate = to_eliminate.loc[lambda x: ~x.buffer(-PRECISION / 10).is_empty]

    to_eliminate = try_for_grid_size(
        split_by_neighbors,
        grid_sizes=grid_sizes,
        args=(to_eliminate, gdf),
        kwargs=dict(tolerance=tolerance),
    )

    to_eliminate["_eliminate_idx"] = range(len(to_eliminate))

    to_eliminate["_cluster"] = get_cluster_mapper(to_eliminate.buffer(PRECISION))

    gdf_geoms_idx = gdf[["_poly_idx", "geometry"]]

    poly_idx_mapper = clean_overlay(
        buff(
            to_eliminate[["_eliminate_idx", "geometry"]],
            tolerance,
            resolution=BUFFER_RES,
        ),
        gdf_geoms_idx,
        geom_type="polygon",
        n_jobs=n_jobs,
    )
    poly_idx_mapper["_area_per_poly"] = poly_idx_mapper.area
    poly_idx_mapper["_area_per_poly"] = poly_idx_mapper.groupby("_poly_idx")[
        "_area_per_poly"
    ].transform("sum")

    poly_idx_mapper: pd.Series = (
        poly_idx_mapper.sort_values("_area_per_poly", ascending=False)
        .drop_duplicates("_eliminate_idx")
        .set_index("_eliminate_idx")["_poly_idx"]
    )
    to_eliminate["_poly_idx"] = to_eliminate["_eliminate_idx"].map(poly_idx_mapper)
    isolated = to_eliminate[lambda x: x["_poly_idx"].isna()]
    intersecting = to_eliminate[lambda x: x["_poly_idx"].notna()]

    for i, grid_size in enumerate(grid_sizes):
        try:
            without_double = update_geometries(
                intersecting,
                geom_type="polygon",
                grid_size=grid_size,
                n_jobs=n_jobs,
            ).drop(columns=["_eliminate_idx", "_double_idx"])
            break
        except GEOSException as e:
            if i == len(grid_sizes) - 1:
                explore_geosexception(e, gdf, intersecting, isolated)
                raise e

    not_really_isolated = isolated[["geometry", "_eliminate_idx", "_cluster"]].merge(
        without_double.drop(columns=["geometry"]),
        on="_cluster",
        how="inner",
    )

    really_isolated = isolated.loc[
        lambda x: ~x["_eliminate_idx"].isin(not_really_isolated["_eliminate_idx"])
    ]

    is_gap = really_isolated["_was_gap"] == 1
    isolated_gaps = really_isolated.loc[is_gap, ["geometry"]].sjoin_nearest(
        gdf, max_distance=PRECISION
    )
    really_isolated = really_isolated[~is_gap]

    really_isolated["_poly_idx"] = (
        really_isolated["_cluster"] + gdf["_poly_idx"].max() + 1
    )

    cleaned = pd.concat(
        [
            gdf,
            without_double,
            not_really_isolated,
            really_isolated,
            isolated_gaps,
        ],
    ).drop(
        columns=[
            "_cluster",
            "_was_gap",
            "_eliminate_idx",
            "index_right",
            "_double_idx",
            "_area_per_poly",
        ],
        errors="ignore",
    )

    try:
        only_one = cleaned.groupby("_poly_idx").transform("size") == 1
        one_hit = cleaned[only_one].drop(columns="_poly_idx")
        many_hits = cleaned[~only_one]
    except IndexError:
        assert not cleaned["_poly_idx"].notna().any(), cleaned
        one_hit = cleaned[lambda x: x.index == min(x.index) - 1].drop(
            columns="_poly_idx", errors="ignore"
        )
        many_hits = cleaned

    for i, grid_size in enumerate(grid_sizes):
        try:
            many_hits = (
                dissexp(
                    many_hits,
                    by="_poly_idx",
                    aggfunc="first",
                    dropna=True,
                    grid_size=grid_size,
                    n_jobs=n_jobs,
                )
                .sort_index()
                .reset_index(drop=True)
            )
            break
        except GEOSException as e:
            if i == len(grid_sizes) - 1:
                explore_geosexception(e, gdf, without_double, isolated, really_isolated)
                raise e

    cleaned = pd.concat([many_hits, one_hit], ignore_index=True)

    gdf = gdf.drop(columns="_poly_idx")

    for i, grid_size in enumerate(grid_sizes):
        try:
            cleaned = clean_overlay(
                gdf,
                cleaned,
                how="update",
                geom_type="polygon",
                grid_size=grid_size,
                n_jobs=n_jobs,
            )
            break
        except GEOSException as e:
            if i == len(grid_sizes) - 1:
                explore_geosexception(
                    e,
                    gdf,
                    cleaned,
                    without_double,
                    isolated,
                    really_isolated,
                )
                raise e

    cleaned = sort_large_first(cleaned)

    # slivers on bottom
    cleaned = pd.concat(split_out_slivers(cleaned, tolerance))

    for i, grid_size in enumerate(grid_sizes):
        try:
            cleaned = update_geometries(
                cleaned,
                geom_type="polygon",
                grid_size=grid_size,
                n_jobs=n_jobs,
            )
            break
        except GEOSException as e:
            if i == len(grid_sizes) - 1:
                explore_geosexception(
                    e,
                    gdf,
                    cleaned,
                    without_double,
                    isolated,
                    really_isolated,
                )
                raise e

    # cleaned = _safe_simplify(cleaned, PRECISION)
    # cleaned.geometry = shapely.make_valid(cleaned.geometry)

    # TODO check why polygons dissappear in rare cases. For now, just add back the missing
    dissapeared_polygons = sfilter_inverse(gdf, cleaned.buffer(-PRECISION))
    cleaned = pd.concat([cleaned, dissapeared_polygons])

    return to_single_geom_type(cleaned, "polygon")


def _safe_simplify(gdf: GeoDataFrame, tolerance: float | int, **kwargs) -> GeoDataFrame:
    """Simplify only if the resulting area is no more than 1 percent larger.

    Because simplifying can result in holes being filled.
    """
    copied = gdf.copy()

    # copied.geometry = shapely.simplify(
    #     shapely.set_precision(copied.geometry.values, 1e-6), 1e-6
    # )
    copied.geometry = shapely.make_valid(
        shapely.set_precision(shapely.simplify(copied.geometry.values, 1e-6), 1e-6)
    )

    simplified = gdf.copy()
    rounded = gdf.copy()
    simplified.geometry = shapely.make_valid(
        shapely.simplify(simplified.geometry.values, 1e-6)
    )
    rounded.geometry = shapely.make_valid(
        shapely.set_precision(rounded.geometry.values, 1e-6)
    )

    explore(gdf, copied, rounded, simplified)

    return copied.dropna()

    length_then = gdf.length
    copied = gdf.copy()
    copied.geometry = shapely.make_valid(
        shapely.simplify(copied.geometry.values, tolerance=tolerance)
    )
    filt = (copied.area > length_then * 1.01) | (copied.geometry.is_empty)
    copied.loc[filt, copied._geometry_column_name] = gdf.loc[
        filt, copied._geometry_column_name
    ]

    return copied


def _remove_interior_slivers(gdf: GeoDataFrame, tolerance: int | float) -> GeoDataFrame:
    gdf, slivers = split_out_slivers(gdf, tolerance)
    slivers["_idx"] = range(len(slivers))
    without_thick = clean_overlay(
        to_lines(slivers), buff(gdf, PRECISION), how="difference"
    )
    return pd.concat(
        [
            gdf,
            slivers[lambda x: x["_idx"].isin(without_thick["_idx"])].drop(
                columns="_idx"
            ),
        ]
    )


def remove_spikes(
    gdf: GeoDataFrame, tolerance: int | float, n_jobs: int = 1
) -> GeoDataFrame:
    """Remove thin spikes from polygons.

    Args:
        gdf: A GeoDataFrame.
        tolerance: Spike tolerance.
        n_jobs: Number of threads.

    Returns:
        A GeoDataFrame.
    """
    return clean_overlay(
        gdf, gdf[["geometry"]], how="intersection", grid_size=tolerance, n_jobs=n_jobs
    )


def _properly_fix_duplicates(
    gdf: GeoDataFrame,
    double: GeoDataFrame,
    slivers: GeoDataFrame,
    thin_gaps_and_double: GeoDataFrame,
    tolerance: int | float,
    n_jobs: int,
) -> GeoDataFrame:
    gdf = _dissolve_thick_double_and_update(gdf, double, thin_gaps_and_double, n_jobs)
    gdf, more_slivers = split_out_slivers(gdf, tolerance)
    slivers = pd.concat([slivers, more_slivers], ignore_index=True)
    gaps = get_gaps(gdf, include_interiors=True)
    gaps["_was_gap"] = 1
    assert "_double_idx" not in gaps
    double = get_intersections(gdf)
    double["_double_idx"] = range(len(double))
    thin_gaps_and_double = pd.concat([gaps, double], ignore_index=True).loc[
        lambda x: x.buffer(-tolerance / 2).is_empty
    ]

    return gdf, thin_gaps_and_double, slivers


def _dissolve_thick_double_and_update(
    gdf: GeoDataFrame, double: GeoDataFrame, thin_double: GeoDataFrame, n_jobs: int
) -> GeoDataFrame:
    large = (
        double.loc[~double["_double_idx"].isin(thin_double["_double_idx"])].drop(
            columns="_double_idx"
        )
        # .pipe(sort_large_first)
        # .sort_values("_poly_idx")
        .pipe(update_geometries, geom_type="polygon", n_jobs=n_jobs)
    )
    return (
        clean_overlay(gdf, large, how="update", geom_type="polygon", n_jobs=n_jobs)
        # .pipe(sort_large_first)
        # .sort_values("_poly_idx")
        .pipe(update_geometries, geom_type="polygon", n_jobs=n_jobs)
    )


def _cleaning_checks(
    gdf: GeoDataFrame, tolerance: int | float, duplicate_action: bool
) -> GeoDataFrame:  # , spike_action):
    if not len(gdf) or not tolerance:
        return gdf
    if tolerance < PRECISION:
        raise ValueError(
            f"'tolerance' must be larger than {PRECISION} to avoid "
            "problems with floating point precision."
        )
    if duplicate_action not in ["fix", "error", "ignore"]:
        raise ValueError("duplicate_action must be 'fix', 'error' or 'ignore'")


def split_out_slivers(
    gdf: GeoDataFrame | GeoSeries, tolerance: float | int
) -> tuple[GeoDataFrame, GeoDataFrame] | tuple[GeoSeries, GeoSeries]:
    is_sliver = gdf.buffer(-tolerance / 2).is_empty
    slivers = gdf.loc[is_sliver]
    gdf = gdf.loc[~is_sliver]
    slivers, isolated = sfilter_split(slivers, gdf.buffer(PRECISION))
    gdf = pd.concat([gdf, isolated])
    return gdf, slivers


def try_for_grid_size(
    func: Callable,
    grid_sizes: tuple[None, float | int],
    args: tuple | None = None,
    kwargs: dict | None = None,
) -> Any:
    args = args or ()
    kwargs = kwargs or {}
    for i, grid_size in enumerate(grid_sizes):
        try:
            return func(*args, grid_size=grid_size, **kwargs)
        except GEOSException as e:
            if i == len(grid_sizes) - 1:
                raise e


def split_and_eliminate_by_longest(
    gdf: GeoDataFrame | list[GeoDataFrame],
    to_eliminate: GeoDataFrame,
    tolerance: int | float,
    grid_sizes: tuple[None | float | int] = (None,),
    n_jobs: int = 1,
    **kwargs,
) -> GeoDataFrame | tuple[GeoDataFrame]:
    if not len(to_eliminate):
        return gdf

    if not isinstance(gdf, (GeoDataFrame, GeoSeries)):
        as_gdf = pd.concat(gdf, ignore_index=True)
    else:
        as_gdf = gdf

    splitted = try_for_grid_size(
        split_by_neighbors,
        grid_sizes=grid_sizes,
        args=(to_eliminate, as_gdf, tolerance),
    ).pipe(sort_small_first)

    splitted = try_for_grid_size(
        update_geometries,
        grid_sizes=grid_sizes,
        args=(splitted,),
        kwargs=dict(geom_type="polygon", n_jobs=n_jobs),
    )

    gdf = try_for_grid_size(
        eliminate_by_longest,
        grid_sizes=grid_sizes,
        args=(
            gdf,
            splitted,
        ),
        kwargs=kwargs | {"n_jobs": n_jobs},
    )

    if not isinstance(gdf, (GeoDataFrame, GeoSeries)):
        as_gdf = pd.concat(gdf, ignore_index=True)
    else:
        as_gdf = gdf

    missing = try_for_grid_size(
        clean_overlay,
        grid_sizes=grid_sizes,
        args=(
            to_eliminate,
            as_gdf,
        ),
        kwargs=dict(
            how="difference",
            geom_type="polygon",
            n_jobs=n_jobs,
        ),
    ).pipe(lambda x: dissexp(x, n_jobs=n_jobs))

    return try_for_grid_size(
        eliminate_by_longest,
        grid_sizes=grid_sizes,
        args=(gdf, missing),
        kwargs=kwargs | {"n_jobs": n_jobs},
    )


def split_by_neighbors(
    df: GeoDataFrame,
    split_by: GeoDataFrame,
    tolerance: int | float,
    grid_size: float | int | None = None,
) -> GeoDataFrame:
    if not len(df):
        return df

    split_by = split_by.copy()
    split_by.geometry = shapely.simplify(split_by.geometry, tolerance)

    intersecting_lines = (
        clean_overlay(
            to_lines(split_by),
            buff(df, tolerance),
            how="intersection",
            grid_size=grid_size,
        )
        .pipe(get_line_segments)
        .reset_index(drop=True)
    )

    endpoints = intersecting_lines.boundary.explode(index_parts=False)

    extended_lines = GeoDataFrame(
        {
            "geometry": extend_lines(
                endpoints.loc[lambda x: ~x.index.duplicated(keep="first")].values,
                endpoints.loc[lambda x: ~x.index.duplicated(keep="last")].values,
                distance=tolerance * 3,
            )
        },
        crs=df.crs,
    )

    buffered = buff(extended_lines, tolerance, single_sided=True)

    return clean_overlay(df, buffered, how="identity", grid_size=grid_size)


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


# def make_lines_between_points(
#     arr1: NDArray[Point], arr2: NDArray[Point]
# ) -> NDArray[LineString]:
#     if arr1.shape != arr2.shape:
#         raise ValueError(
#             f"Arrays must have equal shape. Got {arr1.shape} and {arr2.shape}"
#         )
#     coords: pd.DataFrame = pd.concat(
#         [
#             pd.DataFrame(get_coordinates(arr1), columns=["x", "y"]),
#             pd.DataFrame(get_coordinates(arr2), columns=["x", "y"]),
#         ]
#     ).sort_index()

#     return linestrings(coords.values, indices=coords.index)


def get_line_segments(lines: GeoDataFrame | GeoSeries) -> GeoDataFrame:
    assert lines.index.is_unique
    if isinstance(lines, GeoDataFrame):
        geom_col = lines._geometry_column_name
        multipoints = lines.assign(
            **{geom_col: extract_unique_points(lines.geometry.values)}
        )
        segments = multipoints_to_line_segments(multipoints.geometry)
        return segments.join(lines.drop(columns=geom_col))

    multipoints = GeoSeries(extract_unique_points(lines.values), index=lines.index)

    return multipoints_to_line_segments(multipoints)


def multipoints_to_line_segments(multipoints: GeoSeries) -> GeoDataFrame:
    if not len(multipoints):
        return GeoDataFrame({"geometry": multipoints}, index=multipoints.index)

    try:
        crs = multipoints.crs
    except AttributeError:
        crs = None

    try:
        point_df = multipoints.explode(index_parts=False)
    except AttributeError:
        points, indices = get_parts(multipoints, return_index=True)
        if isinstance(multipoints.index, pd.MultiIndex):
            indices = pd.MultiIndex.from_arrays(indices, names=multipoints.index.names)

        point_df = pd.DataFrame({"geometry": GeometryArray(points)}, index=indices)

    try:
        point_df = point_df.to_frame("geometry")
    except AttributeError:
        pass

    point_df["next"] = point_df.groupby(level=0)["geometry"].shift(-1)

    first_points = point_df.loc[lambda x: ~x.index.duplicated(), "geometry"]
    is_last_point = point_df["next"].isna()

    point_df.loc[is_last_point, "next"] = first_points
    assert point_df["next"].notna().all()

    point_df["geometry"] = [
        LineString([x1, x2])
        for x1, x2 in zip(point_df["geometry"], point_df["next"], strict=False)
    ]
    return GeoDataFrame(point_df.drop(columns=["next"]), geometry="geometry", crs=crs)


def points_to_line_segments(points: GeoDataFrame) -> GeoDataFrame:
    points = points.copy()
    points["next"] = points.groupby(level=0)["geometry"].shift(-1)

    first_points = points.loc[lambda x: ~x.index.duplicated(), "geometry"]
    is_last_point = points["next"].isna()

    points.loc[is_last_point, "next"] = first_points
    assert points["next"].notna().all()

    points["geometry"] = [
        LineString([x1, x2])
        for x1, x2 in zip(points["geometry"], points["next"], strict=False)
    ]
    return GeoDataFrame(
        points.drop(columns=["next"]), geometry="geometry", crs=points.crs
    )


def explore_geosexception(
    e: GEOSException, *gdfs: GeoDataFrame, logger: Any | None = None
) -> None:
    """Extract the coordinates of a GEOSException and show in map.

    Args:
        e: The exception thrown by a GEOS operation, which potentially contains coordinates information.
        *gdfs: One or more GeoDataFrames to display for context in the map.
        logger: An optional logger to log the error with visualization. If None, uses standard output.

    """
    from ..maps.maps import Explore
    from ..maps.maps import explore

    pattern = r"(\d+\.\d+)\s+(\d+\.\d+)"

    matches = re.findall(pattern, str(e))
    coords_in_error_message = [(float(match[0]), float(match[1])) for match in matches]
    exception_point = to_gdf(coords_in_error_message, crs=gdfs[0].crs)
    if len(exception_point):
        exception_point["wkt"] = exception_point.to_wkt()
        if logger:
            logger.error(
                e, Explore(exception_point, *gdfs, mask=exception_point.buffer(100))
            )
        else:
            explore(exception_point, *gdfs, mask=exception_point.buffer(100))
    else:
        if logger:
            logger.error(e, Explore(*gdfs))
        else:
            explore(*gdfs)


####
####
####
import warnings
from collections.abc import Callable

from geopandas import GeoDataFrame
from geopandas import GeoSeries
from numpy.typing import NDArray
from shapely.errors import GEOSException
from shapely.geometry import LineString

# from .general import sort_large_first as _sort_large_first

warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


from ..maps.maps import explore

# def explore(*args, **kwargs):
#     pass


PRECISION = 1e-3
BUFFER_RES = 50


# def snap_to_mask(
#     gdf: GeoDataFrame, tolerance: int | float, mask: GeoDataFrame | GeoSeries | Geometry
# ):
#     return snap_polygons(
#         gdf,
#         mask=mask,
#         tolerance=tolerance,
#         snap_to_nodes=False,
#     )


import sys
from pathlib import Path

import geopandas as gpd
from shapely import minimum_rotated_rectangle

src = str(Path(__file__).parent).strip("tests") + "src"


sys.path.insert(0, src)

import sgis as sg


def test_clean_dissappearing_polygon():
    AREA_SHOULD_BE = 104

    with open(Path(__file__).parent / "testdata/dissolve_error.txt") as f:
        df = sg.to_gdf(f.readlines(), 25833)

    dissappears = sg.to_gdf([5.95201, 62.41451], 4326).to_crs(25833).buffer(100)
    df_problem_area = sg.sfilter(df, dissappears.buffer(0.1))

    assert len(df_problem_area) == 3

    assert (area := int(df_problem_area.area.sum())) == AREA_SHOULD_BE, area

    cleaned = sg.coverage_clean(df, 0.1, duplicate_action="fix")

    cleaned_problem_area = sg.sfilter(cleaned, dissappears.buffer(0.1))

    sg.explore(cleaned, cleaned_problem_area, dissappears, df_problem_area)
    assert (area := int(cleaned_problem_area.area.sum())) == AREA_SHOULD_BE, area

    cleaned_dissolved_problem_area = sg.sfilter(
        sg.dissexp(cleaned), dissappears.buffer(0.1)
    )

    # cleaned_dissolved_problem_area.to_parquet(
    #     "c:/users/ort/downloads/cleaned_dissolved_problem_area.parquet"
    # )

    assert len(cleaned_dissolved_problem_area) == 1, (
        sg.explore(
            cleaned_dissolved_problem_area.assign(
                col=lambda x: [str(i) for i in range(len(x))]
            ),
            "col",
        ),
        cleaned_dissolved_problem_area,
    )

    assert (
        area := int(cleaned_dissolved_problem_area.area.sum())
    ) == AREA_SHOULD_BE, area

    snapped = sg.snap_polygons(df, 0.1)

    snapped_problem_area = sg.sfilter(snapped, dissappears.buffer(0.1))

    sg.explore(snapped, snapped_problem_area, dissappears, df_problem_area)
    assert (area := int(snapped_problem_area.area.sum())) == AREA_SHOULD_BE, area

    snapped_dissolved_problem_area = sg.sfilter(
        sg.dissexp(snapped), dissappears.buffer(0.1)
    )

    # snapped_dissolved_problem_area.to_parquet(
    #     "c:/users/ort/downloads/snapped_dissolved_problem_area.parquet"
    # )

    assert len(snapped_dissolved_problem_area) == 1, (
        sg.explore(
            snapped_dissolved_problem_area.assign(
                col=lambda x: [str(i) for i in range(len(x))]
            ),
            "col",
        ),
        snapped_dissolved_problem_area,
    )

    assert (
        area := int(snapped_dissolved_problem_area.area.sum())
    ) == AREA_SHOULD_BE, area


def test_clean_1144():
    df = gpd.read_parquet(
        Path(__file__).parent / "testdata" / "snap_problem_area_1144.parquet"
    )

    bbox = sg.to_gdf(minimum_rotated_rectangle(df.unary_union), df.crs)

    kommune_utenhav = gpd.read_parquet(
        Path(__file__).parent / "testdata" / "kommune_utenhav_1144_2023.parquet"
    )

    kommune_utenhav = sg.clean_clip(kommune_utenhav, bbox, geom_type="polygon")

    assert sg.get_intersections(df).dissolve().area.sum() == 0
    assert int(df.area.sum()) == 154240, df.area.sum()

    cols = [
        "ARGRUNNF",
        "ARJORDBR",
        "ARKARTSTD",
        "ARSKOGBON",
        "ARTRESLAG",
        "ARTYPE",
        "ARVEGET",
        "ASTSSB",
        "df_idx",
        "geometry",
        "kilde",
    ]

    df["df_idx"] = range(len(df))

    for tolerance in [2, 1, 5]:
        print("\ntolerance")
        print(tolerance)
        # cleaned = sg.coverage_clean(df, tolerance, pre_dissolve_func=_buff).pipe(
        #     sg.clean_clip, df, geom_type="polygon"

        # )

        thick_df_indices = df.loc[
            lambda x: ~x.buffer(-tolerance / 2).is_empty, "df_idx"
        ]

        cleaned = sg.coverage_clean(df, tolerance)

        # cleaned = sg.coverage_clean(
        #     sg.sort_large_first(df), tolerance, mask=kommune_utenhav
        # ).pipe(sg.snap_polygons, 0.1, mask=kommune_utenhav)

        # allow edge cases
        cleaned_clipped = sg.clean_clip(cleaned, bbox.buffer(-tolerance * 1.1))

        gaps = sg.get_gaps(cleaned_clipped)
        double = sg.get_intersections(cleaned_clipped)
        missing = get_missing(
            sg.clean_clip(df, bbox.buffer(-tolerance * 1.1)), cleaned_clipped
        )

        print(
            "cleaned",
            gaps.area.sum(),
            double.area.sum(),
            missing.area.sum(),
        )

        assert thick_df_indices.isin(cleaned_clipped["df_idx"]).all(), sg.explore(
            df,
            cleaned,
            missing_polygons=df[
                (df["df_idx"].isin(thick_df_indices))
                & (~df["df_idx"].isin(cleaned_clipped["df_idx"]))
            ],
        )

        snapped_to_mask = sg.snap_to_mask(
            sg.sort_large_first(df), tolerance, mask=kommune_utenhav
        )

        gaps = sg.get_gaps(snapped_to_mask)
        double = sg.get_intersections(snapped_to_mask)
        missing = sg.clean_overlay(
            sg.clean_clip(df, bbox.buffer(-tolerance * 1.1)),
            snapped_to_mask,
            how="difference",
            geom_type="polygon",
        )

        if missing.area.sum() > 1:
            sg.explore(
                missing,
                gaps,
                double,
                snapped_to_mask,
                df,
                kommune_utenhav,
            )

        assert double.area.sum() < 1e-4, double.area.sum()
        assert missing.area.sum() < 1e-4, missing.area.sum()
        assert gaps.area.sum() < 1e-4, gaps.area.sum()

        # assert thick_df_indices.isin(snapped_to_mask["df_idx"]).all(), sg.explore(
        #     df,
        #     snapped_to_mask,
        #     missing_polygons=df[
        #         (df["df_idx"].isin(thick_df_indices))
        #         & (~df["df_idx"].isin(snapped_to_mask["df_idx"]))
        #     ],
        # )

        print(
            "snapped_to_mask",
            gaps.area.sum(),
            double.area.sum(),
            missing.area.sum(),
        )

        snapped = sg.snap_polygons(
            sg.sort_large_first(df), tolerance, mask=kommune_utenhav
        )

        gaps = sg.get_gaps(snapped)
        double = sg.get_intersections(snapped)

        missing = get_missing(sg.clean_clip(df, bbox.buffer(-tolerance * 1.1)), snapped)

        sg.explore(
            missing,
            gaps,
            double,
            snapped,
            df,
            kommune_utenhav,
        )

        # assert thick_df_indices.isin(snapped["df_idx"]).all(), sg.explore(
        #     df,
        #     snapped_to_mask,
        #     missing_polygons=df[
        #         (df["df_idx"].isin(thick_df_indices))
        #         & (~df["df_idx"].isin(snapped["df_idx"]))
        #     ],
        # )
        assert double.area.sum() < 1e-4, double.area.sum()
        assert missing.area.sum() < 1e-4, missing.area.sum()
        assert gaps.area.sum() < 1e-4, gaps.area.sum()

        print(
            "snapped",
            gaps.area.sum(),
            double.area.sum(),
            missing.area.sum(),
        )

        cleaned_and_snapped_to_mask = sg.snap_to_mask(
            cleaned, tolerance, mask=kommune_utenhav
        )

        # cleaned = sg.coverage_clean(
        #     sg.sort_large_first(df), tolerance, mask=kommune_utenhav
        # ).pipe(sg.snap_polygons, 0.1, mask=kommune_utenhav)

        # allow edge cases
        cleaned_and_snapped_to_mask = sg.clean_clip(
            cleaned_and_snapped_to_mask, bbox.buffer(-tolerance * 1.1)
        )

        gaps = sg.get_gaps(cleaned_and_snapped_to_mask)
        double = sg.get_intersections(cleaned_and_snapped_to_mask)
        missing = get_missing(
            sg.clean_clip(df, bbox.buffer(-tolerance * 1.1)),
            cleaned_and_snapped_to_mask,
        )

        print(
            "cleaned_and_snapped_to_mask",
            gaps.area.sum(),
            double.area.sum(),
            missing.area.sum(),
        )

        sg.explore(
            df,  # =cleaned.assign(wkt=lambda x: x.geometry.to_wkt()),
            snapped_to_mask,  # =cleaned.assign(wkt=lambda x: x.geometry.to_wkt()),
            cleaned,  # =cleaned.assign(wkt=lambda x: x.geometry.to_wkt()),
            double,  # =double.assign(wkt=lambda x: x.to_wkt()),
            gaps,  # =gaps.assign(wkt=lambda x: x.to_wkt()),
            missing,
            kommune_utenhav,
            thick_missing=missing[~missing.buffer(-0.01).is_empty].assign(
                area=lambda x: x.area
            ),
        )

        assert list(sorted(cleaned.columns)) == cols, list(sorted(cleaned.columns))
        assert thick_df_indices.isin(
            cleaned_and_snapped_to_mask["df_idx"]
        ).all(), sg.explore(
            df,
            snapped_to_mask,
            missing_polygons=df[
                (df["df_idx"].isin(thick_df_indices))
                & (~df["df_idx"].isin(cleaned_and_snapped_to_mask["df_idx"]))
            ],
        )

        assert double.area.sum() < 1e-4, double.area.sum()
        assert gaps.area.sum() < 1e-3, (
            gaps.area.sum(),
            gaps.area.max(),
            gaps.area,
            gaps,
        )
        assert missing.area.sum() < 1e-3 or (missing.area.sum() + 1e-10) < 1e-3, (
            missing.area.sum(),
            missing.area.sort_values(),
        )

        # assert int(cleaned.area.sum()) == 154240, (
        #     cleaned.area.sum(),
        #     f"tolerance: {tolerance}",
        # )

        assert sg.get_geom_type(cleaned) == "polygon", sg.get_geom_type(cleaned)

        print("cleaning cleaned")
        cleaned_and_snapped_to_mask = sg.snap_polygons(
            cleaned, tolerance, mask=kommune_utenhav
        )

        gaps = sg.get_gaps(cleaned_and_snapped_to_mask)
        double = sg.get_intersections(cleaned_and_snapped_to_mask)
        missing = get_missing(sg.clean_clip(df, bbox.buffer(-tolerance * 1.1)), cleaned)

        sg.explore(
            df,  # =cleaned.assign(wkt=lambda x: x.geometry.to_wkt()),
            cleaned,  # =cleaned.assign(wkt=lambda x: x.geometry.to_wkt()),
            double,  # =double.assign(wkt=lambda x: x.to_wkt()),
            gaps,  # =gaps.assign(wkt=lambda x: x.to_wkt()),
            missing,
            kommune_utenhav,
            thick_missing=missing[~missing.buffer(-0.01).is_empty].assign(
                area=lambda x: x.area
            ),
            # mask=sg.to_gdf([5.36750884, 59.00830496], 4326).to_crs(25833).buffer(10),
            # browser=True,
            max_zoom=50,
        )

        assert (
            list(sorted(cleaned_and_snapped_to_mask.columns)) == cols
        ), cleaned_and_snapped_to_mask.columns

        assert double.area.sum() < 1e-3, double.area.sum()
        assert gaps.area.sum() < 1e-3, gaps.area.sum()
        assert missing.area.sum() < 1e-3 or (missing.area.sum() + 1e-10), (
            missing.area.sum(),
            missing.area.sort_values(),
        )

        # assert int(cleaned_and_snapped_to_mask.area.sum()) == 154240, (
        #     cleaned_and_snapped_to_mask.area.sum(),
        #     f"tolerance: {tolerance}",
        # )

        assert (
            sg.get_geom_type(cleaned_and_snapped_to_mask) == "polygon"
        ), sg.get_geom_type(cleaned_and_snapped_to_mask)

        # cleaned = sg.coverage_clean(df, tolerance)

        # gaps = sg.get_gaps(cleaned)
        # double = sg.get_intersections(cleaned)
        # missing = sg.clean_overlay(df, cleaned, how="difference", geom_type="polygon")

        # assert list(sorted(cleaned.columns)) == cols, list(sorted(cleaned.columns))

        # assert double.area.sum() < 1e-4, double.area.sum()
        # assert gaps.area.sum() < 1e-3, (
        #     gaps.area.sum(),
        #     gaps.area.max(),
        #     gaps.area,
        #     gaps,
        # )
        # assert missing.area.sum() < 1e-3, (
        #     missing.area.sum(),
        #     missing.area.sort_values(),
        # )


def get_missing(df, other):
    return (
        sg.clean_overlay(df, other, how="difference", geom_type="polygon")
        .pipe(sg.buff, -0.0001)
        .pipe(sg.clean_overlay, other, how="difference", geom_type="polygon")
    )


def test_clean():
    df = gpd.read_parquet(Path(__file__).parent / "testdata" / "polygon_snap.parquet")

    bbox = sg.to_gdf(minimum_rotated_rectangle(df.unary_union), df.crs)

    kommune_utenhav = gpd.read_parquet(
        Path(__file__).parent / "testdata" / "kommune_utenhav_5435_2023.parquet"
    )
    kommune_utenhav = sg.clean_clip(kommune_utenhav, bbox, geom_type="polygon")

    holes = sg.to_gdf(
        [
            "POINT (905200 7878700)",
            "POINT (905250 7878780)",
            "POINT (905242.961 7878773.758)",
        ],
        25833,
    ).pipe(sg.buff, 3)

    df = sg.clean_overlay(df, holes, how="difference")

    mask = sg.close_all_holes(sg.dissexp_by_cluster(df)).dissolve()

    for tolerance in [5, 10]:
        print("tolerance:", tolerance)

        # from shapely import segmentize

        # df.geometry = segmentize(df.geometry, tolerance)

        snapped = sg.coverage_clean(df, tolerance).pipe(sg.coverage_clean, tolerance)
        assert sg.get_geom_type(snapped) == "polygon", sg.get_geom_type(snapped)

        double = sg.get_intersections(snapped).loc[lambda x: ~x.buffer(-1e-9).is_empty]
        gaps = sg.get_gaps(snapped).loc[lambda x: ~x.buffer(-1e-9).is_empty]
        missing = get_missing(df, snapped)

        print(double.area.sum(), missing.area.sum(), gaps.area.sum())

        sg.explore(
            df,
            snapped,
            double,
            missing,
            gaps,
        )

        assert (a := max(list(double.area) + [0])) < 1e-4, a
        assert (a := max(list(missing.area) + [0])) < 1e-4, a
        assert (a := max(list(gaps.area) + [0])) < 1e-4, a

        snapped = sg.snap_polygons(
            df, tolerance, mask=mask.buffer(0.1, resolution=1, join_style=2)
        )
        assert sg.get_geom_type(snapped) == "polygon", sg.get_geom_type(snapped)

        continue

        double = sg.get_intersections(snapped).loc[lambda x: ~x.buffer(-1e-9).is_empty]
        gaps = sg.get_gaps(snapped).loc[lambda x: ~x.buffer(-1e-9).is_empty]
        missing = get_missing(df, snapped)

        print(double.area.sum(), missing.area.sum(), gaps.area.sum())

        sg.explore(
            df,
            snapped,
            double,
            missing,
            gaps,
        )

        assert (a := max(list(gaps.area) + [0])) < 1e-4, a
        assert (a := max(list(double.area) + [0])) < 1e-4, a
        assert (a := max(list(missing.area) + [0])) < 1e-4, a

        # snapped = sg.snap_to_mask(
        #     df, tolerance, mask=mask.buffer(0.1, resolution=1, join_style=2)
        # )

        # double = sg.get_intersections(snapped)
        # missing = sg.clean_overlay(df, snapped, how="difference", geom_type="polygon")
        # gaps = sg.get_gaps(snapped)

        # print(double.area.sum(), missing.area.sum(), gaps.area.sum())

        # sg.explore(
        #     df,

        #     snapped,
        #     double,

        #     missing,
        # )

        snapped_twice = sg.snap_polygons(snapped, tolerance)
        assert sg.get_geom_type(snapped_twice) == "polygon", sg.get_geom_type(
            snapped_twice
        )

        double = sg.get_intersections(snapped_twice)
        missing = get_missing(df, snapped_twice)

        sg.explore(
            df,
            snapped,
            snapped_twice,
            double,
            missing,
        )

        # assert (a := max(list(double.area) + [0])) < 1e-4, a
        # assert (a := max(list(missing.area) + [0])) < 1e-4, a

        snapped = sg.snap_polygons(
            df, tolerance, mask=mask.buffer(0.1, resolution=1, join_style=2)
        )

        sg.explore(snapped.assign(idx=lambda x: [str(i) for i in range(len(x))]), "idx")

        gaps = sg.get_gaps(snapped)
        double = sg.get_intersections(snapped)
        missing = get_missing(df, snapped)

        sg.explore(
            df,
            snapped,
            gaps,
            double,
            missing,
        )

        sg.explore(snapped.assign(idx=lambda x: [str(i) for i in range(len(x))]), "idx")

        assert sg.get_geom_type(snapped) == "polygon", sg.get_geom_type(snapped)
        assert (a := max(list(gaps.area) + [0])) < 1e-4, a
        assert (a := max(list(double.area) + [0])) < 1e-4, a
        assert (a := max(list(missing.area) + [0])) < 1e-4, a

        snapped_to_snapped = sg.snap_to_mask(snapped, tolerance, mask=snapped)

        sg.explore(
            df,
            snapped,
            snapped_to_snapped,
        )
        sg.explore(
            snapped.assign(idx=lambda x: [str(i) for i in range(len(x))]),
            "idx",
        )

        assert sg.get_geom_type(snapped_to_snapped) == "polygon", sg.get_geom_type(
            snapped_to_snapped
        )

        # assert (nums1 := [round(num, 3) for num in sorted(snapped.area)]) == (
        #     nums2 := [round(num, 3) for num in sorted(snapped_to_snapped.area)]
        # ), (nums1, nums2)


def not_test_spikes():
    from shapely.geometry import Polygon

    factor = 10000

    sliver = sg.to_gdf(
        Polygon(
            [
                (0, 0),
                (0.1 * factor, 1 * factor),
                (0, 2 * factor),
                (-0.1 * factor, 1 * factor),
            ]
        )
    ).assign(what="sliver", num=1)
    poly_with_spike = sg.to_gdf(
        Polygon(
            [
                (0 * factor, 0 * factor),
                (-0.1 * factor, 1 * factor),
                (0 * factor, 2 * factor),
                (-0.99 * factor, 2 * factor),
                (-0.99 * factor, 1.5 * factor),
                (-1.01 * factor, 1.5 * factor),
                (-1.01 * factor, 2 * factor),
                (-1.51 * factor, 2 * factor),
                (-1.51 * factor, 1.7 * factor),
                (-1.52 * factor, 2 * factor),
                (-2 * factor, 2 * factor),
                (-1 * factor, 1 * factor),
            ],
            holes=[
                (
                    [
                        (-0.5 * factor, 1.25 * factor),
                        (-0.5 * factor, 1.65 * factor),
                        (-0.49 * factor, 1.65 * factor),
                        (-0.49 * factor, 1.25 * factor),
                    ]
                ),
            ],
        )
    ).assign(what="small", num=2)
    poly_filling_the_spike = sg.to_gdf(
        Polygon(
            [
                (0, 2 * factor),
                (-0.99 * factor, 2 * factor),
                (-0.99 * factor, 1.5 * factor),
                (-1.01 * factor, 1.5 * factor),
                (-1.01 * factor, 2 * factor),
                (-2 * factor, 2 * factor),
                (-2 * factor, 6 * factor),
                (0, 6 * factor),
                (0, 2 * factor),
            ],
        )
    ).assign(what="small", num=2)

    df = pd.concat([sliver, poly_with_spike, poly_filling_the_spike])
    holes = sg.buff(
        sg.to_gdf([(-0.84 * factor, 3 * factor), (-0.84 * factor, 4.4 * factor)]),
        [0.4 * factor, 0.3 * factor],
    )
    df = sg.clean_overlay(df, holes, how="update")
    df.crs = 25833

    tolerance = 0.09 * factor

    snapped = sg.snap_polygons(df, tolerance)
    spikes_removed = sg.remove_spikes(df, tolerance)
    spikes_fixed = sg.split_spiky_polygons(df, tolerance)
    fixed_and_cleaned = sg.coverage_clean(
        spikes_fixed, tolerance  # , pre_dissolve_func=_buff
    )  # .pipe(sg.remove_spikes, tolerance / 100)

    if __name__ == "__main__":
        sg.explore(
            fixed_and_cleaned=fixed_and_cleaned,
            snapped=snapped,
            spikes_removed=spikes_removed,
            # spikes_fixed=spikes_fixed,
            df=df,
        )

    def is_close_enough(num1, num2):
        if num1 >= num2 - 1e-3 and num1 <= num2 + 1e-3:
            return True
        return False

    area_should_be = [
        725264293.6535025,
        20000000.0,
        190000000.0,
        48285369.993336275,
        26450336.353161283,
    ]
    print(list(fixed_and_cleaned.area))
    for area1, area2 in zip(
        sorted(fixed_and_cleaned.area),
        sorted(area_should_be),
        strict=False,
    ):
        assert is_close_enough(area1, area2), (area1, area2)

    length_should_be = [
        163423.91054766334,
        40199.502484483564,
        68384.02248970368,
        24882.8908851665,
        18541.01966249684,
    ]

    print(list(fixed_and_cleaned.length))
    for length1, length2 in zip(
        sorted(fixed_and_cleaned.length),
        sorted(length_should_be),
        strict=False,
    ):
        assert is_close_enough(length1, length2), (length1, length2)

    # cleaned = sg.coverage_clean(df, tolerance)
    # if __name__ == "__main__":
    #     sg.explore(
    #         cleaned=cleaned,
    #         df=df,
    #     )

    # assert (area := sorted([round(x, 3) for x in cleaned.area])) == sorted(

    #     [

    #         7.225,
    #         1.89,
    #         0.503,
    #         0.283,
    #         0.2,
    #     ]
    # ), area
    # assert (length := sorted([round(x, 3) for x in cleaned.length])) == sorted(
    #     [
    #         17.398,
    #         7.838,
    #         2.513,
    #         1.885,

    #         4.02,
    #     ]

    # ), length


def main():
    test_clean()
    test_clean_1144()
    test_clean_dissappearing_polygon()

    not_test_spikes()


if __name__ == "__main__":

    # cProfile.run("main()", sort="cumtime")

    main()
