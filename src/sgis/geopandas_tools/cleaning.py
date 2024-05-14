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
from shapely import get_coordinates
from shapely import get_parts
from shapely import linestrings
from shapely.errors import GEOSException
from shapely.geometry import LineString
from shapely.geometry import Point

from .buffer_dissolve_explode import buff
from .buffer_dissolve_explode import dissexp
from .conversion import coordinate_array
from .conversion import to_gdf
from .duplicates import get_intersections
from .duplicates import update_geometries

# from .general import sort_large_first as _sort_large_first
from .general import clean_geoms
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

    try:
        gdf = _safe_simplify(gdf, PRECISION)
    except GEOSException:
        pass

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


def make_lines_between_points(
    arr1: NDArray[Point], arr2: NDArray[Point]
) -> NDArray[LineString]:
    if arr1.shape != arr2.shape:
        raise ValueError(
            f"Arrays must have equal shape. Got {arr1.shape} and {arr2.shape}"
        )
    coords: pd.DataFrame = pd.concat(
        [
            pd.DataFrame(get_coordinates(arr1), columns=["x", "y"]),
            pd.DataFrame(get_coordinates(arr2), columns=["x", "y"]),
        ]
    ).sort_index()

    return linestrings(coords.values, indices=coords.index)


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
