import re
import warnings
from typing import Callable

import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryArray
from numpy.typing import NDArray
from shapely import (
    Geometry,
    STRtree,
    extract_unique_points,
    force_2d,
    get_coordinates,
    get_exterior_ring,
    get_parts,
    linearrings,
    linestrings,
    make_valid,
    multipoints,
    polygons,
    segmentize,
    simplify,
    unary_union,
)
from shapely.errors import GEOSException
from shapely.geometry import LinearRing, LineString, MultiLineString, Point
from shapely.ops import nearest_points

from ..networkanalysis.closing_network_holes import get_angle
from ..networkanalysis.cutting_lines import split_lines_by_nearest_point
from .buffer_dissolve_explode import buff, buffdissexp, dissexp, dissexp_by_cluster
from .conversion import coordinate_array, to_geoseries
from .duplicates import get_intersections, update_geometries
from .general import clean_clip, clean_geoms
from .general import sort_large_first as _sort_large_first
from .general import sort_long_first, sort_small_first, to_lines
from .geometry_types import get_geom_type, make_all_singlepart, to_single_geom_type
from .neighbors import get_k_nearest_neighbors
from .overlay import clean_overlay
from .polygon_operations import (
    close_all_holes,
    close_small_holes,
    close_thin_holes,
    eliminate_by_longest,
    get_cluster_mapper,
    get_gaps,
)
from .polygons_as_rings import PolygonsAsRings
from .sfilter import sfilter, sfilter_inverse, sfilter_split


warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


from ..maps.maps import explore


PRECISION = 1e-3
BUFFER_RES = 50


def coverage_clean(
    gdf: GeoDataFrame,
    tolerance: int | float,
    duplicate_action: str = "fix",
    grid_sizes: tuple[None | int] = (None,),
    logger=None,
    mask=None,
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
        duplicate action: Either "fix", "error" or "ignore".
            If "fix" (default), double surfaces thicker than the
            tolerance will be updated from top to bottom (function update_geometries)
            and then dissolved into the neighbor polygon with the longest shared border.
            If "error", an Exception is raised if there are any double surfaces thicker
            than the tolerance. If "ignore", double surfaces are kept as is.
        grid_sizes: One or more grid_sizes used in overlay and dissolve operations that
            might raise a GEOSException. Defaults to (None,), meaning no grid_sizes.

    Returns:
        A GeoDataFrame with cleaned polygons.

    Examples
    --------

    >>> cleaned = coverage_clean(
    ...     gdf,
    ...     0.1,
    ...     grid_sizes=[None, 1e-6, 1e-5, 1e-4, 1e-3],
    ... )

    If you have a known mask for your coverage, e.g. municipality polygons,
    it might be a good idea to buffer the gaps, slivers and double surfaces
    before elimination to make sure the polygons are properly dissolved.

    >>> def _small<_buffer(df):
    ...     df.geometry = df.buffer(0.001)
    ...     return df
    ...
    >>> cleaned = coverage_clean(
    ...     gdf,
    ...     0.1,
    ...     grid_sizes=[None, 1e-6, 1e-5, 1e-4, 1e-3],
    ...     pre_dissolve_func=_small_buffer,
    ... ).pipe(sg.clean_clip, your_mask, geom_type="polygon")

    """

    if not len(gdf):
        return gdf

    _cleaning_checks(gdf, tolerance, duplicate_action)

    if not gdf.index.is_unique:
        gdf = gdf.reset_index(drop=True)

    gdf = make_all_singlepart(gdf).loc[
        lambda x: x.geom_type.isin(["Polygon", "MultiPolygon"])
    ]

    gdf = safe_simplify(gdf, PRECISION)

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
                    explore_geosexception(e, gdf, logger=logger)
                    raise e

    gaps["_was_gap"] = 1

    if duplicate_action == "ignore":
        double = GeoDataFrame({"geometry": []}, crs=gdf.crs)
        double["_double_idx"] = None
    else:
        double = get_intersections(gdf)
        double["_double_idx"] = range(len(double))

    gdf, slivers = split_out_slivers(gdf, tolerance)

    gdf["_poly_idx"] = range(len(gdf))

    thin_gaps_and_double = pd.concat([gaps, double]).loc[
        lambda x: (
            shapely.simplify(x.geometry, PRECISION).buffer(-tolerance / 2).is_empty
        )
    ]

    all_are_thin = double["_double_idx"].isin(thin_gaps_and_double["_double_idx"]).all()

    if not all_are_thin and duplicate_action == "fix":
        gdf, thin_gaps_and_double, slivers = _properly_fix_duplicates(
            gdf, double, slivers, thin_gaps_and_double, tolerance
        )

    elif not all_are_thin and duplicate_action == "error":
        raise ValueError("Large double surfaces.")

    to_eliminate = pd.concat([thin_gaps_and_double, slivers], ignore_index=True)
    # to_eliminate = safe_simplify(to_eliminate, PRECISION)

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
                intersecting, geom_type="polygon", grid_size=grid_size
            ).drop(columns=["_eliminate_idx", "_double_idx"])
            break
        except GEOSException as e:
            if i == len(grid_sizes) - 1:
                explore_geosexception(e, gdf, intersecting, isolated, logger=logger)
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

    for i, grid_size in enumerate(grid_sizes):
        try:
            cleaned = (
                dissexp(
                    pd.concat(
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
                    ),
                    by="_poly_idx",
                    aggfunc="first",
                    dropna=True,
                    grid_size=grid_size,
                )
                .sort_index()
                .reset_index(drop=True)
            )
            break
        except GEOSException as e:
            if i == len(grid_sizes) - 1:
                explore_geosexception(
                    e, gdf, without_double, isolated, really_isolated, logger=logger
                )
                raise e

    for i, grid_size in enumerate(grid_sizes):
        try:
            cleaned = clean_overlay(
                gdf.drop(columns="_poly_idx"),
                cleaned,
                how="update",
                geom_type="polygon",
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
                    logger=logger,
                )
                raise e
            try:
                cleaned = update_geometries(
                    sort_small_first(cleaned), geom_type="polygon", grid_size=grid_size
                )
            except GEOSException:
                pass

    cleaned = sort_small_first(cleaned)

    for i, grid_size in enumerate(grid_sizes):
        try:
            cleaned = update_geometries(
                cleaned, geom_type="polygon", grid_size=grid_size
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
                    logger=logger,
                )
                raise e

    # cleaned = safe_simplify(cleaned, PRECISION)
    cleaned.geometry = shapely.make_valid(cleaned.geometry)

    return cleaned


def safe_simplify(gdf, tolerance: float | int):
    """Simplify only if the resulting area is no more than 1 percent larger.

    Because simplifying can result in holes being filled.
    """
    length_then = gdf.length
    copied = gdf.copy()
    copied.geometry = shapely.make_valid(
        shapely.simplify(copied.geometry.values, tolerance=tolerance)
    )
    copied.loc[
        copied.area > length_then * 1.01, copied._geometry_column_name
    ] = gdf.loc[copied.area > length_then * 1.01, copied._geometry_column_name]

    return copied


def simplify_and_put_small_on_top(gdf, tolerance: float | int, grid_size=None):
    copied = sort_small_first(gdf)
    copied.geometry = shapely.make_valid(
        shapely.simplify(
            shapely.segmentize(copied.geometry.values, tolerance), tolerance=tolerance
        )
    )
    return update_geometries(copied, geom_type="polygon", grid_size=grid_size)


def remove_spikes(gdf: GeoDataFrame, tolerance: int | float) -> GeoDataFrame:
    return clean_overlay(
        gdf, gdf[["geometry"]], how="intersection", grid_size=tolerance
    )


def _properly_fix_duplicates(gdf, double, slivers, thin_gaps_and_double, tolerance):
    gdf = _dissolve_thick_double_and_update(gdf, double, thin_gaps_and_double)
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


def _dissolve_thick_double_and_update(gdf, double, thin_double):
    large = (
        double.loc[~double["_double_idx"].isin(thin_double["_double_idx"])].drop(
            columns="_double_idx"
        )
        # .pipe(sort_large_first)
        # .sort_values("_poly_idx")
        .pipe(update_geometries, geom_type="polygon")
    )
    return (
        clean_overlay(gdf, large, how="update")
        # .pipe(sort_large_first)
        # .sort_values("_poly_idx")
        .pipe(update_geometries, geom_type="polygon")
    )


def _cleaning_checks(gdf, tolerance, duplicate_action):  # , spike_action):
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
    func,
    grid_sizes: tuple[None, float | int],
    args: tuple | None = None,
    kwargs: dict | None = None,
):
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
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
    logger=None,
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
        kwargs=dict(geom_type="polygon"),
    )

    gdf = try_for_grid_size(
        eliminate_by_longest,
        grid_sizes=grid_sizes,
        args=(
            gdf,
            splitted,
        ),
        kwargs=kwargs,
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
        ),
    ).pipe(dissexp_by_cluster)

    return try_for_grid_size(
        eliminate_by_longest, grid_sizes=grid_sizes, args=(gdf, missing), kwargs=kwargs
    )


def split_by_neighbors(df, split_by, tolerance, grid_size=None):
    if not len(df):
        return df

    split_by = split_by.copy()
    split_by.geometry = shapely.simplify(split_by.geometry, tolerance)

    intersecting_lines = (
        clean_overlay(
            to_lines(split_by), buff(df, tolerance), how="identity", grid_size=grid_size
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


def extend_lines(arr1, arr2, distance):
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


def explore_geosexception(e: GEOSException, *gdfs, logger=None):
    from ..maps.maps import Explore, explore
    from .conversion import to_gdf

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


def snap_to_mask(
    gdf: GeoDataFrame, tolerance: int | float, mask: GeoDataFrame | GeoSeries | Geometry
):
    return snap_polygons(
        gdf,
        mask=mask,
        tolerance=tolerance,
        snap_to_nodes=False,
    )


def snap_polygons(
    gdf: GeoDataFrame,
    tolerance: int | float,
    mask: GeoDataFrame | GeoSeries | Geometry | None = None,
    snap_to_nodes: bool = True,
):
    if not len(gdf):
        return gdf

    geom_type = "polygon"

    gdf = safe_simplify(gdf, PRECISION)

    gdf = clean_geoms(gdf).pipe(make_all_singlepart, ignore_index=True)

    gdf = close_thin_holes(gdf, tolerance)

    if mask is None:
        mask: GeoDataFrame = close_all_holes(dissexp_by_cluster(gdf)).dissolve()
        mask_was_none = True
    else:
        try:
            mask: GeoDataFrame = mask[["geometry"]]
        except Exception:
            mask: GeoDataFrame = to_geoseries(mask).to_frame("geometry")
        mask_was_none = False

    gdf.geometry = make_valid(
        PolygonsAsRings(gdf.geometry.values)
        .apply_numpy_func(
            _snap_linearrings,
            kwargs=dict(tolerance=tolerance, mask=mask, snap_to_nodes=snap_to_nodes),
        )
        .to_numpy()
    )

    if not mask_was_none:
        gdf = clean_clip(gdf, mask, geom_type="polygon")

    gdf = to_single_geom_type(make_all_singlepart(gdf), geom_type)

    gdf = update_geometries(close_small_holes(gdf, PRECISION), geom_type="polygon")

    return gdf


def _coverage_clean(
    gdf: GeoDataFrame,
    tolerance: int | float,
    mask: GeoDataFrame | GeoSeries | Geometry | None = None,
    grid_sizes=(None,),
    allowed_missing_area=1e-3,
    max_iterations: int = 2,
    # check_for=("double", "gaps", "missing", "slivers"),
    errors="ignore",
    **kwargs,
):
    if errors not in ["ignore", "raise"]:
        raise ValueError("'errors' should be 'raise' or 'ignore'")

    if not len(gdf):
        return gdf

    # gdf = sort_large_first(make_all_singlepart(clean_geoms(gdf), ignore_index=True))
    gdf = clean_geoms(gdf).pipe(make_all_singlepart, ignore_index=True)

    gdf["_poly_idx"] = range(len(gdf))

    # gdf = clean_overlay(gdf, gdf[["geometry"]], how="intersection", grid_size=tolerance)

    # gdf.index = range(len(gdf))

    # gdf.geometry = set_precision(gdf.geometry, PRECISION)

    if mask is None:
        mask: GeoDataFrame = close_all_holes(dissexp_by_cluster(gdf)).dissolve()
    else:
        try:
            mask: GeoDataFrame = mask[["geometry"]]
        except Exception:
            mask: GeoDataFrame = to_geoseries(mask).to_frame("geometry")
        # mask = close_thin_holes(dissexp_by_cluster(gdf), tolerance).dissolve()

    # outskirts = identify_outskirts(gdf)

    # # sort geometries near the mask highest to give snapping priority
    # gdf = pd.concat(
    #     sfilter_split(gdf, extract_unique_points(mask.geometry).buffer(PRECISION)),
    #     ignore_index=True,
    # )

    # gdf.geometry = set_precision(gdf.geometry, PRECISION)

    # remove thin
    # gdf = gdf.loc[lambda x: ~x.buffer(-tolerance / 2).is_empty]

    prev_gaps_sum, prev_double_sum, prev_missing_sum = 0, 0, 0

    copy = gdf.copy()

    i = -1
    while True:
        i += 1
        failed = False

        if i % 2 == 0:
            gdf = gdf.sort_values("_poly_idx")
        else:
            gdf = gdf.sort_values("_poly_idx", ascending=False)
        if i == max_iterations - 1:
            if errors == "raise":
                raise ValueError()
            else:
                return to_single_geom_type(
                    gdf.drop(columns=["_was_gap", "_poly_idx"], errors="ignore"),
                    "polygon",
                )

        gdf = to_single_geom_type(make_all_singlepart(gdf), "polygon")

        gdf.geometry = make_valid(
            PolygonsAsRings(gdf.geometry.values)
            .apply_numpy_func(
                _snap_linearrings,
                kwargs=dict(
                    tolerance=tolerance,
                    mask=mask,
                    snap_to_nodes=True,
                ),
            )
            .to_numpy()
        )

        gdf = to_single_geom_type(make_all_singlepart(gdf), "polygon")

        try:
            gdf = update_geometries(
                close_small_holes(gdf, PRECISION), geom_type="polygon"
            )
        except GEOSException:
            failed = True

        gaps = get_gaps(gdf).loc[
            lambda x: (x.area > 0)
            & (simplify(x.geometry, tolerance).buffer(-tolerance / 2).is_empty)
        ]
        gaps["_was_gap"] = 1

        double = get_intersections(gdf, geom_type="polygon")
        try:
            missing = clean_overlay(
                copy, gdf, how="difference", geom_type="polygon"
            ).pipe(clean_clip, mask, geom_type="polygon")
        except GEOSException:
            continue

        gdf, slivers = split_out_slivers(gdf, tolerance)

        # explore(
        #     missing,
        #     gaps,
        #     double,
        #     gdf,
        #     copy,
        #     mask,
        #     slivers,
        #     thick_missing=missing[lambda x: ~x.buffer(-0.01).is_empty],
        # )

        # print(
        #     gaps.area.max(),
        #     double.area.max(),
        #     missing.area.max(),
        #     slivers.area.max(),
        # )

        if (
            (not failed)
            and (not len(gaps) or gaps.area.max() < allowed_missing_area)
            and (not len(double) or double.area.max() < allowed_missing_area)
            and (not len(missing) or missing.area.max() < allowed_missing_area)
            and (not len(slivers) or slivers.area.max() < allowed_missing_area)
        ):
            break

        to_eliminate = pd.concat([gaps, double, missing, slivers])
        to_eliminate.geometry = to_eliminate.buffer(PRECISION)
        gdf = _eliminate(
            gdf, to_eliminate, tolerance, grid_sizes=grid_sizes, logger=None
        ).pipe(clean_clip, mask, geom_type="polygon")

        try:
            gdf = update_geometries(gdf, geom_type="polygon")
        except GEOSException:
            pass

        # try:
        #     gdf = eliminate_by_longest(
        #         gdf,
        #         pd.concat(
        #             [
        #                 gaps,
        #                 double,
        #                 slivers,
        #                 missing,
        #             ]
        #         )
        #         # .buffer(PRECISION)
        #         # .to_frame("geometry"),
        #     )  # .pipe(clean_clip, mask, geom_type="polygon")
        # except GEOSException:
        #     pass

        gaps = get_gaps(gdf).loc[
            lambda x: simplify(x.geometry, tolerance).buffer(-tolerance / 2).is_empty
        ]
        gaps["_was_gap"] = 1

        double = get_intersections(gdf, geom_type="polygon")
        try:
            missing = clean_overlay(
                copy, gdf, how="difference", geom_type="polygon"
            ).pipe(clean_clip, mask, geom_type="polygon")
        except GEOSException:
            continue

        if (
            (not len(gaps) or gaps.area.max() < allowed_missing_area)
            and (not len(double) or double.area.max() < allowed_missing_area)
            and (not len(missing) or missing.area.max() < allowed_missing_area)
        ):
            break

        slivers = gdf.loc[gdf.buffer(-tolerance / 2).is_empty]

        # explore(
        #     missing,
        #     gaps,
        #     double,
        #     gdf,
        #     copy,
        #     mask,
        #     slivers,
        #     thick_missing=missing[lambda x: ~x.buffer(-0.01).is_empty],
        # )

        # print(
        #     gaps.area.max(),
        #     double.area.max(),
        #     missing.area.max(),
        #     slivers.area.max(),
        # )

        gaps_sum = round(gaps.area.sum(), 3) if len(gaps) else 0
        double_sum = round(double.area.sum(), 3) if len(double) else 0
        missing_sum = round(missing.area.sum(), 3) if len(missing) else 0
        # slivers_sum = round(slivers.area.sum(), 3) if len(slivers) else 0

        if (
            gaps_sum == prev_gaps_sum
            and double_sum == prev_double_sum
            and missing_sum == prev_missing_sum
            # and slivers_sum == prev_slivers_sum
        ):
            break

        prev_gaps_sum = gaps_sum
        prev_double_sum = double_sum
        prev_missing_sum = missing_sum
        # prev_slivers_sum = slivers_sum

    return to_single_geom_type(
        gdf.drop(columns=["_was_gap", "_poly_idx"], errors="ignore"), "polygon"
    )


def _snap_to_anchors(
    points: GeoDataFrame,
    tolerance: int | float,
    anchors: GeoDataFrame | None = None,
    custom_func: Callable | None = None,
) -> GeoDataFrame:
    if not len(points):
        points["_was_snapped"] = 0
        try:
            return points, anchors[["geometry"]]
        except TypeError:
            return points, points[["geometry"]]

    points["_was_anchor"] = 0

    if anchors is not None:
        # anchor_index_mapper = dict(enumerate(anchors.index))
        anchors["_was_anchor"] = 1
        points = pd.concat([anchors, points])

    index_mapper = dict(enumerate(points.index))
    points = points.reset_index(drop=True)

    tree = STRtree(points.geometry.values)
    left, right = tree.query(
        points.geometry.values, predicate="dwithin", distance=tolerance * 2
    )
    indices = pd.Series(right, index=left, name="_right_idx")

    # index_mapper = dict(enumerate(points.index))
    # indices.index = indices.index.map(index_mapper)
    # indices.loc[:] = indices.map(index_mapper)

    # geom_idx_left = indices.index.map(dict(enumerate(points["_geom_idx"])))
    # geom_idx_right = indices.map(dict(enumerate(points["_geom_idx"])))

    # if anchors is not None:
    #     left, right = tree.query(
    #         anchors.geometry.values, predicate="dwithin", distance=tolerance
    #     )
    #     more_indices = pd.Series(right, index=left, name="_right_idx")

    #     more_indices.index = more_indices.index.map(index_mapper)
    #     more_indices.loc[:] = more_indices.map(dict(enumerate(anchors.index)))

    # indices = pd.concat([snap_indices, indices])

    left_on_top = indices.loc[indices.index < indices.values].sort_index()
    # left_on_top = indices.loc[geom_idx_left < geom_idx_right]

    # keep only indices from left if they have not already appeared in right
    # these shouldn't be anchors, but instead be snapped
    new_indices = []
    values = []
    right_indices = set()
    for left, right in left_on_top.items():
        if left not in right_indices:
            # if left in anchor_indices or left not in right_indices:
            new_indices.append(left)
            values.append(right)
            right_indices.add(right)

    snap_indices = pd.Series(values, index=new_indices)

    if custom_func:
        snap_indices = custom_func(snap_indices)

    # remove "duplicate" anchors, i.e. anchors with identical right indices
    # snap_indices = snap_indices.loc[
    #     ~snap_indices.groupby(level=0).unique().apply(sorted).duplicated()
    # ]

    new_anchors = points.loc[points.index.isin(snap_indices.index), ["geometry"]]
    if anchors is not None:
        anchors = pd.concat([anchors, new_anchors]).loc[
            lambda x: ~x.geometry.duplicated()
        ]
        # anchors = new_anchors
    else:
        anchors = new_anchors
        anchors["_was_anchor"] = 0

    # explore(
    #     old_anchors=old_anchors,
    #     points=to_gdf(points, 25833).assign(wkt=lambda x: x.geometry.to_wkt()),
    #     anchors=points.loc[points.index.isin(snap_indices.index)]
    #     .set_crs(25833)
    #     .assign(wkt=lambda x: x.geometry.to_wkt()),
    # )

    points = points.loc[lambda x: (x["_was_anchor"] == 0)].drop(columns=["_was_anchor"])

    # anchors["_cluster"] = get_cluster_mapper(anchors.buffer(PRECISION))
    # anchors = anchors.loc[lambda x: ~x["_cluster"].duplicated()]

    to_be_snapped = points.loc[
        lambda x: (x.index.isin(snap_indices.values))
        # points.index.isin(anchors.index)
    ]

    # explore(
    #     to_be_snapped,
    #     old_anchors=old_anchors,
    #     points=to_gdf(points, 25833).assign(wkt=lambda x: x.geometry.to_wkt()),
    #     anchors=points.loc[points.index.isin(snap_indices.index)]
    #     .set_crs(25833)
    #     .assign(wkt=lambda x: x.geometry.to_wkt()),
    # )

    # snapping with sjoin_nearest because it's faster than nearest_points
    # anchors.index = anchors.geometry
    anchors["_right_geom"] = anchors.geometry
    # anchors = buffdissexp(anchors, PRECISION)
    # anchors.geometry = anchors["_right_geom"]

    snapped = (
        # to_be_snapped.sjoin(buff(anchors, tolerance))
        to_be_snapped.sjoin_nearest(anchors, max_distance=tolerance * 2)
        # .loc[lambda x: (~x.index.isin(mask_nodes.index))]
        # .sort_values("index_right")
        ["_right_geom"]  # .sort_index()
        # .loc[lambda x: x["_geom_idx"] != x["_geom_idx_right"], "index_right"]
        .loc[lambda x: ~x.index.duplicated()]
    )

    # assert len(snapped) == len(to_be_snapped), (
    #     len(snapped),
    #     len(to_be_snapped),
    #     len(anchors),
    # )

    # to_be_snapped["geometry"] = snapped

    points.loc[snapped.index, "geometry"] = snapped
    points.loc[snapped.index, "_was_snapped"] = 1

    assert points.geometry.notna().all()

    # agged = (
    #     # points.loc[lambda x: (x["_geom_idx"] >= idx_start) & (x["_is_thin"] != True)]
    #     points.loc[lambda x: (x["_is_thin"] != True)]
    #     .sort_index()
    #     .loc[lambda x: x.groupby("_geom_idx").transform("size") > 2]
    #     .groupby("_geom_idx")["geometry"]
    #     .agg(LinearRing)
    #     .reset_index()
    # )

    # explore(
    #     to_be_snapped,
    #     new_anchors=new_anchors,
    #     points=to_gdf(points, 25833).assign(wkt=lambda x: x.geometry.to_wkt()),
    #     anchors=points.loc[points.index.isin(snap_indices.index)]
    #     .set_crs(25833)
    #     .assign(wkt=lambda x: x.geometry.to_wkt()),
    #     agged=agged.buffer(0.01).to_frame("geometry"),
    #     snapped=to_gdf(snapped).assign(wkt=lambda x: x.geometry.to_wkt()),
    # )

    # print("\nneeeeede")
    # print(
    #     sfilter(points, to_gdf("POINT (905293.6391 7878540.5661)", 25833).buffer(0.01))
    # )

    # print(
    #     "anchors",
    #     sfilter(
    #         points.loc[points.index.isin(snap_indices.index), ["geometry"]],
    #         to_gdf("POINT (905293.6391 7878540.5661)", 25833).buffer(0.01),
    #     ),
    # )

    # print(
    #     sfilter(snapped, to_gdf("POINT (905293.6391 7878540.5661)", 25833).buffer(0.01))
    # )

    points.index = points.index.map(index_mapper)
    assert points.index.notna().all()
    # print(points.sort_index())
    # print(anchors.sort_index())
    # print("heh")
    # print(new_anchors.sort_index())
    # print(anchors[anchors.index.map(index_mapper).isna()])
    # anchors.index = anchors.index.map(index_mapper)

    # map index only where index was reset
    anchors.loc[anchors["_was_anchor"] != 1].index = anchors.loc[
        anchors["_was_anchor"] != 1
    ].index.map(index_mapper)

    assert anchors.index.notna().all(), anchors[anchors.index.isna()]

    # anchors = pd.concat([original_anchors, anchors])[["geometry"]]

    return points, anchors[["geometry"]]


def _snap_linearrings(
    geoms: NDArray[LinearRing],
    tolerance: int | float,
    mask: GeoDataFrame | None = None,
    snap_to_nodes: bool = True,
):
    if not len(geoms):
        return geoms

    if mask is None:
        idx_start = 0
    else:
        mask = make_all_singlepart(mask).geometry
        idx_start = len(mask)

    gdf = GeoDataFrame(
        {"geometry": geoms, "_geom_idx": np.arange(idx_start, len(geoms) + idx_start)}
    )

    is_thin = GeoSeries(polygons(gdf.geometry)).buffer(-tolerance / 2).is_empty

    gdf["_is_thin"] = is_thin

    thin = is_thin[lambda x: x == True]
    thin.loc[:] = None
    thin.index = thin.index.map(gdf["_geom_idx"])

    # points_from_thin = (
    #     extract_unique_points(gdf.loc[is_thin, "geometry"])
    #     .to_frame("geometry")
    #     .explode(ignore_index=True)
    #     .pipe(sfilter_inverse, gdf.buffer(PRECISION))
    # )

    points: GeoDataFrame = gdf.assign(
        geometry=lambda x: extract_unique_points(x.geometry.values)
    ).explode(ignore_index=True)

    mask_nodes = GeoDataFrame(
        {
            "geometry": extract_unique_points(mask.geometry),
            "_geom_idx": range(len(mask)),
        }
    ).explode(ignore_index=True)

    # if mask is not None:
    #     mask_nodes = GeoDataFrame(
    #         {
    #             "geometry": extract_unique_points(mask.geometry),
    #             "_geom_idx": range(len(mask)),
    #         }
    #     )

    #     points = pd.concat([mask_nodes, points], ignore_index=True).explode(
    #         ignore_index=True
    #     )
    #     n_original_points = len(points) - len(mask_nodes)
    # else:
    #     points = points.explode(ignore_index=True)
    #     n_original_points = len(points)

    # appear_only_once = points.loc[
    #     lambda x: x.groupby("geometry").transform("size") == 1
    # ]

    # should_be_snapped, on_outskirts = sfilter_split(appear_only_once, mask.buffer(-PRECISION))
    # # points = pd.concat([should_be_snapped, sfilter_inverse(points, pd.concat([should_be_snapped, on_outskirts]).buffer(PRECISION)), on_outskirts])
    # explore(
    #     on_outskirts,
    #     should_be_snapped=should_be_snapped.set_crs(25833),
    # )

    # appear_only_once["_cluster"] = get_cluster_mapper(
    #     appear_only_once.buffer(PRECISION)
    # )
    # appear_only_once = appear_only_once.drop_duplicates("_cluster")

    # points = pd.concat([points[~points.index.isin(appear_only_once.index)], appear_only_once])

    # sliver_points_on_outskirts = sfilter_inverse(
    #     points[lambda x: x["_is_thin"] == True],
    #     points[lambda x: x["_is_thin"] != True].buffer(PRECISION),
    # )
    # sliver_points = points[lambda x: x["_is_thin"] == True]

    # # points = points[lambda x: x["_is_thin"] != True]

    # sliver_points_on_outskirts = sfilter_inverse(
    #     sliver_points,
    #     points.buffer(PRECISION),
    # )

    # thin_rings = gdf.loc[lambda x: x["_is_thin"] == True].rename(
    #     columns={"_geom_idx": "_geom_idx_right"}
    # )
    # thick_rings = gdf.loc[lambda x: x["_is_thin"] != True]
    # thick_and_slivers_joined = thick_rings.sjoin(buff(thin_rings, PRECISION))
    # sliver_to_thick_pair_mapper = dict(
    #     zip(
    #         thick_and_slivers_joined["_geom_idx_right"],
    #         thick_and_slivers_joined["_geom_idx"],
    #     )
    # )

    # points = pd.concat(
    #     [sliver_points_on_outskirts, points[lambda x: x["_is_thin"] != True]]
    # )

    # step 1: add new points to segments near mask

    if 0:
        segments = points_to_line_segments(points.set_index("_geom_idx"))
        segments["_geom_idx"] = segments.index
        segments.index = points.index

        mask_nodes["rgeom"] = mask_nodes.geometry

        assert mask_nodes.index.is_unique

        joined = (
            segments.sjoin_nearest(mask_nodes, max_distance=tolerance).sort_values(
                "index_right"
            )
            # .loc[lambda x: ~x.index.duplicated()]
            .drop_duplicates("index_right")
        )

        new_segs = []
        while True:
            no_dups = joined.loc[lambda x: ~x.index.duplicated()]
            nearest_ring_points = GeoSeries(
                nearest_points(no_dups.geometry.values, no_dups["rgeom"].values)[0],
                index=no_dups.index,
            )

            boundaries = (
                extract_unique_points(no_dups.geometry)
                .explode(index_parts=False)
                .groupby(level=0)
            )

            with_midpoints = pd.concat(
                [
                    boundaries.nth(0),
                    nearest_ring_points,
                    boundaries.nth(-1),
                ]
            ).sort_index()

            to_int_index = {
                ring_idx: i
                for i, ring_idx in enumerate(sorted(set(with_midpoints.index)))
            }

            no_dups.geometry = linestrings(
                get_coordinates(with_midpoints.values),
                indices=with_midpoints.index.map(to_int_index),
            )

            new_segs.append(no_dups)

            joined = joined.loc[lambda x: ~x["_range_idx"].isin(no_dups["_range_idx"])]

            if not len(joined):
                break

        segments.loc[joined.index, "geometry"] = lines_with_minpoint

        segments.geometry = extract_unique_points(segments.geometry)
        points = segments.explode(index_parts=False)

    if 1:
        mask_rings: GeoSeries = PolygonsAsRings(mask).get_rings()

        segments = points_to_line_segments(points.set_index("_geom_idx"))
        segments["_geom_idx"] = segments.index
        segments.index = points.index

        segs_by_mask, not_by_mask = sfilter_split(
            segments, mask_rings.buffer(tolerance)
        )

        relevant_mask_nodes = sfilter(mask_nodes, segs_by_mask.buffer(tolerance))

        points_to_join = (
            mask_nodes.copy()
        )  # .rename(columns={"_geom_idx": "_geom_idx_right"})
        points_to_join["rgeom"] = points_to_join.geometry
        points_to_join.geometry = points_to_join.buffer(tolerance)

        joined = (
            segs_by_mask.sjoin(points_to_join)
            # .loc[lambda x: x["_geom_idx"] != x["_geom_idx_right"]]
            # .sort_values("index_right")
        )

        from .conversion import to_gdf

        explore(
            to_gdf(joined, 25833),
            to_gdf(points_to_join),
            mask_nodes=mask_nodes,
        )

        # new_midpoints = GeoSeries(
        #     nearest_points(
        #         joined.geometry.values, joined["rgeom"].unary_union
        #     )[0],
        #     index=joined.index,
        # )

        boundaries = (
            extract_unique_points(joined.geometry.loc[lambda x: ~x.index.duplicated()])
            .explode(index_parts=False)
            .groupby(level=0)
        )

        midpoints = GeoSeries(joined.groupby(level=0)["rgeom"].unique().explode())

        # with_midpoints = pd.concat(
        #     [
        #         boundaries.nth(0),
        #         joined.groupby(level=0)["rgeom"].unique().explode(),
        #         boundaries.nth(-1),
        #     ]
        # ).sort_index()

        # dist_to_source = shapely.distance(with_midpoints.groupby(level=0).nth(0), with_midpoints.groupby(level=0).nth(1))
        # dist_to_target = shapely.distance(with_midpoints.groupby(level=0).nth(-1), with_midpoints.groupby(level=0).nth(1))

        dist_to_source = boundaries.nth(0).distance(midpoints.groupby(level=0).nth(1))
        dist_to_target = boundaries.nth(-1).distance(midpoints.groupby(level=0).nth(1))

        should_be_flipped = dist_to_source > dist_to_target

        print(should_be_flipped)

        midpoints.loc[should_be_flipped == True] = (
            midpoints.loc[should_be_flipped == True].groupby(level=0).apply(reversed)
        )
        print(midpoints)
        sss

        dist_to_source = shapely.distance(
            boundaries.nth(0), midpoints.groupby(level=0).nth(1)
        )
        dist_to_target = shapely.distance(
            boundaries.nth(-1), midpoints.groupby(level=0).nth(1)
        )

        # print(points.loc[1922])
        # print(segments.loc[1922])
        # print(segs_by_mask.loc[1922])
        # print(with_midpoints.loc[1922])

        explore(
            to_gdf(with_midpoints.loc[1922], 25833),
            to_gdf(segs_by_mask.loc[1922]),
            to_gdf(joined.loc[1922].rgeom),
            mask_nodes=mask_nodes,
            points_to_join=to_gdf(points_to_join, 25833),
        )

        to_int_index = {
            ring_idx: i for i, ring_idx in enumerate(sorted(set(with_midpoints.index)))
        }

        aggged = linestrings(
            get_coordinates(with_midpoints.values),
            indices=with_midpoints.index.map(to_int_index),
        )

        explore(
            to_gdf(aggged, 25833),
            with_midpoints,
            joined,
            segs_by_mask,
            mask_nodes=mask_nodes,
        )
        explore(
            to_gdf(extract_unique_points(aggged), 25833)[["geometry"]].explode(),
            to_gdf(extract_unique_points(segs_by_mask.geometry))[
                ["geometry"]
            ].explode(),
        )

        sss

        new_midpoints = GeoSeries(
            nearest_points(
                segs_by_mask.geometry.values, relevant_mask_points.unary_union
            )[0],
            index=segs_by_mask.index,
        )

        assert segs_by_mask.index.is_unique
        assert (segs_by_mask.geom_type == "LineString").all()
        boundaries = segs_by_mask.boundary.explode(index_parts=False).groupby(level=0)

        with_midpoints = pd.concat(
            [
                boundaries.nth(0),
                new_midpoints,
                boundaries.nth(-1),
            ]
        ).sort_index()

        print(segs_by_mask)
        print(boundaries)
        print(with_midpoints)

        print(
            with_midpoints.index.map(
                {idx: i for i, idx in enumerate(with_midpoints.index)}
            )
        )

        segs_by_mask.geometry = linestrings(
            get_coordinates(with_midpoints),
            indices=with_midpoints.index.map(
                {idx: i for i, idx in enumerate(with_midpoints.index)}
            ),
        )

    # explore(points, gdf, mask.to_frame("geometry").to_crs(25833))
    # step 1: snap to mask nodes

    points_by_mask_nodes = sfilter(
        points.loc[lambda x: x["_geom_idx"] >= idx_start], mask_nodes.buffer(tolerance)
    )
    relevant_mask_nodes = sfilter(
        mask_nodes.explode(ignore_index=True),
        points_by_mask_nodes.buffer(tolerance),
        predicate="within",
    )

    if len(relevant_mask_nodes):
        if 0:
            anchor_points = nearest_points(
                points_by_mask_nodes.geometry.values, relevant_mask_nodes.unary_union
            )[1]
            anchors = GeoSeries(
                anchor_points,
                index=points_by_mask_nodes.index,
            )
            points_by_mask_nodes.geometry = anchors
            anchors = anchors.to_frame("geometry")
        else:
            relevant_mask_nodes["_right_geom"] = relevant_mask_nodes.geometry
            snapped = points_by_mask_nodes.sjoin_nearest(
                relevant_mask_nodes, max_distance=tolerance * 2
            )

            anchors = snapped.drop_duplicates("index_right")[["geometry"]]

            points_by_mask_nodes.geometry = snapped["_right_geom"].loc[
                lambda x: ~x.index.duplicated()
            ]
    else:
        anchors = None

    # step 2: snap from points to mask edges (between the nodes)

    if 0:
        mask_rings: GeoSeries = PolygonsAsRings(mask).get_rings()

        points_by_mask = sfilter(
            points.loc[
                lambda x: (x["_geom_idx"] >= idx_start)
                & (~x.index.isin(points_by_mask_nodes.index))
            ],
            mask_rings.buffer(tolerance),
        )
        relevant_mask = mask_rings.clip(points_by_mask.buffer(tolerance))

        mask_anchors = GeoSeries(
            nearest_points(points_by_mask.geometry.values, relevant_mask.unary_union)[
                1
            ],
            index=points_by_mask.index,
        )
        points_by_mask.geometry = mask_anchors

        snapped_indices: pd.Index = points_by_mask_nodes.index.union(mask_anchors.index)
        assert snapped_indices.is_unique

        anchors = pd.concat(
            [
                anchors,
                mask_anchors.to_frame("geometry"),
            ]
        )

    else:
        snapped_indices: pd.Index = points_by_mask_nodes.index

    # step 2: snap from edges to mask nodes

    if 0:
        from .conversion import to_gdf

        explore(to_gdf(relevant_mask_points, 25833), to_gdf(mask_anchors, 25833))
        explore(
            to_gdf(relevant_mask_points, 25833), to_gdf(points_by_mask_nodes, 25833)
        )
        explore(
            to_gdf(relevant_mask_points, 25833),
            to_gdf(mask_anchors, 25833),
            to_gdf(points_by_mask_nodes, 25833),
        )
        explore(to_gdf(mask_anchors, 25833), to_gdf(points_by_mask_nodes, 25833))
        explore(to_gdf(lines_by_mask, 25833))

        snapped_indices: pd.Index = lines_by_mask.index.union(mask_anchors.index)
        assert snapped_indices.is_unique

        anchors = pd.concat(
            [
                anchors,
                mask_anchors.to_frame("geometry"),
            ]
        )

    # step 3: snap to gdf nodes

    points_by_mask_nodes["_was_snapped"] = 1
    # points_by_mask["_was_snapped"] = 1
    points["_was_snapped"] = 0

    if snap_to_nodes:
        # snapped_to_nodes, anchors = _snap_to_anchors(
        #     points,
        #     tolerance,
        #     anchors=anchors,
        # )
        snapped_to_nodes, _ = _snap_to_anchors(
            points.loc[lambda x: ~x.index.isin(snapped_indices)],
            tolerance,
            anchors=anchors,  # mask_nodes.explode(ignore_index=True),  # anchors,
        )
        # snapped_to_nodes = snapped_to_nodes.loc[
        #     lambda x: ~x.index.isin(snapped_indices)
        # ]
        points = (
            pd.concat(
                [
                    points_by_mask_nodes,
                    # points_by_mask,
                    snapped_to_nodes,
                ]
            )
            # pd.concat([points_by_mask_nodes, points_by_mask, snapped_to_nodes, points])
            # .loc[lambda x: ~x.index.duplicated()]
            # .loc[lambda x: x["_is_thin"] != True]
            .sort_index()
        )
    else:
        points = pd.concat(
            [
                points_by_mask_nodes,
                # points_by_mask,
                points.loc[~points.index.isin(snapped_indices)],
            ]
        ).sort_index()

    assert points.geometry.notna().all()
    assert points.index.is_unique, points[points.index.duplicated()]

    snapped = (
        points.loc[lambda x: (x["_geom_idx"] >= idx_start)]
        # snapped.loc[lambda x: (x["_geom_idx"] >= idx_start) & (x["_is_thin"] != True)]
        .sort_index()
        .set_index("_geom_idx")
        # .pipe(_remove_legit_spikes)
        .loc[lambda x: x.groupby(level=0).size() > 2]
        .groupby(level=0)["geometry"]
        .agg(LinearRing)
    )

    missing = gdf.set_index("_geom_idx")["geometry"].loc[
        lambda x: (~x.index.isin(snapped.index))
        & (~x.index.isin(thin.index))
        & (x.index >= idx_start)
    ]

    return pd.concat([snapped, missing]).sort_index()


def _remove_legit_spikes(df):
    """Remove points where the next and previous points are the same.

    The lines these points make are as spiky as they come,
    hence the term "legit spikes".
    """
    df["next"] = df.groupby(level=0)["geometry"].shift(-1)
    df["prev"] = df.groupby(level=0)["geometry"].shift(1)

    first_points = df.loc[lambda x: ~x.index.duplicated(keep="first"), "geometry"]
    is_last_point = df["next"].isna()
    df.loc[is_last_point, "next"] = first_points

    last_points = df.loc[lambda x: ~x.index.duplicated(keep="last"), "geometry"]
    is_first_point = df["prev"].isna()
    df.loc[is_first_point, "prev"] = last_points

    assert df["next"].notna().all()
    assert df["prev"].notna().all()

    print(df)
    print(df.loc[lambda x: x["next"] != x["prev"]])
    explore(
        df.set_crs(25833),
        df.loc[lambda x: x["next"] != x["prev"]],
        df.loc[lambda x: x["next"] == x["prev"]],
    )

    return df.loc[lambda x: x["next"] != x["prev"]]


def _eliminate(gdf, to_eliminate, tolerance, grid_sizes=(None,), logger=None):
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
                intersecting, geom_type="polygon", grid_size=grid_size
            ).drop(columns=["_eliminate_idx"])
            break
        except GEOSException as e:
            if i == len(grid_sizes) - 1:
                explore_geosexception(e, gdf, intersecting, isolated, logger=logger)
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

    for i, grid_size in enumerate(grid_sizes):
        try:
            return (
                dissexp(
                    pd.concat(
                        [
                            gdf,
                            without_double,
                            not_really_isolated,
                            really_isolated,
                            isolated_gaps,
                        ]
                    ).drop(
                        columns=[
                            "_cluster",
                            "_eliminate_idx",
                            "index_right",
                            "_area_per_poly",
                            "_was_gap",
                        ],
                        errors="ignore",
                    ),
                    by="_poly_idx",
                    aggfunc="first",
                    dropna=True,
                    grid_size=grid_size,
                )
                .sort_index()
                .reset_index(drop=False)
            )
            break
        except GEOSException as e:
            if i == len(grid_sizes) - 1:
                explore_geosexception(
                    e, gdf, without_double, isolated, really_isolated, logger=logger
                )
                raise e


def get_line_segments(lines) -> GeoDataFrame:
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
        LineString([x1, x2]) for x1, x2 in zip(point_df["geometry"], point_df["next"])
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
        LineString([x1, x2]) for x1, x2 in zip(points["geometry"], points["next"])
    ]
    return GeoDataFrame(
        points.drop(columns=["next"]), geometry="geometry", crs=points.crs
    )
