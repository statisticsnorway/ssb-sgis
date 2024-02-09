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
    reverse,
    segmentize,
    simplify,
    unary_union,
)
from shapely.errors import GEOSException
from shapely.geometry import LinearRing, LineString, MultiLineString, MultiPoint, Point
from shapely.ops import nearest_points

from ..networkanalysis.closing_network_holes import get_angle
from ..networkanalysis.cutting_lines import split_lines_by_nearest_point
from .buffer_dissolve_explode import buff, buffdissexp, dissexp, dissexp_by_cluster
from .conversion import coordinate_array, to_gdf, to_geoseries
from .duplicates import get_intersections, update_geometries

# from .general import sort_large_first as _sort_large_first
from .general import (
    clean_clip,
    clean_geoms,
    sort_large_first,
    sort_long_first,
    sort_small_first,
    to_lines,
)
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


PRECISION = 1e-3
BUFFER_RES = 50


def coverage_clean(
    gdf: GeoDataFrame,
    tolerance: int | float,
    duplicate_action: str = "fix",
    grid_sizes: tuple[None | int] = (None,),
    logger=None,
    mask=None,
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
        double = get_intersections(gdf, n_jobs=n_jobs)
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
    to_eliminate = safe_simplify(to_eliminate, PRECISION)

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
        one_hit = cleaned[lambda x: x.index == min(x.index) - 1]
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
                explore_geosexception(
                    e, gdf, without_double, isolated, really_isolated, logger=logger
                )
                raise e

    cleaned = pd.concat([many_hits, one_hit], ignore_index=True)

    for i, grid_size in enumerate(grid_sizes):
        try:
            cleaned = clean_overlay(
                gdf.drop(columns="_poly_idx"),
                cleaned,
                how="update",
                geom_type="polygon",
                n_jobs=n_jobs,
            )
            break
        except GEOSException as e:
            if 1 == 0:
                try:
                    cleaned = update_geometries(
                        sort_small_first(cleaned),
                        geom_type="polygon",
                        grid_size=grid_size,
                        n_jobs=n_jobs,
                    )
                except GEOSException:
                    pass
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

    cleaned = sort_small_first(cleaned)

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
                    logger=logger,
                )
                raise e

    cleaned = safe_simplify(cleaned, PRECISION)
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
    copied.loc[copied.area > length_then * 1.01, copied._geometry_column_name] = (
        gdf.loc[copied.area > length_then * 1.01, copied._geometry_column_name]
    )

    return copied


def simplify_and_put_small_on_top(gdf, tolerance: float | int, grid_size=None):
    copied = sort_small_first(gdf)
    copied.geometry = shapely.make_valid(
        shapely.simplify(
            shapely.segmentize(copied.geometry.values, tolerance), tolerance=tolerance
        )
    )
    return update_geometries(copied, geom_type="polygon", grid_size=grid_size)


def remove_spikes(
    gdf: GeoDataFrame, tolerance: int | float, n_jobs: int = 1
) -> GeoDataFrame:
    return clean_overlay(
        gdf, gdf[["geometry"]], how="intersection", grid_size=tolerance, n_jobs=n_jobs
    )


def _properly_fix_duplicates(
    gdf, double, slivers, thin_gaps_and_double, tolerance, n_jobs
):
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


def _dissolve_thick_double_and_update(gdf, double, thin_double, n_jobs):
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
    **kwargs,
):
    if not len(gdf):
        return gdf

    geom_type = "polygon"

    gdf = safe_simplify(gdf, PRECISION)

    gdf = (
        clean_geoms(gdf)
        .pipe(make_all_singlepart, ignore_index=True)
        .pipe(to_single_geom_type, geom_type)
    )

    gdf = close_thin_holes(gdf, tolerance)

    if mask is None:
        mask: GeoDataFrame = close_all_holes(dissexp_by_cluster(gdf)).dissolve()
    else:
        try:
            mask: GeoDataFrame = mask[["geometry"]]
        except Exception:
            mask: GeoDataFrame = to_geoseries(mask).to_frame("geometry")

    gdf_copy = gdf.copy()

    gdf.geometry = (
        PolygonsAsRings(gdf.geometry.values)
        .apply_numpy_func(
            _snap_linearrings,
            kwargs=dict(tolerance=tolerance, mask=mask, snap_to_nodes=snap_to_nodes),
        )
        .to_numpy()
    )

    gdf = to_single_geom_type(make_all_singlepart(clean_geoms(gdf)), geom_type)

    if snap_to_nodes:
        missing = clean_overlay(gdf_copy, gdf, how="difference")

        missing, isolated = sfilter_split(missing, gdf)
        isolated.geometry = isolated.buffer(PRECISION * 10)
        gdf = eliminate_by_longest(
            gdf, pd.concat([missing, isolated]), remove_isolated=False
        )

    missing = clean_overlay(mask, gdf, how="difference")

    gdf = eliminate_by_longest(
        gdf, missing.buffer(PRECISION * 10).to_frame("geometry"), remove_isolated=False
    ).pipe(clean_clip, mask, geom_type="polygon")

    gdf = update_geometries(
        sort_small_first(close_small_holes(gdf, PRECISION)), geom_type="polygon"
    )

    return gdf


def _snap_to_anchors(
    points: GeoDataFrame,
    tolerance: int | float,
    anchors: GeoDataFrame | None = None,
    custom_func: Callable | None = None,
) -> GeoDataFrame:
    if not len(points):
        try:
            return points, anchors[["geometry"]]
        except TypeError:
            return points, points[["geometry"]]

    assert points.index.is_unique

    tree = STRtree(points.geometry.values)
    left, right = tree.query(
        points.geometry.values,
        predicate="dwithin",
        distance=tolerance,
    )
    indices = pd.Series(right, index=left, name="_right_idx")

    geom_idx_left = indices.index.map(dict(enumerate(points["_geom_idx"])))
    geom_idx_right = indices.map(dict(enumerate(points["_geom_idx"])))

    left_on_top = indices.loc[geom_idx_left < geom_idx_right].sort_index()

    # keep only indices from left if they have not already appeared in right
    # these shouldn't be anchors, but instead be snapped
    new_indices = []
    values = []
    right_indices = set()
    for left, right in left_on_top.items():
        if left not in right_indices:
            new_indices.append(left)
            values.append(right)
            right_indices.add(right)

    snap_indices = pd.Series(values, index=new_indices)

    if custom_func:
        snap_indices = custom_func(snap_indices)

    new_anchors = points.loc[
        points.index.isin(snap_indices.index), ["geometry", "_geom_idx"]
    ]
    new_anchors["_cluster"] = get_cluster_mapper(new_anchors.buffer(0.1))

    assert new_anchors["_geom_idx"].notna().all()

    no_longer_anchors: pd.Index = new_anchors.loc[
        lambda x: (x["_cluster"].duplicated())  # & (x["_geom_idx"] >= idx_start)
    ].index
    new_anchors = new_anchors.loc[lambda x: ~x.index.isin(no_longer_anchors)]

    if anchors is not None:
        anchors = pd.concat([anchors, new_anchors]).loc[
            lambda x: ~x.geometry.duplicated()
        ]
    else:
        anchors = new_anchors
        anchors["_was_anchor"] = 0

    should_be_snapped = (points.index.isin(snap_indices.values)) | (
        points.index.isin(no_longer_anchors)
    )
    if anchors is not None:
        should_be_snapped |= points.index.isin(
            sfilter(points, anchors.buffer(tolerance)).index
        )

    to_be_snapped = points.loc[should_be_snapped]

    anchors["_right_geom"] = anchors.geometry

    snapped = (
        to_be_snapped.sjoin_nearest(anchors, max_distance=tolerance)
        .sort_values("index_right")["_right_geom"]
        .loc[lambda x: ~x.index.duplicated()]
    )

    # explore(
    #     anchors,
    #     to_be_snapped,
    #     snapped=snapped,
    #     left_on_top=points.loc[lambda x: (~x.index.isin(left_on_top.values))],
    #     indices=points.loc[lambda x: (~x.index.isin(indices.values))],
    #     points_i_snap_to=points.set_crs(25833),
    # )

    points.loc[snapped.index, "geometry"] = snapped

    return points, anchors[["geometry"]]


def _snap_linearrings(
    geoms: NDArray[LinearRing],
    tolerance: int | float,
    mask: GeoDataFrame | None = None,
    snap_to_nodes: bool = True,
    gaps=None,
):
    if not len(geoms):
        return geoms

    if mask is None:
        idx_start = 0
    else:
        mask: GeoSeries = make_all_singlepart(mask).geometry
        mask_nodes = GeoDataFrame(
            {
                "geometry": extract_unique_points(mask.geometry),
                "_geom_idx": range(len(mask)),
            }
        ).explode(ignore_index=True)

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

    gdf = gdf.loc[is_thin == False]

    points: GeoDataFrame = gdf.assign(
        geometry=lambda x: extract_unique_points(x.geometry.values)
    ).explode(ignore_index=True)

    # step 1: add vertices nearest to mask nodes

    segments = points_to_line_segments(points.set_index("_geom_idx"))
    segments["_geom_idx"] = segments.index
    segments.index = points.index

    mask_nodes["rgeom"] = mask_nodes.geometry
    joined = segments.sjoin_nearest(mask_nodes, max_distance=tolerance)

    midpoints = shapely.get_point(
        shapely.shortest_line(joined.geometry.values, joined["rgeom"].values), 0
    )

    boundaries_groupby = joined.boundary.explode(index_parts=False).groupby(level=0)

    with_new_midpoints = (
        pd.concat(
            [
                # first point
                GeoSeries(boundaries_groupby.nth(0)),
                GeoSeries(midpoints, index=joined.index),
                # last point
                GeoSeries(boundaries_groupby.nth(-1)),
            ]
        )
        .groupby(level=0)
        .agg(lambda x: MultiPoint(x.values))
    )

    segments.loc[with_new_midpoints.index, "geometry"] = with_new_midpoints

    segments.geometry = extract_unique_points(segments.geometry)
    points = segments.explode(ignore_index=True)

    # step 2: snap to mask nodes

    points_by_mask_nodes = sfilter(
        points.loc[lambda x: x["_geom_idx"] >= idx_start], mask_nodes.buffer(tolerance)
    )

    relevant_mask_nodes = sfilter(
        mask_nodes,
        points_by_mask_nodes.buffer(tolerance),
        predicate="within",
    )
    # explore(
    #     relevant_mask_nodes,
    #     points_by_mask_nodes,
    #     points=points.set_crs(25833),
    #     mask=to_gdf([5.37166432, 59.00987036], 4326).to_crs(25833).buffer(100),
    # )

    # explore(
    #     mask,
    #     gdf,
    #     relevant_mask_nodes,
    #     points_by_mask_nodes,
    #     segments,
    #     points=points.set_crs(25833),
    #     mask=to_gdf([5.37166432, 59.00987036], 4326).to_crs(25833).buffer(100),
    # )

    if len(relevant_mask_nodes):
        mask_nodes["_right_geom"] = mask_nodes.geometry
        snapped = points_by_mask_nodes.sjoin_nearest(mask_nodes, max_distance=tolerance)

        anchors = GeoDataFrame(
            {"geometry": snapped.drop_duplicates("index_right")["_right_geom"].values}
        )

        snapmapper = snapped["_right_geom"].loc[lambda x: ~x.index.duplicated()]

        points.loc[snapmapper.index, "geometry"] = snapmapper
    else:
        anchors = None

    if snap_to_nodes:
        snapped, anchors = _snap_to_anchors(
            points, tolerance, anchors=mask_nodes
        )  # anchors)
    else:
        snapped = points

    # remove duplicates
    snapped = pd.concat(
        snapped.loc[lambda x: x["_geom_idx"] == i].loc[lambda x: ~x.duplicated()]
        for i in snapped.loc[
            lambda x: (x["_geom_idx"] >= idx_start), "_geom_idx"
        ].unique()
    )

    assert (snapped["_geom_idx"] >= idx_start).all()

    as_rings = (
        snapped.sort_index()
        .set_index("_geom_idx")
        # .pipe(_remove_legit_spikes)
        .loc[lambda x: x.groupby(level=0).size() > 2]
        .groupby(level=0)["geometry"]
        .agg(LinearRing)
    )

    as_polygons = GeoDataFrame(
        {"geometry": polygons(as_rings.values), "_geom_idx": as_rings.index}
    )

    slivers = as_polygons.loc[lambda x: x.buffer(-tolerance / 2).is_empty]
    snapped = snapped.loc[lambda x: ~x["_geom_idx"].isin(slivers["_geom_idx"])]

    as_polygons = update_geometries(sort_small_first(as_polygons))

    missing_mask_nodes = sfilter_inverse(
        mask_nodes, as_polygons.buffer(PRECISION)
    ).pipe(sfilter, as_polygons.buffer(PRECISION + tolerance))

    # explore(
    #     mask,
    #     gdf,
    #     anchors,
    #     missing_mask_nodes,
    #     snapped,
    #     as_polygons,
    #     points=points.set_crs(25833),
    #     mask=to_gdf([5.37166432, 59.00987036], 4326).to_crs(25833).buffer(100),
    # )

    if snap_to_nodes or len(missing_mask_nodes):
        thin_gaps = get_gaps(as_polygons, include_interiors=True).loc[
            lambda x: x.buffer(-tolerance / 2).is_empty
        ]
        thin_gaps.geometry = thin_gaps.buffer(-PRECISION).buffer(PRECISION)

        assert snapped.index.is_unique
        segments = points_to_line_segments(snapped.set_index("_geom_idx"))
        segments["_geom_idx"] = segments.index
        segments.index = snapped.index

        assert segments.index.is_unique

        segs_by_gaps = sfilter(
            segments,
            pd.concat([thin_gaps, slivers]).buffer(PRECISION),
        )
        # gap_nodes = pd.concat(
        #     [
        #         missing_mask_nodes,
        #         extract_unique_points(thin_gaps.geometry).to_frame("geometry"),
        #     ]
        # )

        # explore(
        #     # missing_mask_polygons,
        #     missing_mask_nodes,
        #     segs_by_gaps,
        #     thin_gaps,
        #     as_polygons=as_polygons,
        #     anchors=anchors.set_crs(25833),
        # )

        # segs_by_gaps = _add_midpoints_to_segments2(
        #     segs_by_gaps, points=gap_nodes, tolerance=tolerance
        # )

        segs_by_gaps.geometry = segmentize(segs_by_gaps.geometry, tolerance)
        segs_by_gaps.geometry = extract_unique_points(segs_by_gaps.geometry)
        assert segs_by_gaps.index.is_unique

        snapped = pd.concat(
            [snapped.loc[lambda x: ~x.index.isin(segs_by_gaps.index)], segs_by_gaps]
        ).sort_index()

        snapped = pd.concat(
            snapped.loc[lambda x: x["_geom_idx"] == i].loc[lambda x: ~x.duplicated()]
            for i in snapped["_geom_idx"].unique()
        ).explode(ignore_index=True)

        snapped, _ = _snap_to_anchors(snapped, tolerance, anchors=mask_nodes)

    as_rings = (
        snapped.loc[lambda x: (x["_geom_idx"] >= idx_start)]
        .sort_index()
        .set_index("_geom_idx")
        # .pipe(_remove_legit_spikes)
        .loc[lambda x: x.groupby(level=0).size() > 2]
        .groupby(level=0)["geometry"]
        .agg(LinearRing)
    )

    missing = gdf.set_index("_geom_idx")["geometry"].loc[
        lambda x: (~x.index.isin(as_rings.index)) & (x.index >= idx_start)
    ]
    missing.loc[:] = None

    return pd.concat([as_rings, thin, missing]).sort_index()


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

    return df.loc[lambda x: x["next"] != x["prev"]]


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


def split_spiky_polygons(
    gdf: GeoDataFrame, tolerance: int | float, grid_sizes: tuple[None | int] = (None,)
) -> GeoDataFrame:
    if not len(gdf):
        return gdf

    gdf = to_single_geom_type(make_all_singlepart(gdf), "polygon")

    if not gdf.index.is_unique:
        gdf = gdf.reset_index(drop=True)

    # remove both inwards and outwards spikes
    polygons_without_spikes = (
        gdf.buffer(-tolerance / 2, join_style=2)
        .buffer(tolerance, join_style=2)
        .buffer(-tolerance / 2, join_style=2)
    )

    donuts_around_polygons = to_lines(
        polygons_without_spikes.to_frame("geometry")
    ).pipe(buff, 1e-3, copy=False)

    # donuts_around_polygons["_poly_idx"] = donuts_around_polygons.index

    def _remove_spikes(df):
        df = df.to_frame("geometry")
        # df = df.reset_index(drop=True)
        df["_poly_idx"] = df.index
        df["_ring_idx"] = range(len(df))

        points = df.copy()
        points.geometry = extract_unique_points(points.geometry)
        points = points.explode(index_parts=False).explode(index_parts=False)
        points["_idx"] = range(len(points))

        # keep only matches from same polygon
        not_spikes = points.sjoin(donuts_around_polygons).loc[
            lambda x: x["_poly_idx"] == x["index_right"]
        ]
        can_be_polygons = not_spikes.iloc[
            (not_spikes.groupby("_ring_idx").transform("size") >= 3).values
        ]

        without_spikes = (
            can_be_polygons.sort_values("_idx")
            .groupby("_ring_idx")["geometry"]
            .agg(LinearRing)
        )

        missing = df.loc[
            ~df["_ring_idx"].isin(without_spikes.index), df._geometry_column_name
        ]
        return pd.concat(
            [without_spikes, missing]
        ).sort_index()  # .to_frame("geometry")

    without_spikes = GeoDataFrame(
        {
            "geometry": PolygonsAsRings(gdf.geometry)
            .apply_geoseries_func(_remove_spikes)
            .to_numpy()
        },
        crs=gdf.crs,
    ).pipe(to_single_geom_type, "polygon")
    without_spikes.index = gdf.index

    is_thin = without_spikes.buffer(-tolerance / 2).is_empty
    without_spikes = pd.concat(
        [
            split_by_neighbors(
                without_spikes[is_thin], without_spikes, tolerance=tolerance
            ),
            without_spikes[~is_thin],
        ]
    )

    # for _ in range(2):
    if 1:
        for i, grid_size in enumerate(grid_sizes):
            try:
                without_spikes = update_geometries(
                    sort_small_first(without_spikes), geom_type="polygon"
                )
                break
            except GEOSException as e:
                if i == len(grid_sizes) - 1:
                    raise e

    for i, grid_size in enumerate(grid_sizes):
        try:
            return clean_overlay(
                gdf, without_spikes, how="identity", grid_size=grid_size
            )
        except GEOSException as e:
            if i == len(grid_sizes) - 1:
                raise e
