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
    extract_unique_points,
    force_2d,
    get_coordinates,
    get_exterior_ring,
    get_parts,
    linearrings,
    linestrings,
    make_valid,
    polygons,
    simplify,
    unary_union,
)
from shapely.errors import GEOSException
from shapely.geometry import LinearRing, LineString, Point

from ..networkanalysis.closing_network_holes import get_angle
from .buffer_dissolve_explode import buff, dissexp, dissexp_by_cluster
from .conversion import coordinate_array, to_geoseries
from .duplicates import get_intersections, update_geometries
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
    close_thin_holes,
    eliminate_by_longest,
    get_cluster_mapper,
    get_gaps,
)
from .polygons_as_rings import PolygonsAsRings
from .sfilter import sfilter, sfilter_inverse, sfilter_split


warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


PRECISION = 1e-4
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

    actually_to_eliminate = pd.concat(
        [without_double, not_really_isolated, really_isolated, isolated_gaps]
    )

    for i, grid_size in enumerate(grid_sizes):
        try:
            cleaned = (
                dissexp(
                    pd.concat(
                        [
                            gdf,
                            actually_to_eliminate,
                            # without_double,
                            # not_really_isolated,
                            # really_isolated,
                            # isolated_gaps,
                        ]
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
            # cleaned.geometry = shapely.simplify(
            #     cleaned.geometry, PRECISION * (10 * i + 1)
            # )
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

    # from ..maps.maps import explore
    # from .conversion import to_gdf

    # explore(
    #     gdf,
    #     gaps,
    #     double,
    #     actually_to_eliminate,
    #     slivers,
    #     thin_gaps_and_double,
    #     to_eliminate,
    #     isolated,
    #     intersecting,
    #     without_double,
    #     not_really_isolated,
    #     really_isolated,
    #     cleaned,
    #     to_eliminate_buff=buff(
    #         to_eliminate[["_eliminate_idx", "geometry"]],
    #         tolerance,
    #         resolution=BUFFER_RES,
    #     ),
    #     mask=to_gdf([5.36750884, 59.00830496], 4326).to_crs(25833).buffer(10),
    #     # browser=True,
    #     max_zoom=50,
    # )

    return cleaned


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
        from ..maps.maps import explore

        explore(
            gdf,
            # points,
            # not_spikes,
            without_spikes.buffer(1e-3).to_frame(),
            donuts_around_polygons,
            center=(56249, 6901798),
            size=100,
        )  # , without_spikes, donuts_around_polygons, polygons_without_spikes)

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


def split_up_slivers(
    slivers: GeoDataFrame,
    gdf: GeoDataFrame,
    tolerance: int | float,
    grid_sizes: tuple[None | int] = (None,),
) -> GeoDataFrame:
    if not len(slivers):
        return slivers

    nearby_lines = clean_overlay(
        to_lines(gdf),
        slivers.buffer(tolerance).to_frame("geometry"),
        keep_geom_type=True,
    )

    gdf = to_single_geom_type(make_all_singlepart(gdf), "polygon")

    if not gdf.index.is_unique:
        gdf = gdf.reset_index(drop=True)

    polygons_without_spikes = gdf.buffer(tolerance / 2, join_style=2).buffer(
        -tolerance / 2, join_style=2
    )

    donuts_around_polygons = to_lines(
        polygons_without_spikes.to_frame("geometry")
    ).pipe(buff, 1e-3, copy=False)

    def _remove_spikes(df):
        df = df.to_frame("geometry")
        df["_ring_idx"] = range(len(df))
        df = df.reset_index(drop=True)

        points = df.copy()
        points.geometry = extract_unique_points(points.geometry)
        points = points.explode(index_parts=False)
        points["_idx"] = range(len(points))

        not_spikes = points.sjoin(donuts_around_polygons).loc[
            lambda x: x["_ring_idx"] == x["index_right"]
        ]
        can_be_polygons = not_spikes.iloc[
            (not_spikes.groupby("_ring_idx").transform("size") >= 3).values
        ]

        without_spikes = (
            can_be_polygons.sort_values("_idx")
            .groupby("_ring_idx")["geometry"]
            .agg(LinearRing)
        )

        missing = df[~df["_ring_idx"].isin(without_spikes.index)].geometry

        return pd.concat([without_spikes, missing]).sort_index()

    without_spikes = GeoDataFrame(
        {
            "geometry": PolygonsAsRings(gdf.geometry)
            .apply_geoseries_func(_remove_spikes)
            .to_numpy()
        },
        crs=gdf.crs,
    ).pipe(to_single_geom_type, "polygon")

    is_thin = without_spikes.buffer(-tolerance / 2).is_empty
    without_spikes = pd.concat(
        [
            split_by_neighbors(
                without_spikes[is_thin], without_spikes, tolerance=tolerance
            ),
            without_spikes[~is_thin],
        ]
    )

    for _ in range(2):
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


def remove_spikes(gdf: GeoDataFrame, tolerance: int | float) -> GeoDataFrame:
    """Remove thin spikes in polygons.

    Note that this function might be slow. Should only be used if nessecary.

    Args:
        gdf: GeoDataFrame of polygons
        tolerance: distance (usually meters) used as the minimum thickness
            for polygons to be eliminated. Any spike thinner than the tolerance
            will be removed.

    Returns:
        A GeoDataFrame of polygons without spikes thinner.
    """

    gdf.geometry = (
        PolygonsAsRings(gdf.geometry)
        .apply_numpy_func(_remove_spikes, args=(tolerance,))
        .to_numpy()
    )
    return gdf


def remove_spikes(gdf: GeoDataFrame, tolerance: int | float) -> GeoDataFrame:
    return clean_overlay(
        gdf, gdf[["geometry"]], how="intersection", grid_size=tolerance
    )


def _remove_spikes(
    geoms: NDArray[LinearRing], tolerance: int | float
) -> NDArray[LinearRing]:
    if not len(geoms):
        return geoms
    geoms = to_geoseries(geoms).reset_index(drop=True)

    points = (
        extract_unique_points(geoms).explode(index_parts=False).to_frame("geometry")
    )

    points = get_angle_between_indexed_points(points)

    def to_buffered_rings_without_spikes(x):
        polys = GeoSeries(make_valid(polygons(get_exterior_ring(x))))

        return (
            polys.buffer(-tolerance, resolution=BUFFER_RES)
            .explode(index_parts=False)
            .pipe(close_all_holes)
            .pipe(get_exterior_ring)
            .buffer(tolerance * 10)
        )

    buffered = to_buffered_rings_without_spikes(
        geoms.buffer(tolerance / 2, resolution=BUFFER_RES)
    )

    points_without_spikes = (
        extract_unique_points(geoms)
        .explode(index_parts=False)
        .loc[lambda x: x.index.isin(sfilter(x, buffered).index)]
    )

    # linearrings require at least 4 coordinate pairs, or three unique
    points_without_spikes = points_without_spikes.loc[
        lambda x: x.groupby(level=0).size() >= 3
    ]

    # need an index from 0 to n-1 in 'linearrings'
    to_int_index = {
        ring_idx: i
        for i, ring_idx in enumerate(sorted(set(points_without_spikes.index)))
    }
    int_indices = points_without_spikes.index.map(to_int_index)

    as_lines = pd.Series(
        linearrings(
            get_coordinates(points_without_spikes.geometry.values),
            indices=int_indices,
        ),
        index=points_without_spikes.index.unique(),
    )

    # the missing polygons are thin and/or spiky. Let's remove them
    missing = geoms.loc[~geoms.index.isin(as_lines.index)]

    missing = pd.Series(
        [None] * len(missing),
        index=missing.index.values,
    )

    return pd.concat([as_lines, missing]).sort_index()


def get_angle_between_indexed_points(point_df: GeoDataFrame):
    """ "Get angle difference between the two lines"""

    point_df["next"] = point_df.groupby(level=0)["geometry"].shift(-1)

    notna = point_df["next"].notna()

    this = coordinate_array(point_df.loc[notna, "geometry"].values)
    next_ = coordinate_array(point_df.loc[notna, "next"].values)

    point_df.loc[notna, "angle"] = get_angle(this, next_)
    point_df["prev_angle"] = point_df.groupby(level=0)["angle"].shift(1)

    point_df["angle_diff"] = point_df["angle"] - point_df["prev_angle"]

    return point_df


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


def get_line_segments(lines) -> GeoDataFrame:
    assert lines.index.is_unique
    if isinstance(lines, GeoDataFrame):
        multipoints = lines.assign(
            **{
                lines._geometry_column_name: force_2d(
                    extract_unique_points(lines.geometry.values)
                )
            }
        )
        return multipoints_to_line_segments(multipoints.geometry)

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
        if isinstance(point_df, GeoSeries):
            point_df = point_df.to_frame("geometry")
    except AttributeError:
        points, indices = get_parts(multipoints, return_index=True)
        if isinstance(multipoints.index, pd.MultiIndex):
            indices = pd.MultiIndex.from_arrays(indices, names=multipoints.index.names)

        point_df = pd.DataFrame({"geometry": GeometryArray(points)}, index=indices)

    point_df["next"] = point_df.groupby(level=0)["geometry"].shift(-1)

    first_points = point_df.loc[lambda x: ~x.index.duplicated(), "geometry"]
    is_last_point = point_df["next"].isna()

    point_df.loc[is_last_point, "next"] = first_points
    assert point_df["next"].notna().all()

    point_df["geometry"] = [
        LineString([x1, x2]) for x1, x2 in zip(point_df["geometry"], point_df["next"])
    ]
    return GeoDataFrame(point_df.drop(columns=["next"]), geometry="geometry", crs=crs)


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


# hei


# %%

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd


src = str(Path(__file__).parent).strip("tests") + "src"


sys.path.insert(0, src)

import sgis as sg


def _buff(df):
    return sg.buff(df, 0.001)


import itertools

import geopandas as gpd
import igraph
import networkx as nx
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryArray
from IPython.display import display
from numpy import ndarray
from pandas import Index
from shapely import (
    STRtree,
    buffer,
    distance,
    extract_unique_points,
    get_coordinates,
    get_parts,
    make_valid,
    polygons,
    segmentize,
    set_precision,
)
from shapely.geometry import (
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
)
from shapely.ops import nearest_points

from ..maps.maps import explore, explore_locals, qtm
from ..networkanalysis.cutting_lines import split_lines_by_nearest_point
from .buffer_dissolve_explode import buff, dissexp_by_cluster
from .conversion import coordinate_array, to_gdf
from .duplicates import get_intersections
from .general import _determine_geom_type_args, clean_geoms, sort_large_first, to_lines
from .geometry_types import get_geom_type, make_all_singlepart, to_single_geom_type
from .neighbors import get_all_distances, k_nearest_neighbors
from .overlay import clean_overlay
from .polygon_operations import close_small_holes, get_holes
from .polygons_as_rings import PolygonsAsRings
from .sfilter import sfilter_inverse


PRECISION = 1e-3


def coverage_clean(
    gdf: GeoDataFrame,
    tolerance: int | float,
    mask: GeoDataFrame | GeoSeries | Geometry | None = None,
    grid_sizes=(None,),
    allowed_missing_area=1e-3,
    max_iterations: int = 3,
    # check_for=("double", "gaps", "missing", "slivers"),
    errors="ignore",
    **kwargs,
):
    if errors not in ["ignore", "raise"]:
        raise ValueError("'errors' should be 'raise' or 'ignore'")

    # gdf = sort_large_first(make_all_singlepart(clean_geoms(gdf), ignore_index=True))
    gdf = (
        clean_geoms(gdf)
        .pipe(make_all_singlepart, ignore_index=True)
        .pipe(
            simplify_and_put_small_on_top,
            PRECISION,
        )
    )

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

    gaps = get_gaps(gdf).loc[
        lambda x: (x.area > 0)
        & (simplify(x.geometry, tolerance).buffer(-tolerance / 2).is_empty)
    ]
    gaps["_was_gap"] = 1
    # gdf["_was_gap"] = 0

    # gdf = pd.concat([gdf, gaps])

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

        # gdf, slivers = split_out_slivers(gdf, tolerance)
        # gdf["_is_sliver"] = 1

        gdf.geometry = make_valid(
            PolygonsAsRings(gdf.geometry.values)
            .apply_numpy_func(
                _snap_linearrings,
                kwargs=dict(
                    tolerance=tolerance,
                    mask=mask,
                    gaps=gaps,
                ),
            )
            .to_numpy()
        )

        gdf = to_single_geom_type(make_all_singlepart(gdf), "polygon")

        try:
            gdf = update_geometries(
                close_small_holes(gdf, PRECISION), geom_type="polygon"
            )
            # gdf = update_geometries(gdf.sort_values("_was_gap"), geom_type="polygon")
        except GEOSException:
            failed = True

        # try:
        #     gdf = clean_overlay(copy, gdf, how="update", geom_type="polygon")
        # except GEOSException:
        #     pass

        gaps = get_gaps(gdf).loc[
            lambda x: (x.area > 0)
            & (simplify(x.geometry, tolerance).buffer(-tolerance / 2).is_empty)
        ]
        gaps["_was_gap"] = 1

        double = get_intersections(gdf, geom_type="polygon")
        try:
            missing = clean_overlay(
                copy, gdf, how="difference", geom_type="polygon"
            ).pipe(sg.clean_clip, mask, geom_type="polygon")
        except GEOSException:
            continue

        gdf, slivers = split_out_slivers(gdf, tolerance)

        # sg.explore(
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
            ).pipe(sg.clean_clip, mask, geom_type="polygon")
        except GEOSException:
            continue

        if (
            (not len(gaps) or gaps.area.max() < allowed_missing_area)
            and (not len(double) or double.area.max() < allowed_missing_area)
            and (not len(missing) or missing.area.max() < allowed_missing_area)
        ):
            break

        slivers = gdf.loc[gdf.buffer(-tolerance / 2).is_empty]

        # sg.explore(
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


def sorted_unary_union(geoms: GeoSeries) -> LineString:
    points = get_parts(extract_unique_points(unary_union(geoms.buffer(PRECISION))))
    first_point = points[0]
    last_point = points[-1]

    coords: ndarray = get_coordinates(points)
    sorted_coords = coords[np.argsort(coords[:, -1])]

    if first_point != Point(sorted_coords[0]):
        sorted_coords = sorted_coords[::-1]

    try:
        line = LineString(sorted_coords)
    except GEOSException as e:
        print(points)
        print(sorted_coords)
        raise e

    if first_point != Point(sorted_coords[0]):
        return LineString([first_point, last_point])

    if line.length < PRECISION:
        return first_point

    return LineString(sorted_coords)


def _snap_to_anchors(
    points: GeoDataFrame,
    tolerance: int | float,
    anchors: GeoDataFrame | None = None,
    # snap_indices: pd.Series | None = None,
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

    anchor_indices = set(
        indices.loc[
            lambda x: x.index.isin(points[points["_was_anchor"] == 1].index)
        ].index
    )

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

    agged = (
        # points.loc[lambda x: (x["_geom_idx"] >= idx_start) & (x["_is_thin"] != True)]
        points.loc[lambda x: (x["_is_thin"] != True)]
        .sort_index()
        .loc[lambda x: x.groupby("_geom_idx").transform("size") > 2]
        .groupby("_geom_idx")["geometry"]
        .agg(LinearRing)
        .reset_index()
    )

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
    gaps=None,
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
    gdf.geometry = segmentize(gdf.geometry, tolerance)

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
    )

    if mask is not None:
        mask_nodes = GeoDataFrame(
            {
                "geometry": extract_unique_points(mask.geometry),
                "_geom_idx": range(len(mask)),
            }
        )

        points = pd.concat([mask_nodes, points], ignore_index=True).explode(
            ignore_index=True
        )
        n_original_points = len(points) - len(mask_nodes)
    else:
        points = points.explode(ignore_index=True)
        n_original_points = len(points)

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

    sliver_points_on_outskirts = sfilter_inverse(
        points[lambda x: x["_is_thin"] == True],
        points[lambda x: x["_is_thin"] != True].buffer(PRECISION),
    )
    sliver_points = points[lambda x: x["_is_thin"] == True]

    # points = points[lambda x: x["_is_thin"] != True]

    sliver_points_on_outskirts = sfilter_inverse(
        sliver_points,
        points.buffer(PRECISION),
    )

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

    if 0:
        points_by_mask_nodes = sfilter(
            points.loc[lambda x: x["_geom_idx"] >= idx_start],
            mask_nodes.buffer(tolerance),
        )

        relevant_mask_nodes = sfilter(
            mask_nodes.explode(ignore_index=True),
            points_by_mask_nodes.buffer(tolerance),
            predicate="within",
        )

        # step 0: add new midpoints to edges near the mask

        segments = points_to_line_segments(points.set_index("_geom_idx"))
        segments["_geom_idx"] = segments.index
        segments.index = points.index
        segs_by_mask, not_by_mask = sfilter_split(segments, mask.buffer(tolerance))

        relevant_mask_nodes["rgeom"] = relevant_mask_nodes.geometry
        relevant_mask_nodes.geometry = relevant_mask_nodes.buffer(tolerance)
        joined = segs_by_mask.sjoin(relevant_mask_nodes)

        midpoints_from_mask_nodes = GeoSeries(
            nearest_points(joined.geometry.values, relevant_mask_nodes.unary_union)[0],
            index=joined.index,
        )

        segs_with_midpoints = (
            pd.concat(
                [
                    GeoSeries(segs_by_mask.boundary.groupby(level=0).nth(0)),
                    # midpoints_from_mask_edges,
                    midpoints_from_mask_nodes,
                    GeoSeries(segs_by_mask.boundary.groupby(level=0).nth(-1)),
                    # segments_by_mask_edges.geometry,
                ]
            )
            .groupby(level=0)
            .agg(sorted_unary_union)
        )

        segments.loc[segs_with_midpoints.index, "geometry"] = segs_with_midpoints

        # segments: GeoSeries = pd.concat([not_by_mask, segs_with_midpoints]).sort_index()

        # explore(
        #     segments,
        #     # relevant_mask_segments,
        #     relevant_mask_nodes,
        #     # midpoints_from_mask_edges,
        #     midpoints_from_mask_nodes,
        #     segs_with_midpoints.set_crs(25833),
        #     msk=to_gdf(mask),
        # )

        points: GeoDataFrame = segments.assign(
            geometry=lambda x: extract_unique_points(x.geometry.values)
        ).explode(ignore_index=True)

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
        mask_node_anchors = GeoSeries(
            nearest_points(
                points_by_mask_nodes.geometry.values, relevant_mask_nodes.unary_union
            )[1],
            index=points_by_mask_nodes.index,
        )
        points_by_mask_nodes.geometry = mask_node_anchors
    else:
        mask_node_anchors = GeoSeries()

    # step 2: snap from points to mask edges (between the nodes)

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
        nearest_points(points_by_mask.geometry.values, relevant_mask.unary_union)[1],
        index=points_by_mask.index,
    )
    points_by_mask.geometry = mask_anchors

    # step 3: snap to gdf nodes

    snapped_indices: pd.Index = points_by_mask_nodes.index.union(mask_anchors.index)
    assert snapped_indices.is_unique

    missing_sliver_points_on_outskirts = sfilter_inverse(
        sliver_points_on_outskirts.geometry,
        points.loc[snapped_indices].buffer(PRECISION),
    )

    anchors = pd.concat(
        [
            mask_node_anchors,
            mask_anchors,
        ]
    ).to_frame("geometry")

    #     points.loc[
    #         lambda x:
    #         # (~x.index.isin(snapped_indices)) &
    #         (x["_geom_idx"] >= idx_start)
    #     ],
    snapped_to_nodes, anchors = _snap_to_anchors(
        points.loc[
            lambda x: (~x.index.isin(snapped_indices)) & (x["_geom_idx"] >= idx_start)
        ],
        tolerance,
        anchors=anchors,
    )
    snapped_to_nodes = snapped_to_nodes.loc[lambda x: ~x.index.isin(snapped_indices)]
    # ).loc[lambda x: x["_was_snapped"] == 1]

    snapped_indices: pd.Index = snapped_indices.union(
        snapped_to_nodes[snapped_to_nodes["_was_snapped"] == 1].index
    )

    # explore(
    #     points_by_mask_nodes,
    #     points_by_mask,
    #     snapped_to_nodes,
    #     # thin,
    #     gdf,
    #     points,
    #     geoms=to_gdf([x for x in geoms if x is not None], 25833),
    # )

    points = (
        pd.concat([points_by_mask_nodes, points_by_mask, snapped_to_nodes])
        # pd.concat([points_by_mask_nodes, points_by_mask, snapped_to_nodes, points])
        # .loc[lambda x: ~x.index.duplicated()]
        # .loc[lambda x: x["_is_thin"] != True]
        .sort_index()
    )

    # points = pd.concat([points, sliver_points]).sort_index()

    # assert len(points) == n_original_points + len(mask_nodes)
    assert points.geometry.notna().all()
    assert points.index.is_unique, points[points.index.duplicated()]

    snapped = points

    # step 4: snap gdf edges together where there are no points
    # this reqires making new nodes in between

    # points = _add_points_between_nodes(points, tolerance * 1.5, mask_nodes)
    if 0:
        points = _add_midpoints(points, tolerance * 1.2)
        snapped, anchors = _snap_to_anchors(
            points,
            tolerance,
            anchors=anchors,
        )

    # snapped = _add_points_between_nodes(snapped, tolerance * 1.5, mask_nodes)
    # snapped, anchors = _snap_to_anchors(
    #     snapped,
    #     tolerance,
    #     anchors=anchors,
    # )

    # snapped_to_edges = snapped_to_edges.loc[
    #     # is_not_snapped
    #     lambda x: x["_was_snapped"]
    #     == 1
    # ]  # .loc[lambda x: ~x.index.isin(snapped_indices.union(snapped_to_nodes.index))]

    # snapped_to_edges.index = snapped_to_edges.index.map(not_snapped_index_mapper)

    # snapped = (
    #     pd.concat([snapped_to_edges, points]).loc[lambda x: x["_geom_idx"] >= idx_start]
    #     # .loc[lambda x: ~x.index.duplicated()]
    #     .sort_index()
    # )

    # print("\nhei etter edges")
    # explore(
    #     snapped,
    #     points,
    #     anchors,
    # )  # snapped_to_edges)

    # step 1: snap to mask nodes

    # points_by_mask_nodes = sfilter(
    #     snapped.loc[lambda x: x["_geom_idx"] >= idx_start], mask_nodes.buffer(tolerance)
    # )
    # relevant_mask_nodes = sfilter(
    #     mask_nodes.explode(ignore_index=True),
    #     points_by_mask_nodes.buffer(tolerance),
    #     predicate="within",
    # )

    # if len(relevant_mask_nodes):
    #     mask_node_anchors = GeoSeries(
    #         nearest_points(
    #             points_by_mask_nodes.geometry.values, relevant_mask_nodes.unary_union
    #         )[1],
    #         index=points_by_mask_nodes.index,
    #     )
    #     points_by_mask_nodes.geometry = mask_node_anchors
    # else:
    #     mask_node_anchors = GeoSeries()

    # # step 2: snap to mask edges (between the nodes)

    # mask_rings: GeoSeries = PolygonsAsRings(mask).get_rings().buffer(tolerance)

    # points_by_mask = sfilter(
    #     snapped.loc[
    #         lambda x: (x["_geom_idx"] >= idx_start)
    #         & (~x.index.isin(points_by_mask_nodes.index))
    #     ],
    #     mask_rings,
    # )
    # relevant_mask = mask_rings.clip(points_by_mask.buffer(tolerance))

    # mask_anchors = GeoSeries(
    #     nearest_points(points_by_mask.geometry.values, relevant_mask.unary_union)[1],
    #     index=points_by_mask.index,
    # )
    # points_by_mask.geometry = mask_anchors

    # snapped = pd.concat([points_by_mask_nodes, points_by_mask, snapped]).loc[
    #     lambda x: ~x.index.duplicated()
    # ]

    # assert snapped.index.is_unique

    # not_snapped = snapped.loc[
    #     lambda x: (~x.index.isin(snapped_indices.union(snapped_to_nodes.index)))
    #     & (x["_was_snapped"] != 1)
    # ]

    # def remove_duplicate_anchors(snap_indices):
    #     return snap_indices.loc[
    #         snap_indices.groupby(level=0).unique().apply(sorted).duplicated()
    #     ]

    # snapped_now = _snap_to_anchors(
    #     not_snapped,
    #     not_snapped,
    #     tolerance,
    #     idx_start=idx_start,
    #     custom_func=remove_duplicate_anchors,
    # )

    # snapped = pd.concat([points.loc[
    #     lambda x: (x.index.isin(snapped_indices.union(snapped_to_nodes.index)))
    #     | (x["_was_snapped"] == 1)
    # ], snapped_now]).sort_index()

    # snapped = pd.concat([points, snapped_to_edges]).loc[lambda x: x.index >= idx_start].sort_index()
    # assert (snapped["_geom_idx"] >= idx_start).all()
    # assert snapped.geometry.notna().all()

    # snapped = pd.concat(
    #     [
    #         snapped_to_edges,
    #         points.loc[
    #             lambda x: (x.index.isin(snapped_to_mask.index))
    #             | (x.index.isin(snapped_to_nodes.index))
    #         ],
    #     ]
    # )

    # edge_anchors = points.loc[points.index.isin(snap_vertice_index.index), "geometry"]
    # filt = points.index.isin(snap_vertice_index.values)  # ["_right_idx"])

    # explore(
    #     # snap_vertice_index,
    #     points_by_mask_nodes,
    #     points_by_mask,
    #     # snapped_to_nodes,
    #     with_new_midpoints=to_gdf(extract_unique_points(with_new_midpoints.geometry)),
    #     m=mask,
    #     mask_nodes=to_gdf(extract_unique_points(mask.geometry), 25833).buffer(0.3),
    #     mask_anchors=mask_anchors.to_frame().assign(wkt=lambda x: x.geometry.to_wkt()),
    #     gdf=gdf[lambda x: x["_geom_idx"] >= idx_start]
    #     .assign(wkt=lambda x: x.geometry.to_wkt())
    #     .set_crs(25833),
    #     points=points.assign(wkt=lambda x: x.geometry.to_wkt()),
    #     snapped=snapped.assign(wkt=lambda x: x.geometry.to_wkt()),
    #     # segments=segments.assign(wkt=lambda x: x.geometry.to_wkt()),
    #     # not_snapped=(points.loc[filt, "geometry"].values),
    #     # missing=clean_overlay(
    #     #     to_gdf(mask, 25833),
    #     #     gdf.assign(wkt=lambda x: x.geometry.to_wkt()).set_crs(25833),
    #     #     how="difference",
    #     #     geom_type="polygon",
    #     # ),
    #     # snapped=nearest_points(
    #     # points.loc[filt, "geometry"].values, anchors.unary_union
    #     # )[1],
    #     # mask=sg.to_gdf([5.36750884, 59.00830496], 4326).to_crs(25833).buffer(10),
    #     # missing_segments=missing_segments.assign(wkt=lambda x: x.geometry.to_wkt()),
    #     # points_in_index=points_in_index.assign(wkt=lambda x: x.geometry.to_wkt()),
    #     # to_change=points.loc[snap_points.index, "geometry"].values,
    #     # changed=points.loc[snap_points["_int_idx"].values, "geometry"].values,
    # )

    # to_be_snapped = points.loc[filt, "geometry"].values
    # snapped = nearest_points(to_be_snapped, edge_anchors.unary_union)[1]

    # dists = distance(snapped, to_be_snapped)
    # snapped = np.where(dists < tolerance, snapped, to_be_snapped)

    # points.loc[filt, "geometry"] = snapped

    # segs_by_mask = segs_by_mask.to_frame()
    # segs_by_mask["_geom_idx"] = segs_by_mask.index.map(points["_geom_idx"])

    # assert segs_by_mask._geom_idx.notna().all()

    # snapped = snapped.loc[
    #     lambda x: (x["_geom_idx"] >= idx_start)  # & (x["_is_thin"] != True)
    # ]

    snapped = (
        snapped.loc[lambda x: (x["_geom_idx"] >= idx_start)]
        # snapped.loc[lambda x: (x["_geom_idx"] >= idx_start) & (x["_is_thin"] != True)]
        .sort_index()
        .set_index("_geom_idx")
        # .pipe(_remove_legit_spikes)
        .loc[lambda x: x.groupby(level=0).size() > 2]
        .groupby(level=0)["geometry"]
        .agg(LinearRing)
    )

    # explore(
    #     snapped=to_gdf(snapped.buffer(0.01).unary_union, 25833),
    #     m=mask,
    #     # with_new_midpoints=with_new_midpoints,
    #     points_by_mask=points_by_mask,
    #     points_by_mask_nodes=points_by_mask_nodes,
    #     snapped_to_nodes=snapped_to_nodes,
    #     # segments=segments,
    #     # gdf=gdf,
    # )

    # explore(
    #     snapped=to_gdf(snapped, 25833),
    #     m=mask,
    #     points_by_mask=points_by_mask,
    #     points_by_mask_nodes=points_by_mask_nodes,
    #     snapped_to_nodes=snapped_to_nodes,
    #     segments=segments,
    #     gdf=gdf,
    # )

    missing = gdf.set_index("_geom_idx")["geometry"].loc[
        lambda x: (~x.index.isin(snapped.index))
        & (~x.index.isin(thin.index))
        & (x.index >= idx_start)
    ]

    # is_thin = (
    #     GeoSeries(polygons(snapped.geometry.values))
    #     .buffer(-tolerance / 2)
    #     .is_empty.values
    # )
    # print(snapped)
    # print(is_thin)
    # thin, thick = snapped[is_thin], snapped[~is_thin]

    # points = extract_unique_points(thin.geometry).explode(index_parts=False)
    # points = sfilter_inverse(points, thick)
    # all_coords_are_in_thick = ~snapped.index.isin(points.index)

    # snapped.loc[all_coords_are_in_thick] = None

    return pd.concat([snapped, missing]).sort_index()
    # return pd.concat([snapped, thin, missing]).sort_index()


def _add_points_between_nodes(points, tolerance, mask_nodes):
    points = pd.concat([mask_nodes.explode(ignore_index=True), points])

    segments = points_to_line_segments(points.set_index("_geom_idx"))
    segments["_geom_idx"] = segments.index
    segments.index = points.index

    assert len(segments) == len(points)

    # # # step 4: snap from edges to mask edges

    # edges_by_mask = sfilter(segments, mask_rings.buffer(tolerance * 1.25))
    # relevant_mask = mask_rings.clip(edges_by_mask.buffer(tolerance))

    # mask_anchors = GeoSeries(
    #     nearest_points(edges_by_mask.geometry.values, relevant_mask.unary_union)[1],
    #     index=points_by_mask.index,
    # )
    # edges_by_mask.geometry = mask_anchors

    not_snapped = segments  # [is_not_snapped]

    segs_to_join = segments.rename(columns={"_geom_idx": "_geom_idx_right"})
    segs_to_join["rgeom"] = segs_to_join.geometry
    segs_to_join.geometry = segs_to_join.buffer(tolerance)
    joined = not_snapped.sjoin(segs_to_join).loc[
        lambda x: x["_geom_idx"] != x["_geom_idx_right"]
    ]

    nearest_left, nearest_right = nearest_points(
        joined.geometry.values, joined["rgeom"].values
    )

    points_to_join = points.rename(columns={"_geom_idx": "_geom_idx_right"})
    points_to_join["rgeom"] = points_to_join.geometry
    points_to_join.geometry = points_to_join.buffer(tolerance)
    joined2 = not_snapped.sjoin(points_to_join).loc[
        lambda x: x["_geom_idx"] != x["_geom_idx_right"]
    ]

    nearest_points_left, nearest_points_right = nearest_points(
        joined2.geometry.values, joined2["rgeom"].values
    )

    with_new_midpoints = pd.concat(
        [
            # first point
            GeoSeries(joined.boundary.groupby(level=0).nth(0)),
            GeoSeries(
                joined["rgeom"].values,
                index=joined["index_right"],
            ),
            GeoSeries(nearest_left, index=joined.index),
            GeoSeries(nearest_right, index=joined["index_right"]),
            GeoSeries(nearest_points_left, index=joined2.index),
            GeoSeries(nearest_points_right, index=joined2["index_right"]),
            # last point
            GeoSeries(joined.boundary.groupby(level=0).nth(-1)),
        ]
    )
    with_new_midpoints_as_lines = (
        with_new_midpoints.loc[
            lambda x:
            # ~((x.index.duplicated() & (x.duplicated()))) &
            (x.index.isin(not_snapped.index))
        ]
        # .pipe(set_precision, PRECISION)
        # .loc[lambda x: x.index.isin(not_snapped.index)]
        .groupby(level=0).apply(sorted_unary_union)
    )

    # print("her inne")
    # explore(
    #     not_snapped,
    #     with_new_midpoints_as_lines,
    #     points,
    #     with_new_midpoints=with_new_midpoints.to_frame().set_crs(25833),
    # )

    not_snapped.loc[
        with_new_midpoints_as_lines.index, "geometry"
    ] = with_new_midpoints_as_lines

    not_snapped.geometry = extract_unique_points(not_snapped.geometry.values)
    not_snapped = not_snapped.explode(index_parts=False, ignore_index=False)

    return pd.concat(
        [points.loc[~points.index.isin(not_snapped.index)], not_snapped]
    ).sort_index()


def _add_midpoints(points, tolerance):
    segments = points_to_line_segments(points.set_index("_geom_idx"))
    segments["_geom_idx"] = segments.index
    segments.index = points.index

    assert len(segments) == len(points)

    segs_to_join = segments.rename(columns={"_geom_idx": "_geom_idx_right"})
    segs_to_join["rgeom"] = segs_to_join.geometry
    segs_to_join.geometry = segs_to_join.buffer(tolerance)
    joined = segments.sjoin(segs_to_join).loc[
        lambda x: x["_geom_idx"] != x["_geom_idx_right"]
    ]

    nearest_left, nearest_right = nearest_points(
        joined.geometry.values, joined["rgeom"].values
    )

    points_to_join = points.rename(columns={"_geom_idx": "_geom_idx_right"})
    points_to_join["rgeom"] = points_to_join.geometry
    points_to_join.geometry = points_to_join.buffer(tolerance)
    joined2 = segments.sjoin(points_to_join).loc[
        lambda x: x["_geom_idx"] != x["_geom_idx_right"]
    ]

    nearest_points_left, nearest_points_right = nearest_points(
        joined2.geometry.values, joined2["rgeom"].values
    )

    with_new_midpoints = pd.concat(
        [
            # first point
            GeoSeries(joined.boundary.groupby(level=0).nth(0)),
            GeoSeries(
                joined["rgeom"].values,
                index=joined["index_right"],
            ),
            GeoSeries(nearest_left, index=joined.index),
            GeoSeries(nearest_right, index=joined["index_right"]),
            GeoSeries(nearest_points_left, index=joined2.index),
            GeoSeries(nearest_points_right, index=joined2["index_right"]),
            # last point
            GeoSeries(joined.boundary.groupby(level=0).nth(-1)),
        ]
    )  # .loc[lambda x: ~((x.index.duplicated()) & (x.geometry.duplicated()))]
    with_new_midpoints_as_lines = (
        with_new_midpoints  # .loc[
        #     lambda x:
        #     # ~((x.index.duplicated() & (x.duplicated()))) &
        #     (x.index.isin(segments.index))
        # ]
        # .pipe(set_precision, PRECISION)
        # .loc[lambda x: x.index.isin(segments.index)]
        .groupby(level=0).apply(sorted_unary_union)
    )

    # print("her inne")
    # explore(
    #     segments,
    #     with_new_midpoints_as_lines,
    #     points,
    #     with_new_midpoints=with_new_midpoints.to_frame().set_crs(25833),
    # )

    segments.loc[
        with_new_midpoints_as_lines.index, "geometry"
    ] = with_new_midpoints_as_lines

    segments.geometry = extract_unique_points(segments.geometry.values)
    segments = segments.explode(index_parts=False, ignore_index=False)

    return pd.concat(
        [points.loc[~points.index.isin(segments.index)], segments]
    ).sort_index()


def _remove_legit_spikes(df):
    """Remove points where the next and previous points are the same.

    The lines these points make are as spiky as they come,
    hence the term "legit spikes".
    """
    return df
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

    print("_remove_legit_spikes")
    print(df)
    print(df.loc[lambda x: x["next"] != x["prev"]])
    explore(
        df.set_crs(25833),
        df.loc[lambda x: x["next"] != x["prev"]],
        df.loc[lambda x: x["next"] == x["prev"]],
    )

    return df.loc[lambda x: x["next"] != x["prev"]]


# %%
