"""
The
"""
import warnings

import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from numpy.typing import NDArray
from shapely import (
    extract_unique_points,
    get_coordinates,
    get_exterior_ring,
    linearrings,
    make_valid,
    polygons,
)
from shapely.geometry import LinearRing

from .buffer_dissolve_explode import buff, dissexp, dissexp_by_cluster
from .conversion import to_geoseries
from .duplicates import get_intersections, update_geometries
from .general import sort_long_first
from .geometry_types import get_geom_type
from .overlay import clean_overlay
from .polygon_operations import close_all_holes, close_thin_holes, get_gaps
from .polygons_as_rings import PolygonsAsRings
from .sfilter import sfilter_inverse


warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


PRECISION = 1e-4
BUFFER_RES = 50


def remove_spikes(gdf: GeoDataFrame, tolerance: int | float) -> GeoDataFrame:
    def _remove_spikes(geoms: NDArray[LinearRing]) -> NDArray[LinearRing]:
        if not len(geoms):
            return geoms
        geoms = to_geoseries(geoms).reset_index(drop=True)

        def to_buffered_rings_without_spikes(x):
            polys = GeoSeries(make_valid(polygons(get_exterior_ring(x))))

            return (
                polys.buffer(-tolerance * 2, resolution=BUFFER_RES)
                .explode(index_parts=False)
                .pipe(close_all_holes)
                .pipe(get_exterior_ring)
            )

        buffered = geoms.buffer(tolerance, resolution=BUFFER_RES).pipe(
            to_buffered_rings_without_spikes
        )

        points_without_spikes = (
            extract_unique_points(geoms)
            .explode(index_parts=False)
            .loc[lambda x: x.distance(buffered.unary_union) <= tolerance * 10]
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
        missing = geoms.loc[~geoms.index.isin(as_lines.index)]

        missing = pd.Series(
            [None] * len(missing),
            index=missing.index.values,
        )

        return pd.concat([as_lines, missing]).sort_index()

    gdf.geometry = (
        PolygonsAsRings(gdf.geometry).apply_numpy_func(_remove_spikes).to_numpy()
    )
    return gdf


def coverage_clean(
    gdf: GeoDataFrame,
    tolerance: int | float,
    duplicate_action: str = "fix",
) -> GeoDataFrame:
    """Fix thin gaps, holes, slivers and double surfaces.

    Rules:
    - Holes (interiors) thinner than the tolerance are closed.
    - Gaps between polygons are filled if thinner than the tolerance.
    - Sliver polygons thinner than the tolerance are eliminated
        into the neighbor polygon with the longest shared border.
    - Double surfaces thinner than the tolerance are eliminated.
        If duplicate_action is "fix", thicker double surfaces will
        be updated from top to bottom of the GeoDataFrame's rows.
    - Line and point geometries are removed.
    - MultiPolygons are exploded to Polygons.
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

    Returns:


    """

    _cleaning_checks(gdf, tolerance, duplicate_action)

    if not gdf.index.is_unique:
        gdf = gdf.reset_index(drop=True)

    gdf = close_thin_holes(gdf, tolerance)

    gaps = get_gaps(gdf, include_interiors=True)
    double = get_intersections(gdf)
    double["_double_idx"] = range(len(double))

    gdf, slivers = split_out_slivers(gdf, tolerance)

    thin_gaps_and_double = pd.concat([gaps, double]).loc[
        lambda x: x.buffer(-tolerance / 2).is_empty
    ]

    all_are_thin = double["_double_idx"].isin(thin_gaps_and_double["_double_idx"]).all()

    if not all_are_thin and duplicate_action == "fix":
        gdf = _dissolve_thick_double_and_update(gdf, double, thin_gaps_and_double)
        gdf, more_slivers = split_out_slivers(gdf, tolerance)
    elif not all_are_thin and duplicate_action == "error":
        raise ValueError("Large double surfaces.")
    else:
        more_slivers = GeoDataFrame({"geometry": []})

    to_eliminate = pd.concat(
        [thin_gaps_and_double, slivers, more_slivers], ignore_index=True
    ).loc[lambda x: ~x.buffer(-PRECISION).is_empty]
    gdf["_poly_idx"] = range(len(gdf))

    to_eliminate["_gap_idx"] = to_eliminate.index

    intersected = (
        buff(gdf, tolerance, resolution=BUFFER_RES)
        .pipe(clean_overlay, to_eliminate, geom_type="polygon")
        .pipe(sort_long_first)[["geometry", "_poly_idx"]]
        .pipe(update_geometries)
    )

    cleaned = (
        dissexp(pd.concat([gdf, intersected]), by="_poly_idx", aggfunc="first")
        .reset_index(drop=True)
        .loc[lambda x: ~x.buffer(-PRECISION).is_empty]
        .pipe(remove_spikes, tolerance=PRECISION)
    )

    missing_indices: pd.Index = sfilter_inverse(
        gdf.representative_point(), cleaned
    ).index

    missing = clean_overlay(
        gdf.loc[missing_indices].drop(columns="_poly_idx"),
        cleaned,
        how="difference",
        geom_type="polygon",
    )

    return pd.concat([cleaned, missing], ignore_index=True)


def _dissolve_thick_double_and_update(gdf, double, thin_double):
    large = (
        double.loc[~double["_double_idx"].isin(thin_double["_double_idx"])]
        .drop(columns="_double_idx")
        .pipe(dissexp_by_cluster)
    )
    return clean_overlay(gdf, large, how="update")


def _cleaning_checks(gdf, tolerance, duplicate_action):
    if not len(gdf) or not tolerance:
        return gdf
    if get_geom_type(gdf) != "polygon":
        raise ValueError("Must be polygons.")
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
    return gdf, slivers