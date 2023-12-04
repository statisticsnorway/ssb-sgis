"""
The
"""
import functools
import itertools
import warnings
from typing import Callable, Iterable

import geopandas as gpd
import igraph
import networkx as nx
import numba
import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryArray
from IPython.display import display
from numpy import ndarray
from numpy.typing import NDArray
from pandas import Index
from pandas.core.groupby import SeriesGroupBy
from shapely import (
    Geometry,
    box,
    buffer,
    centroid,
    difference,
    distance,
    extract_unique_points,
    get_coordinates,
    get_exterior_ring,
    get_interior_ring,
    get_num_interior_rings,
    get_parts,
    get_rings,
    intersection,
    intersects,
    is_empty,
    is_ring,
    length,
    line_merge,
    linearrings,
    linestrings,
    make_valid,
    points,
    polygonize,
    polygons,
    segmentize,
    snap,
    unary_union,
)
from shapely.geometry import (
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.ops import nearest_points

from ..maps.maps import explore, explore_locals, qtm
from ..networkanalysis.closing_network_holes import (
    _close_holes_all_lines,
    close_network_holes,
    close_network_holes_to,
    get_angle,
)
from ..networkanalysis.cutting_lines import (
    change_line_endpoint,
    split_lines_by_nearest_point,
)
from ..networkanalysis.nodes import make_edge_wkt_cols, make_node_ids
from ..networkanalysis.traveling_salesman import traveling_salesman_problem
from .bounds import get_total_bounds, make_grid
from .buffer_dissolve_explode import (
    buff,
    buffdissexp_by_cluster,
    dissexp,
    dissexp_by_cluster,
)
from .centerlines import (
    get_line_segments,
    get_rough_centerlines,
    get_traveling_salesman_lines,
    multipoints_to_line_segments,
)
from .conversion import coordinate_array, to_gdf, to_geoseries
from .duplicates import get_intersections, update_geometries
from .general import clean_clip, clean_geoms
from .general import sort_large_first as sort_large_first_func
from .general import sort_long_first, to_lines
from .geometry_types import get_geom_type, make_all_singlepart, to_single_geom_type
from .neighbors import get_all_distances, k_nearest_neighbors
from .overlay import clean_overlay
from .polygon_operations import (
    close_all_holes,
    close_small_holes,
    close_thin_holes,
    eliminate_by_longest,
    get_cluster_mapper,
    get_gaps,
    get_holes,
)
from .polygons_as_rings import PolygonsAsRings
from .sfilter import sfilter, sfilter_inverse, sfilter_split


warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

mask = to_gdf(
    [
        "POINT (905200 7878700)",
        "POINT (905250 7878780)",
    ],
    25833,
).pipe(buff, 30)

PRECISION = 1e-4
BUFFER_RES = 50


def remove_spikes(gdf: GeoDataFrame, tolerance: int | float) -> GeoDataFrame:
    def _remove_spikes(
        geoms: NDArray[LinearRing], tolerance: int | float
    ) -> NDArray[LinearRing]:
        if not len(geoms):
            return geoms
        geoms = to_geoseries(geoms).reset_index(drop=True)

        def to_buffered_rings_without_spikes(x):
            polys = GeoSeries(make_valid(polygons(get_exterior_ring(x))))

            return (
                polys.buffer(-PRECISION * 2, resolution=BUFFER_RES)
                .explode(index_parts=False)
                # .buffer(PRECISION, resolution=BUFFER_RES)
                .pipe(close_all_holes)
                .pipe(get_exterior_ring)
                # .buffer(PRECISION, resolution=BUFFER_RES)
            )

        buffered = (
            geoms.buffer(PRECISION, resolution=BUFFER_RES).pipe(
                to_buffered_rings_without_spikes
            )  # Polygon(x.exterior))
            # .buffer(-PRECISION * 2, resolution=BUFFER_RES)
            # .explode(index_parts=False)
            # # .buffer(PRECISION, resolution=BUFFER_RES)
            # .pipe(close_all_holes)
            # .pipe(get_exterior_ring)
            # .buffer(PRECISION, resolution=BUFFER_RES)
        )
        if 0:
            for i, g in geoms.geometry.items():
                print(i, g)
            is_empty = buffered.is_empty
            print(geoms.index)
            print(buffered.index)
            print(is_empty)
            buffered.loc[is_empty] = geoms.loc[is_empty]

        points = extract_unique_points(geoms).explode(index_parts=False)
        # points_without_spikes = sfilter(points, buffered)
        points_without_spikes = points[
            points.distance(buffered.unary_union) <= PRECISION * 2
        ]
        if 0:
            explore(
                points,
                p6=points.loc[6],
                points_without_spikes=points_without_spikes,
                geoms=geoms,
                buffered=buffered,
            )
            print(points_without_spikes.index.value_counts())
            for i in points_without_spikes.index.unique():
                explore(
                    points_without_spikes.loc[i],
                    p0=points.loc[i],
                    geoms=geoms.loc[i],
                )
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
        PolygonsAsRings(gdf.geometry)
        .apply_numpy_func(_remove_spikes, args=(tolerance,))
        .to_numpy()
    )
    return gdf


def coverage_clean(
    gdf: GeoDataFrame,
    tolerance: int | float,
    duplicate_action: str = "fix",
) -> GeoDataFrame:
    """

    Rules:
    - Internal holes thinner than the tolerance are closed.
    - Polygons thinner than the tolerance are removed and filled
        by the surrounding polygons.
    - Line and point geometries are removed.
    - MultiPolygons are exploded to Polygons.
    - Index is reset.

    """

    _cleaning_checks(gdf, tolerance, duplicate_action)

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

    return (
        dissexp(pd.concat([gdf, intersected]), by="_poly_idx", aggfunc="first")
        .reset_index(drop=True)
        .loc[lambda x: ~x.buffer(-PRECISION).is_empty]
        .pipe(remove_spikes, tolerance=tolerance)
    )


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
    if not gdf.index.is_unique:
        raise ValueError("Index must be unique")
    if get_geom_type(gdf) != "polygon":
        raise ValueError("Must be polygons.")
    if tolerance < PRECISION:
        raise ValueError(
            f"'tolerance' must be larger than {PRECISION} to avoid "
            "problems with floating point precision."
        )
    if duplicate_action not in ["fix", "error"]:
        raise ValueError("duplicate_action must be 'fix' or 'error'")


def split_out_slivers(
    gdf: GeoDataFrame | GeoSeries, tolerance: float | int
) -> tuple[GeoDataFrame, GeoDataFrame] | tuple[GeoSeries, GeoSeries]:
    is_sliver = gdf.buffer(-tolerance / 2).is_empty
    slivers = gdf.loc[is_sliver]
    gdf = gdf.loc[~is_sliver]
    return gdf, slivers
