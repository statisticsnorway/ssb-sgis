import functools
import itertools
from typing import Callable, Iterable

import geopandas as gpd
import igraph
import networkx as nx
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryArray
from IPython.display import display
from networkx.algorithms import approximation as approx
from numpy import ndarray
from numpy.typing import NDArray
from pandas import Index
from shapely import (
    Geometry,
    STRtree,
    area,
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
    polygons,
    segmentize,
    simplify,
    unary_union,
    voronoi_polygons,
)
from shapely.errors import GEOSException
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
from ..networkanalysis.closing_network_holes import close_network_holes, get_angle
from ..networkanalysis.cutting_lines import split_lines_by_nearest_point
from .buffer_dissolve_explode import buff, dissexp_by_cluster
from .conversion import coordinate_array, to_gdf
from .duplicates import get_intersections
from .general import _push_geom_col, clean_geoms, get_common_crs, get_grouped_centroids
from .general import sort_large_first as sort_large_first_func
from .general import to_gdf, to_lines
from .geometry_types import get_geom_type, make_all_singlepart, to_single_geom_type
from .neighbors import (
    get_all_distances,
    get_k_nearest_neighbors,
    get_neighbor_indices,
    k_nearest_neighbors,
)
from .overlay import clean_overlay
from .polygon_operations import close_small_holes, close_thin_holes, get_gaps, get_holes
from .polygons_as_rings import PolygonsAsRings
from .sfilter import sfilter, sfilter_inverse, sfilter_split


def get_cheap_centerlines(
    gdf: GeoDataFrame,
    max_segment_length: int | None = None,
    # simplify_tolerance: int = 1,
    max_num_points: int = 800,
) -> GeoDataFrame:
    """Get a cheaply calculated centerline of a polygon.

    The line is not guaraneteed to be completely within the polygon.

    The function is meant for making centerlines of slivers in coverage_clean and
    snap_polygons. It will give weird results and be extremely slow for
     complext polygons like (buffered) road networks.

    """
    if not len(gdf):
        return gdf

    if max_segment_length is None:
        hull = gdf.convex_hull
        max_segment_length = (
            ((gdf.area / hull.area) + (hull.length / gdf.length))
            / 2
            * np.log(gdf.length + 1)
        )

        # max_segment_length = np.log(gdf.length + 1) + 0.05

    print(max_segment_length)

    segmentized = segmentize(gdf.geometry, max_segment_length=max_segment_length)

    voronoi_lines = voronoi_polygons(segmentized, only_edges=True)
    crossing_lines = (
        segmentized.buffer(0.01).intersection(voronoi_lines).explode(index_parts=False)
    )
    within_polygons, not_within = sfilter_split(crossing_lines, gdf, predicate="within")

    not_within_but_relevant = not_within.loc[
        lambda x: x.intersection(unary_union(get_rings(segmentized))).geom_type
        == "MultiPoint"
    ]

    # make sure to include the endpoints
    endpoints = get_approximate_polygon_endpoints(segmentized)

    centroids = pd.concat(
        [endpoints, within_polygons, not_within_but_relevant]
    ).centroid.to_frame("geometry")

    # centroids.index = endpoints.groupby(level=0).agg(unary_union)

    # add the endpoints as source and target columns
    endpoints = endpoints.loc[centroids.index]
    sources = endpoints.loc[lambda x: ~x.index.duplicated(keep="first")]
    targets = endpoints.loc[lambda x: ~x.index.duplicated(keep="last")]

    centroids["source"] = sources
    centroids["target"] = targets

    centerlines = (
        centroids.groupby(level=0).apply(get_shortest_line_between_points).explode()
    )

    intersect_length_ratio: NDArray[np.float] = length(
        intersection(centerlines.values, gdf.loc[centerlines.index].geometry.values)
    ) / length(centerlines.values)

    centerlines = (
        centerlines.loc[intersect_length_ratio > 0.3]
        .groupby(level=0)
        .agg(lambda x: line_merge(unary_union(x)))
    )

    def connect_multilines(multiline):
        lines = get_parts(multiline)
        lines_between = []
        for line in lines:
            lines_between += [
                LineString(nearest_points(line, multiline.difference(line)))
            ]
        return MultiLineString(list(lines) + lines_between)

    is_multiline = GeoSeries(centerlines).geom_type == "MultiLineString"
    centerlines.loc[is_multiline] = centerlines.loc[is_multiline].apply(
        connect_multilines
    )

    """not_simple = ~centerlines.is_simple
    if not_simple.any():
        pass"""

    # centerlines = simplify(centerlines, tolerance=precision)

    """explore(
        segmentized=to_gdf(segmentized, 25833),
        voronoi_lines=to_gdf(voronoi_lines, 25833),
        within_polygons=to_gdf(within_polygons, 25833),
        not_within_but_relevant=to_gdf(not_within_but_relevant, 25833),
        endpoints=to_gdf(endpoints, 25833),
        centroids=to_gdf(centroids.geometry, 25833),
        centerlines=to_gdf(centerlines, 25833),
    )"""

    centerlines = GeoDataFrame({"geometry": centerlines}, crs=gdf.crs)

    return centerlines


def get_approximate_polygon_endpoints(geoms: GeoSeries) -> GeoSeries:
    # TODO
    # la det være mulig med flere? hvis polygonet er et kors, trekant osv.

    out_geoms = []

    rectangles = geoms.minimum_rotated_rectangle()

    rectangles.loc[:] = get_rings(rectangles)

    corner_points = (
        extract_unique_points(rectangles)
        # adding a buffer, dissolve and explode to not get two corners in same end for very thin polygons
        .buffer(0.01)
        .groupby(level=0)
        .agg(unary_union)
        .explode(index_parts=False)
        .centroid
    )

    nearest_ring_point = nearest_points(corner_points, geoms.loc[corner_points.index])[
        1
    ]
    distance_to_corner = distance(nearest_ring_point, corner_points)

    is_two_nearest: NDArray[np.bool] = (
        distance_to_corner.groupby(level=0)
        .apply(lambda x: x <= x.nsmallest(2).iloc[-1])
        .values
    )

    two_nearest = nearest_ring_point.iloc[is_two_nearest]

    # geometries with more than two endpoints now, are probably crosses/star-like
    more_than_two = two_nearest.loc[lambda x: x.groupby(level=0).size() > 2]
    if len(more_than_two):
        precision = rectangles.length / 100

        two_nearest = two_nearest.loc[lambda x: ~x.index.isin(more_than_two.index)]

        by_rectangle = (
            geoms.intersection(rectangles.buffer(precision))
            .explode(index_parts=False)
            .centroid
        )

        nearest_rectangle_points = nearest_points(by_rectangle, rectangles)[1]
        nearest_geom_points = nearest_points(nearest_rectangle_points, geoms)[1]

        out_geoms.append(nearest_geom_points)

        explore(
            geoms=to_gdf(geoms, 25833),
            nearest_rectangle_points=to_gdf(nearest_rectangle_points, 25833),
            by_rectangle=to_gdf(by_rectangle, 25833),
            nearest_geom_points=to_gdf(nearest_geom_points, 25833),
        )

    lines_around_geometries = multipoints_to_line_segments(
        extract_unique_points(rectangles)
    )

    distance_to_rect = distance(
        two_nearest.values, rectangles.loc[two_nearest.index].values
    )

    # move the points to the rectangle
    to_be_moved = two_nearest.loc[distance_to_rect > 0.01]
    not_to_be_moved = two_nearest.loc[distance_to_rect <= 0.01]
    out_geoms.append(not_to_be_moved)

    if len(to_be_moved):
        tree = STRtree(lines_around_geometries.values)
        nearest_indices = tree.nearest(to_be_moved.values)

        to_be_moved.loc[:] = lines_around_geometries.iloc[nearest_indices].values

        # then move the points to the closest vertice
        to_be_moved.loc[:] = nearest_points(
            to_be_moved.values,
            extract_unique_points(geoms.loc[to_be_moved.index]).values,
        )[1]
        out_geoms.append(to_be_moved)

    explore(
        geoms=to_gdf(geoms, 25833),
        lines_around_geometries=to_gdf(lines_around_geometries, 25833),
        to_be_moved=to_gdf(to_be_moved, 25833),
        not_to_be_moved=to_gdf(not_to_be_moved, 25833),
        two_nearest=to_gdf(two_nearest, 25833),
    )

    return pd.concat(out_geoms)


def get_shortest_line_between_points(
    df: GeoDataFrame,
    max_num_points: int = 800,
    dissolve: bool = False,
    cycle: bool = False,
) -> LineString:
    points = df.geometry
    # source = df["source"].iloc[0]
    # target = df["target"].iloc[0]

    if len(points) <= 2:
        # nested if to save time
        if len(points) == 1:
            return points.iloc[0]
        return LineString(points.values)

    if len(points) > max_num_points:
        raise ValueError(
            f"Too many points {len(points)}. Traveling salesman calculation will be slow"
        )

    points.index = points.values
    distances: pd.DataFrame = get_all_distances(points, points).loc[
        lambda x: x.index != x["neighbor_index"]
    ]
    # distances: pd.DataFrame = get_k_nearest_neighbors(points, points, k=4)

    edges = list(
        zip(distances.index, distances["neighbor_index"], distances["distance"])
    )

    graph = nx.Graph()
    graph.add_weighted_edges_from(edges)

    path = nx.approximation.traveling_salesman_problem(
        graph, nodes=list(points.index), cycle=cycle
    )
    if not dissolve:
        return [LineString([p1, p2]) for p1, p2 in zip(path[:-1], path[1:])]

    return LineString(path)

    source_to_target = ((path[:-1] == source) & (path[1:] == target)) | (
        (path[:-1] == source) & (path[1:] == target)
    )
    # add False to the last point of the path
    source_to_target = np.concatenate([source_to_target, np.array([False])])

    return LineString(path[~source_to_target])


def multipoints_to_line_segments(multipoints: GeoSeries) -> GeoDataFrame:
    if not len(multipoints):
        return multipoints

    try:
        crs = multipoints.crs
    except AttributeError:
        crs = None

    points, indices = get_parts(multipoints, return_index=True)
    point_df = pd.DataFrame({"geometry": GeometryArray(points)}, index=indices)

    point_df["next"] = point_df.groupby(level=0)["geometry"].shift(-1)

    first_points = point_df.loc[lambda x: ~x.index.duplicated(), "geometry"]
    is_last_point = point_df["next"].isna()

    point_df.loc[is_last_point, "next"] = first_points
    assert point_df["next"].notna().all()

    lines = [
        LineString([x1, x2]) for x1, x2 in zip(point_df["geometry"], point_df["next"])
    ]
    return GeoSeries(lines, index=point_df.index, crs=crs)
