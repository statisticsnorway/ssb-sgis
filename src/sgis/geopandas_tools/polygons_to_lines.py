import functools
import itertools
from typing import Callable, Iterable

import geopandas as gpd
import igraph
import networkx as nx
import numba
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryArray
from IPython.display import display
from networkx.algorithms import approximation as approx
from networkx.utils import pairwise
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
    get_num_coordinates,
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
from ..networkanalysis._od_cost_matrix import _get_od_df
from ..networkanalysis.closing_network_holes import close_network_holes, get_angle
from ..networkanalysis.cutting_lines import split_lines_by_nearest_point
from .buffer_dissolve_explode import buff, dissexp_by_cluster
from .conversion import coordinate_array, to_gdf
from .duplicates import get_intersections
from .general import _push_geom_col, clean_geoms, get_common_crs, get_grouped_centroids
from .general import sort_large_first as sort_large_first_func
from .general import sort_long_first, to_lines
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


def get_rough_centerlines(
    gdf: GeoDataFrame,
    max_segment_length: int | None = None,
) -> GeoDataFrame:
    """Get a cheaply calculated centerline of a polygon.

    The line is not guaraneteed to be completely within the polygon.

    The function is meant for making centerlines of slivers in coverage_clean and
    snap_polygons. It will give weird results and be extremely slow for
     complext polygons like (buffered) road networks.

    """

    precision = 0.01

    if not len(gdf):
        return gdf

    if not gdf.index.is_unique:
        raise ValueError("Index must be unique")

    geoms = gdf.geometry if isinstance(gdf, GeoDataFrame) else gdf

    """if max_segment_length is None:
        hull = geoms.convex_hull
        max_segment_length = (
            ((geoms.area / hull.area) + (hull.length / geoms.length))
            / 2
            * np.log2(geoms.length + 1)
            * np.log2(get_num_coordinates(geoms.geometry) + 1)
            / 10
        )

    max_segment_length = 1"""

    segmentized = segmentize(geoms.geometry, max_segment_length=max_segment_length)

    # voronoi can cause problems if coordinates are nearly identical
    # buffering solves it
    try:
        voronoi_lines = voronoi_polygons(segmentized, only_edges=True)
    except GEOSException:
        try:
            segmentized = make_valid(segmentized)
            voronoi_lines = voronoi_polygons(segmentized, only_edges=True)
        except GEOSException:
            voronoi_lines = voronoi_polygons(
                segmentized.buffer(precision).buffer(-precision), only_edges=True
            )

    crossing_lines = (
        segmentized.buffer(precision)
        .intersection(voronoi_lines)
        .explode(index_parts=False)
    )
    within_polygons, not_within = sfilter_split(
        crossing_lines, geoms, predicate="within"
    )

    intersect_polys_at_two_places = (
        not_within.intersection(unary_union(get_rings(segmentized))).geom_type
        == "MultiPoint"
    )
    not_within_but_relevant = not_within.loc[intersect_polys_at_two_places]

    points: GeoSeries = pd.concat([within_polygons, not_within_but_relevant]).centroid

    # Geometries that have no lines inside, might be perfect circles.
    # These can get the centroid as centerline
    has_no_points = geoms.loc[~geoms.index.isin(points.index)]
    has_no_points.loc[:] = geoms.loc[has_no_points.index].centroid

    segmentized = segmentized.loc[~segmentized.index.isin(has_no_points.index)]
    geoms = geoms.loc[~geoms.index.isin(has_no_points.index)]

    # make sure to include the endpoints
    endpoints = get_approximate_polygon_endpoints(segmentized)

    # check if line between endpoints make up a decent centerline
    has_two = endpoints.groupby(level=0).size() == 2
    endpoint1 = endpoints.loc[has_two].groupby(level=0).first()
    endpoint2 = endpoints.loc[has_two].groupby(level=0).last()
    assert (endpoint1.index == endpoint2.index).all()

    end_to_end = GeoSeries(
        make_lines_between_points(endpoint1, endpoint2), index=endpoint1.index
    )

    # keep lines 90 percent intersecting the polygon
    length_now = end_to_end.length
    end_to_end = (
        end_to_end.intersection(geoms.buffer(precision))
        .dropna()
        .loc[lambda x: x.length > length_now * 0.9]
    )

    # straight end buffer to remove all in between ends
    to_be_erased = points.index.isin(end_to_end.index)

    intersect, dont_intersect = sfilter_split(
        points.iloc[to_be_erased], end_to_end.buffer(precision, cap_style=2)
    )
    """
    centroids = nearest_points(
        geoms.centroid, intersect.groupby(level=0).agg(unary_union)
    )[1]"""

    """points.iloc[to_be_erased] = points.iloc[to_be_erased].difference(
        end_to_end.buffer(precision, cap_style=2)
    )"""

    """qtm(
        intersect=to_gdf(intersect, 25833),
        dont_intersect=to_gdf(dont_intersect, 25833),
        end_to_end=to_gdf(end_to_end, 25833),
        not_within=to_gdf(not_within, 25833),
        # centroids=to_gdf(centroids, 25833),
        # not_to_be_erased=to_gdf(points.iloc[~to_be_erased], 25833),
    )"""

    points = (
        clean_geoms(
            pd.concat(
                [
                    points.iloc[~to_be_erased],
                    dont_intersect,
                ]
            )
        )
        .buffer(0.1)
        .groupby(level=0)
        .agg(unary_union)
        .explode(index_parts=False)
        .centroid
    )
    points = pd.concat(
        [
            points,
            endpoints,
        ]
    )

    """
    index_mapper = dict(enumerate(points.index))
    point_mapper = dict(enumerate(points.values))
    points.index = range(len(points))
    neighbors = get_k_nearest_neighbors(points, points, k=5).loc[
        lambda x: x["distance"] > 0
    ]

    neighbors["lines"] = make_lines_between_points(
        neighbors.index.map(point_mapper).values,
        neighbors["neighbor_index"].map(point_mapper).values,
    )
    neighbors.index = neighbors.index.map(index_mapper)

    def tryhard_line_merge(x):
        return unary_union(line_merge(unary_union(x)))

    longest_lines = (
        GeoSeries(neighbors.groupby(level=0)["lines"].agg(tryhard_line_merge))
        .explode(index_parts=False)
        .pipe(sort_long_first)
        .loc[lambda x: ~x.index.duplicated()]
    )
    """

    def get_traveling_salesman_lines(df):
        path = traveling_salesman_problem(df, return_to_start=False)
        try:
            return [LineString([p1, p2]) for p1, p2 in zip(path[:-1], path[1:])]
        except IndexError as e:
            if len(path) == 1:
                return path
            raise e

    centerlines = GeoSeries(
        points.groupby(level=0).apply(get_traveling_salesman_lines).explode()
    )

    # fix sharp turns by using the centroids of the centerline
    centerlines2 = GeoSeries(
        (
            pd.concat(
                [
                    centerlines.centroid,
                    endpoints,
                ]
            )
            .groupby(level=0)
            .apply(get_traveling_salesman_lines)
        ).explode()
    )

    centerlines3 = GeoSeries(
        (
            pd.concat(
                [
                    centerlines2.centroid,
                    endpoints,
                ]
            )
            .groupby(level=0)
            .apply(get_traveling_salesman_lines)
        ).explode()
    )

    """centerlines3 = centerlines2.groupby(level=0).agg(
        lambda x: line_merge(unary_union(x))
    )
    centerlines3 = simplify(centerlines3, tolerance=tolerance / 10)"""

    explore(
        centerlines=to_gdf(centerlines, 25833),
        centerlines2=to_gdf(centerlines2, 25833),
        centerlines3=to_gdf(centerlines3, 25833),
        voronoi_lines=to_gdf(voronoi_lines, 25833),
        # within_polygons=to_gdf(within_polygons, 25833),
        # not_within_but_relevant=to_gdf(not_within_but_relevant, 25833).clip(
        #    g.buffer(5)
        # ),
        points=points.geometry.reset_index(),
        # has_no_points=to_gdf(has_no_points, 25833),
        segmentized=segmentized.reset_index(),
        endpoints=endpoints.reset_index(),
    )

    if 0:
        more_than_two = centerlines.loc[lambda x: x.groupby(level=0).size() > 2]
        one_or_two = centerlines.loc[lambda x: x.groupby(level=0).size() <= 2]
        print(more_than_two)
        print(more_than_two.groupby(level=0).apply(lambda x: x.length < x.length.max()))
        without_longest = more_than_two.iloc[
            lambda x: x.groupby(level=0)
            .apply(lambda x: x.length < x.length.max())
            .values
        ]
        print(without_longest)
        centerlines = pd.concat([one_or_two, without_longest])

    print(centerlines)

    # centerlines = centerlines.loc[without_longest]

    centerlines = centerlines3.groupby(level=0).agg(
        lambda x: line_merge(unary_union(x))
    )
    explore(centerlines=to_gdf(centerlines, 25833))

    """
    # try because pd.Series.explode doesn't take index_parts
    try:
        centerlines = centerlines.explode(index_parts=False)
    except Exception:
        centerlines = centerlines.explode()

    intersect_length_ratio: NDArray[float] = length(
        intersection(
            centerlines.values,
            geoms.loc[centerlines.index].buffer(precision * 2).values,
        )
    ) / length(centerlines.values)

    centerlines = GeoSeries(centerlines.loc[intersect_length_ratio > 0.1])

    is_ring = centerlines.is_ring
    if is_ring.any():
        without_longest = (
            centerlines.loc[is_ring]
            .groupby(level=0)
            .agg(lambda x: x.length != x.length.idxmax())
        )
        centerlines.loc[is_ring] = centerlines.loc[without_longest]

    centerlines = centerlines.groupby(level=0).agg(lambda x: line_merge(unary_union(x)))

    def connect_multilines(multiline):
        lines = get_parts(multiline)
        lines_between = []
        for line in lines:
            lines_between += [
                LineString(nearest_points(line, multiline.difference(line)))
            ]
        return MultiLineString(list(lines) + lines_between)

    is_multiline = centerlines.geom_type == "MultiLineString"
    centerlines.loc[is_multiline] = line_merge(
        centerlines.loc[is_multiline].apply(connect_multilines)
    )"""

    # simplify by twice the tolerance to not get sharp and short turns
    # centerlines = simplify(centerlines, tolerance=tolerance / 10)

    """not_simple = ~centerlines.is_simple
    if not_simple.any():
        centerlines.loc[not_simple] = simplify(
            centerlines.loc[not_simple],
            tolerance=centerlines.loc[not_simple].length / 100,
        )"""

    """explore(
        voronoi_lines=to_gdf(voronoi_lines, 25833),
        # within_polygons=to_gdf(within_polygons, 25833),
        # not_within_but_relevant=to_gdf(not_within_but_relevant, 25833).clip(
        #    g.buffer(5)
        # ),
        points=points.geometry.reset_index(),
        # has_no_points=to_gdf(has_no_points, 25833),
        segmentized=segmentized.reset_index(),
        centerlines=centerlines.reset_index(),
        endpoints=endpoints.reset_index(),
    )"""

    """for g in segmentized:
        qtm(
            voronoi_lines=to_gdf(voronoi_lines, 25833).clip(g.buffer(5)),
            # within_polygons=to_gdf(within_polygons, 25833).clip(g.buffer(5)),
            # not_within_but_relevant=to_gdf(not_within_but_relevant, 25833).clip(
            #    g.buffer(5)
            # ),
            points=points.geometry.reset_index().clip(g.buffer(5)),
            # has_no_points=to_gdf(has_no_points, 25833).clip(g.buffer(5)),
            segmentized=segmentized.reset_index().clip(g.buffer(5)),
            centerlines=centerlines.reset_index().clip(g.buffer(5)),
            endpoints=endpoints.reset_index().clip(g.buffer(5)),
        )"""

    if isinstance(gdf, GeoSeries):
        return GeoSeries(
            pd.concat([centerlines, has_no_points]), crs=gdf.crs
        ).sort_index()

    centerlines = GeoDataFrame(
        {"geometry": pd.concat([centerlines, has_no_points])}, crs=gdf.crs
    )

    return centerlines


def make_lines_between_points(
    arr1: NDArray[Point], arr2: NDArray[Point]
) -> NDArray[LineString]:
    if arr1.shape != arr2.shape:
        raise ValueError("Arrays must have equal shape.")
    coords: pd.DataFrame = pd.concat(
        [
            pd.DataFrame(get_coordinates(arr1), columns=["x", "y"]),
            pd.DataFrame(get_coordinates(arr2), columns=["x", "y"]),
        ]
    ).sort_index()

    try:
        return linestrings(coords.values, indices=coords.index)
    except Exception as e:
        print(arr1)
        print(arr2)
        print(coords[coords.notna()])
        print(coords[coords.isna()])
        raise e.__class__(e, coords)


def get_approximate_polygon_endpoints(geoms: GeoSeries) -> GeoSeries:
    out_geoms = []

    rectangles = geoms.minimum_rotated_rectangle()

    # get rings returns array with integer index that must be mapped to pandas index
    rings, indices = get_rings(rectangles, return_index=True)
    int_to_pd_index = dict(enumerate(sorted(set(rectangles.index))))
    indices = [int_to_pd_index[i] for i in indices]

    rectangles.loc[:] = (
        pd.Series(rings, index=indices).groupby(level=0).agg(unary_union)
    )

    corner_points = (
        GeoSeries(
            extract_unique_points(rectangles)
            # adding a buffer, dissolve and explode to not get two corners in same end for very thin polygons
            .buffer(0.01)
            .groupby(level=0)
            .agg(unary_union)
        )
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

        """explore(
            geoms=to_gdf(geoms, 25833),
            nearest_rectangle_points=to_gdf(nearest_rectangle_points, 25833),
            by_rectangle=to_gdf(by_rectangle, 25833),
            nearest_geom_points=to_gdf(nearest_geom_points, 25833),
        )"""

    lines_around_geometries = multipoints_to_line_segments(
        extract_unique_points(rectangles)
    )

    distance_to_rect = distance(
        two_nearest.values, rectangles.loc[two_nearest.index].values
    )

    # move the points to the rectangle
    to_be_moved = two_nearest.loc[distance_to_rect > 0.01]
    not_to_be_moved = two_nearest.loc[distance_to_rect <= 0.01]
    assert len(not_to_be_moved) + len(to_be_moved) == len(two_nearest)
    out_geoms.append(not_to_be_moved)

    explore(
        geoms=to_gdf(geoms, 25833),
        lines_around_geometries=to_gdf(lines_around_geometries, 25833),
        to_be_moved=to_gdf(to_be_moved, 25833),
        not_to_be_moved=to_gdf(not_to_be_moved, 25833),
        two_nearest=to_gdf(two_nearest, 25833),
        out_geoms=pd.concat(out_geoms).set_crs(25833),
    )

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

    return pd.concat(out_geoms)


def traveling_salesman_problem(
    points: GeoDataFrame | GeoSeries,
    distances: pd.DataFrame | None = None,
    return_to_start: bool = True,
) -> list[Point]:
    try:
        points = GeoSeries(points.geometry).drop_duplicates()
    except AttributeError:
        points = GeoSeries(points).drop_duplicates()

    if len(points) <= 2:
        return points

    if distances is None:
        idx_to_point: dict[int, Point] = dict(enumerate(points))
        points.index = range(len(points))
        distances: pd.DataFrame = get_all_distances(points, points)
    else:
        idx_to_point: dict[int, Point] = dict(enumerate(points))

        distances = distances.loc[
            lambda x: (x.index.isin(points.index))
            & (x["neighbor_index"].isin(points.index))
        ]

    if not return_to_start:
        distances["mean_distance"] = distances.groupby(level=0)["distance"].transform(
            "mean"
        )

        distances = distances.sort_values(
            ["mean_distance", "distance"], ascending=[True, False]
        )
        max_dist_idx = distances["mean_distance"].idxmax()

        dummy_node_idx = points.index.max() + 1
        n_points = dummy_node_idx + 1
        max_dist_and_some = distances["distance"].max() * 1.1
        dummy_node = pd.DataFrame(
            {
                "neighbor_index": [i for i in range(n_points)]
                + [dummy_node_idx] * dummy_node_idx,
                "distance": [max_dist_and_some for _ in range(n_points * 2 - 1)],
            },
            index=[dummy_node_idx] * (n_points) + [i for i in range(dummy_node_idx)],
        )

        dummy_node.loc[
            (dummy_node["neighbor_index"] == max_dist_idx)
            | (dummy_node.index == max_dist_idx)
            | (dummy_node["neighbor_index"] == dummy_node.index),
            "distance",
        ] = 0

        distances = pd.concat([distances, dummy_node])
    else:
        n_points = points.index.max()

    # now to mimick the return values of nx.all_pairs_dijkstra, nested dictionaries of distances and nodes/edges
    dist, path = {}, {}
    for i in distances.index.unique():
        dist[i] = dict(distances.loc[i, ["neighbor_index", "distance"]].values)
        path[i] = {
            neighbor: [i, neighbor] for neighbor in distances.loc[i, "neighbor_index"]
        }

    # the rest of the function is copied from networkx' traveling_salesman_problem

    nx_graph = nx.Graph()
    for u in range(n_points):
        for v in range(n_points):
            if u == v:
                continue
            nx_graph.add_edge(u, v, weight=dist[u][v])
    best = nx.approximation.christofides(nx_graph, "weight")

    best_path = []
    for u, v in pairwise(best):
        best_path.extend(path[u][v][:-1])
    best_path.append(v)

    """qtm(
        line=to_gdf(
            LineString([idx_to_point[i] for i in best_path if i != dummy_node_idx])
        ),
        # points=to_gdf(points),
    )"""

    if return_to_start:
        return [idx_to_point[i] for i in best_path]

    # drop duplicates, but keep order
    best_path = list(dict.fromkeys(best_path))

    idx_start = best_path.index(dummy_node_idx)  # - 1
    # print(dummy_node_idx)
    # print(max_dist_idx)
    # print(idx_start)
    # print(best_path)
    best_path = best_path[idx_start:] + best_path[:idx_start]
    """qtm(
        line=to_gdf(
            LineString([idx_to_point[i] for i in best_path if i != dummy_node_idx])
        ),
        # points=to_gdf(points),
    )"""
    # print(best_path)

    # best_path.pop(0)

    return [idx_to_point[i] for i in best_path if i != dummy_node_idx]

    # print(best_path)
    """qtm(
        line=to_gdf(
            LineString([idx_to_point[i] for i in best_path if i != dummy_node_idx])
        ),
        # points=to_gdf(points),
    )"""


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


def traveling_salesman_problem_nx(
    df: GeoDataFrame,
    dissolve: bool = False,
    cycle: bool = True,
) -> LineString:
    points = df.geometry.drop_duplicates()

    if len(points) <= 2:
        return list(points)

    points.index = points.values
    distances: pd.DataFrame = get_all_distances(points, points).loc[
        lambda x: x.index != x["neighbor_index"]
    ]

    edges = list(
        zip(distances.index, distances["neighbor_index"], distances["distance"])
    )

    # distance from all origins to all vertices/nodes in the graph
    if 0:
        import igraph

        edgemapper = {p: str(i) for i, p in enumerate(points.values)}

        graph = igraph.Graph.TupleList(
            [
                (edgemapper[s], edgemapper[t])
                for s, t in zip(distances.index, distances["neighbor_index"])
            ],
            directed=False,
        )
        graph.es["weight"] = list(distances["distance"])

        for i in edgemapper.values():
            res = graph.get_shortest_paths(
                weights="weight", v=i, to=list(edgemapper.values()), output="epath"
            )

    graph = nx.Graph()
    graph.add_weighted_edges_from(edges)

    return nx.approximation.traveling_salesman_problem(
        graph, nodes=list(points.index), cycle=cycle
    )

    print(traveling_salesman_problem(df))

    if not dissolve:
        return [LineString([p1, p2]) for p1, p2 in zip(path[:-1], path[1:])]

    return LineString(path)

    source_to_target = ((path[:-1] == source) & (path[1:] == target)) | (
        (path[:-1] == source) & (path[1:] == target)
    )
    # add False to the last point of the path
    source_to_target = np.concatenate([source_to_target, np.array([False])])

    return LineString(path[~source_to_target])


# print(c_lib.matte())
# sss


@numba.njit  # (parallel=True)
def get_shortest(distance_matrix, permutations):
    shortest = np.sum(distance_matrix)

    for perm in permutations:
        length = 0
        for p1, p2 in zip(perm[:-1], perm[1:]):
            length += distance_matrix[p1][p2]

        # length = sum([od_pairs[p1 + "_" + p2] for p1, p2 in zip(perm[:-1], perm[1:])])
        if length < shortest:
            shortest = length
            shortest_route = perm

    return shortest_route
