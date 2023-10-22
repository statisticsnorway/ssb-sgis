import networkx as nx
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryArray
from networkx.utils import pairwise
from numpy.typing import NDArray
from shapely import (
    STRtree,
    distance,
    extract_unique_points,
    get_coordinates,
    get_parts,
    get_rings,
    line_merge,
    linestrings,
    make_valid,
    segmentize,
    unary_union,
    voronoi_polygons,
)
from shapely.errors import GEOSException
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

from .general import clean_geoms
from .neighbors import get_all_distances
from .sfilter import sfilter_split


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

    _, dont_intersect = sfilter_split(
        points.iloc[to_be_erased], end_to_end.buffer(precision, cap_style=2)
    )

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

    centerlines = centerlines3.groupby(level=0).agg(
        lambda x: line_merge(unary_union(x))
    )

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

    return linestrings(coords.values, indices=coords.index)


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

    if return_to_start:
        return [idx_to_point[i] for i in best_path]

    # drop duplicates, but keep order
    best_path = list(dict.fromkeys(best_path))

    idx_start = best_path.index(dummy_node_idx)  # - 1

    best_path = best_path[idx_start:] + best_path[:idx_start]

    return [idx_to_point[i] for i in best_path if i != dummy_node_idx]


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
