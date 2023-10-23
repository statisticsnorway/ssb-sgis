import networkx as nx
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from networkx.utils import pairwise
from shapely.errors import GEOSException
from shapely.geometry import Point

from ..geopandas_tools.conversion import to_geoseries
from ..geopandas_tools.neighbors import get_all_distances


def traveling_salesman_problem(
    points: GeoDataFrame | GeoSeries,
    distances: pd.DataFrame | None = None,
    return_to_start: bool = True,
) -> list[Point]:
    points = to_geoseries(points).drop_duplicates()

    if len(points) <= 2:
        return list(points.dropna())

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
