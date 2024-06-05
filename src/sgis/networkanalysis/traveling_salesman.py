import networkx as nx
import pandas as pd
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from networkx.utils import pairwise
from shapely.geometry import Point

from ..geopandas_tools.conversion import to_geoseries
from ..geopandas_tools.neighbors import get_all_distances


def traveling_salesman_problem(
    points: GeoDataFrame | GeoSeries,
    return_to_start: bool = True,
    distances: pd.DataFrame | None = None,
) -> list[Point]:
    """Get the shortest path visiting all points.

    Args:
        points: An iterable of point geometries.
        return_to_start: If True (default), the path
            will make a full circle to the startpoint.
            If False, a dummy node will be added to make the
            salesman focus only on getting to the last node. Not
            guaranteed to work, meaning the wrong edge might be removed.
        distances: Optional DataFrame of distances between all points.
            If not provided, the calculation is done within this function.
            The DataFrame should be identical to the DataFrame created
            from sgis.get_all_distances from and to the points.

    Returns:
        List of Points making up the traveling salesman's path.

    Examples:
    ---------
    >>> import sgis as sg
    >>> from shapely.geometry import LineString
    >>> points = sg.to_gdf(
    ...     [
    ...         (0, 0),
    ...         (10, -10),
    ...         (10, 10),
    ...         (0, 10),
    ...         (0, -10),
    ...         (10, 0),
    ...         (20, 0),
    ...         (0, 20),
    ...     ]
    ... )
    >>> roundtrip = sg.traveling_salesman_problem(points)
    >>> roundtrip
    [<POINT (0 0)>, <POINT (10 -10)>, <POINT (0 -10)>, <POINT (10 0)>, <POINT (20 0)>, <POINT (10 10)>, <POINT (0 10)>, <POINT (0 20)>, <POINT (0 0)>]

    >>> LineString(roundtrip)
    LINESTRING (0 0, 10 -10, 0 -10, 10 0, 20 0, 10 10, 0 10, 0 20, 0 0)

    >>> oneway_trip = sg.traveling_salesman_problem(points, return_to_start=False)
    >>> oneway_trip
    [<POINT (0 0)>, <POINT (10 0)>, <POINT (20 0)>, <POINT (10 -10)>, <POINT (0 -10)>, <POINT (10 10)>, <POINT (0 10)>, <POINT (0 0)>]

    >>> LineString(oneway_trip)
    LINESTRING (0 20, 0 10, 10 10, 0 0, 10 0, 20 0, 10 -10, 0 -10)

    """
    points = to_geoseries(points).drop_duplicates()

    if len(points) <= 2:
        return list(points.dropna())

    if distances is None:
        points.index = range(len(points))
        distances: pd.DataFrame = get_all_distances(points, points)
    else:
        if not points.index.is_unique:
            raise ValueError("Index must be unique when passing 'distances'")

        distances = distances.loc[
            lambda x: (x.index.isin(points.index))
            & (x["neighbor_index"].isin(points.index))
        ]

        # need tange integer index
        to_int_idx = {idx: i for i, idx in enumerate(points.index)}
        points.index = points.index.map(to_int_idx)
        points = points.sort_index()
        distances.index = distances.index.map(to_int_idx)
        distances["neighbor_index"] = distances["neighbor_index"].map(to_int_idx)

    idx_to_point: dict[int, Point] = dict(enumerate(points))

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
        max_dist_and_some = distances["distance"].sum() * 1.01

        # add edges in both directions to the dummy node
        dummy_node = pd.DataFrame(
            {
                "neighbor_index": [i for i in range(n_points)]
                + [dummy_node_idx] * dummy_node_idx,
                "distance": [max_dist_and_some for _ in range(n_points * 2 - 1)],
            },
            index=[dummy_node_idx] * (n_points) + [i for i in range(dummy_node_idx)],
        )

        dummy_node.loc[
            lambda x: (x["neighbor_index"] == max_dist_idx)
            | (x.index == max_dist_idx)
            | (x["neighbor_index"] == x.index),
            "distance",
        ] = 0

        distances = pd.concat([distances, dummy_node])
    else:
        n_points = points.index.max() + 1

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

    as_points = [idx_to_point[i] for i in best_path if i != dummy_node_idx]

    return as_points  # + [as_points[0]]
