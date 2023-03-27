import warnings

import pandas as pd
from geopandas import GeoDataFrame
from igraph import Graph

from .network import _edge_ids


# run functions for get_route, get_k_routes and get_route_frequencies

# TODO: clean up this mess. Make smaller base functions and three separated for route, frequency and k_routes


def _get_route(
    graph: Graph,
    origins: GeoDataFrame,
    destinations: GeoDataFrame,
    weight: str,
    roads: GeoDataFrame,
    summarise: bool = False,
    rowwise: bool = False,
    k: int = 1,
    drop_middle_percent: int = 0,
):
    """Super function used in the NetworkAnalysis class.

    Big, ugly super function that is used in the get_route, get_k_routes
    and get_route_frequencies methods of the NetworkAnalysis class.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if k > 1:
        route_func = _run_get_k_routes
    else:
        route_func = _run_get_route

    resultlist: list[GeoDataFrame] = []
    if rowwise:
        for ori_id, des_id in zip(origins["temp_idx"], destinations["temp_idx"]):
            resultlist = resultlist + route_func(
                ori_id, des_id, graph, roads, summarise, weight, k, drop_middle_percent
            )
    else:
        for ori_id in origins["temp_idx"]:
            for des_id in destinations["temp_idx"]:
                resultlist = resultlist + route_func(
                    ori_id,
                    des_id,
                    graph,
                    roads,
                    summarise,
                    weight,
                    k,
                    drop_middle_percent,
                )

    if not resultlist:
        warnings.warn("No paths were found.")
        return pd.DataFrame(columns=["origin", "destination", weight, "geometry"])

    if summarise:
        counted = (
            pd.concat(resultlist, ignore_index=True)
            .assign(n=1)
            .groupby("source_target_weight")["n"]
            .count()
        )

        roads["source_target_weight"] = _edge_ids(roads, weight)

        roads["n"] = roads["source_target_weight"].map(counted)

        roads_visited = roads.loc[
            roads.n.notna(), roads.columns.difference(["source_target_weight"])
        ]

        return roads_visited

    try:
        results = pd.concat(resultlist)
    except Exception:
        raise ValueError(
            "No paths were found. Try larger search_tolerance or search_factor. "
            "Or close_network_holes() or remove_isolated()."
        )

    cols = ["origin", "destination", weight, "geometry"]
    if "k" in results.columns:
        cols.append("k")

    results = results.loc[:, cols].reset_index(drop=True)

    return results


def _run_get_route(
    ori_id: str,
    des_id: str,
    graph: Graph,
    roads: GeoDataFrame,
    summarise: bool,
    weight: str,
    k: int,
    drop_middle_percent: int,
) -> list[GeoDataFrame] | tuple[GeoDataFrame, list[tuple] | None]:
    res = graph.get_shortest_paths(
        weights="weight", v=ori_id, to=des_id, output="epath"
    )

    if not res[0]:
        return []

    source_target_weight = graph.es[res[0]]["source_target_weight"]

    if summarise:
        return [pd.DataFrame({"source_target_weight": source_target_weight})]

    roads["source_target_weight"] = _edge_ids(roads, weight)
    line = roads.loc[
        roads["source_target_weight"].isin(source_target_weight),
        ["geometry", weight, "source_target_weight"],
    ]

    # if len(line) != len(source_target_weight) - 2:
    #    raise ValueError("length mismatch", len(line), len(source_target_weight))

    if not len(line):
        return []

    weight_sum = line[weight].sum()
    line = line.dissolve()

    line["origin"] = ori_id
    line["destination"] = des_id
    line[weight] = weight_sum

    if k == 1:
        return [line]
    else:
        return [line], graph.es[res[0]]["edge_tuples"]


def _run_get_k_routes(
    ori_id: str,
    des_id: str,
    graph: Graph,
    roads: GeoDataFrame,
    summarise: bool,
    weight: str,
    k: int,
    drop_middle_percent,
) -> list[GeoDataFrame]:
    """Workaround for igraph's get_k_shortest_paths.

    igraph's get_k_shorest_paths doesn't seem to work (gives just the same path k
    times), so doing it manually. Run _run_get_route, then remove the edges in the
    middle of the route, given with drop_middle_percent, repeat k times.
    """
    graph = graph.copy()

    lines: list[GeoDataFrame] = []

    for i in range(k):
        line = _run_get_route(
            ori_id, des_id, graph, roads, summarise, weight, k, drop_middle_percent
        )

        if not isinstance(line, tuple):
            continue

        line, edge_tuples = line
        line = line[0]
        line["k"] = i + 1

        lines.append(line)

        n_edges_to_keep = (
            len(edge_tuples) - len(edge_tuples) * drop_middle_percent / 100
        ) / 2

        n_edges_to_keep = int(round(n_edges_to_keep, 0))

        if n_edges_to_keep == 0:
            n_edges_to_keep = 1

        to_be_dropped = edge_tuples[n_edges_to_keep:-n_edges_to_keep]
        graph.delete_edges(to_be_dropped)

    return lines
