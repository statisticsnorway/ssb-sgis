import warnings

import pandas as pd
from geopandas import GeoDataFrame
from igraph import Graph

from .geopandas_utils import gdf_concat
from .network import _edge_ids


# TODO: clean up this mess, make smaller base functions and three nicely separated
# run functions for get_route, get_k_routes and get_route_frequencies


def _get_route(
    graph: Graph,
    origins: GeoDataFrame,
    destinations: GeoDataFrame,
    weight: str,
    roads: GeoDataFrame,
    summarise=False,
    cutoff: int = None,
    destination_count: int = None,
    rowwise: bool = False,
    k=1,
    drop_middle_percent=0,
):
    """big, ugly super function that is used in the get_route, get_k_routes
    and get_route_frequencies methods of the NetworkAnalysis class

    """

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if k > 1:
        func = _run_get_k_routes
    else:
        func = _run_get_route

    results = []
    if rowwise:
        for ori_id, des_id in zip(origins["temp_idx"], destinations["temp_idx"]):
            results = results + func(
                ori_id, des_id, graph, roads, summarise, weight, k, drop_middle_percent
            )
    else:
        for ori_id in origins["temp_idx"]:
            for des_id in destinations["temp_idx"]:
                results = results + func(
                    ori_id,
                    des_id,
                    graph,
                    roads,
                    summarise,
                    weight,
                    k,
                    drop_middle_percent,
                )

    if summarise:
        counted = (
            pd.concat(results, ignore_index=True)
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
        results = gdf_concat(results)
    except Exception:
        raise ValueError(
            f"No paths were found. Try larger search_tolerance or search_factor. "
            f"Or close_network_holes() or remove_isolated()."
        )

    if cutoff:
        results = results[results[weight] < cutoff]

    if destination_count:
        results = results.loc[~results[weight].isna()]
        weight_ranked = results.groupby("origin")[weight].rank()
        results = results.loc[weight_ranked <= destination_count]

    cols = ["origin", "destination", weight, "geometry"]
    if "k" in results.columns:
        cols.append("k")

    results = results.loc[:, cols].reset_index(drop=True)

    return results


def _run_get_route(
    ori_id,
    des_id,
    graph: Graph,
    roads: GeoDataFrame,
    summarise: bool,
    weight: str,
    k: int,
    drop_middle_percent: int,
) -> list[GeoDataFrame] | list:
    res = graph.get_shortest_paths(
        weights="weight", v=ori_id, to=des_id, output="epath"
    )

    if not res[0]:
        return []

    source_target_weight = graph.es[res[0]]["source_target_weight"]

    if summarise:
        return [pd.DataFrame({"source_target_weight": source_target_weight})]

    roads["source_target_weight"] = _edge_ids(
        roads, weight
    )  # [f"{s}_{t}_{1}" for s, t, w in zip(roads["source"], roads["target"], roads[weight])]
    line = roads.loc[
        roads["source_target_weight"].isin(source_target_weight),
        ["geometry", weight, "source_target_weight"],
    ]

    if len(line) != len(source_target_weight) - 2:
        raise ValueError

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
    ori_id,
    des_id,
    graph: Graph,
    roads: GeoDataFrame,
    summarise: bool,
    weight,
    k,
    drop_middle_percent,
) -> list[GeoDataFrame]:
    """igraph's get_k_shorest_paths doesn't seem to work (gives just the same path k
    times), so doing it manually.

    Run _run_get_route, then remove the edges in the middle of the route, given with
    drop_middle_percent, repeat k times.
    """
    graph = graph.copy()
    lines = []
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

        keep_n = (len(edge_tuples) - len(edge_tuples) * drop_middle_percent / 100) / 2

        keep_n = int(round(keep_n, 0))

        keep_n = 1 if not keep_n else keep_n

        to_be_dropped = edge_tuples[keep_n:-keep_n]
        graph.delete_edges(to_be_dropped)

    return lines
