import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from igraph import Graph

from .geopandas_utils import gdf_concat


def shortest_path(
    graph: Graph,
    startpoints: GeoDataFrame,
    endpoints: GeoDataFrame,
    weight: str,
    roads: GeoDataFrame,
    summarise=False,
    cutoff: int = None,
    destination_count: int = None,
    rowwise=False,
):
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    results = []
    if rowwise:
        for ori_id, des_id in zip(startpoints["temp_idx"], endpoints["temp_idx"]):
            results = results + _run_shortest_path(
                ori_id, des_id, graph, roads, weight, summarise
            )
    else:
        for ori_id in startpoints["temp_idx"]:
            for des_id in endpoints["temp_idx"]:
                results = results + _run_shortest_path(
                    ori_id, des_id, graph, roads, weight, summarise
                )

    if summarise:
        counted = (
            pd.concat(results, ignore_index=True)
            .assign(n=1)
            .groupby("source_target")["n"]
            .count()
        )

        roads = roads[["geometry", "source", "target"]]
        roads["source_target"] = roads.source + "_" + roads.target

        return roads.merge(counted, on="source_target", how="inner").drop(
            "source_target", axis=1
        )

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

    results = results[["origin", "destination", weight, "geometry"]]

    return results.reset_index(drop=True)


def _run_shortest_path(
    ori_id, des_id, graph: Graph, roads: GeoDataFrame, weight: str, summarise: bool
):
    res = graph.get_shortest_paths(weights="weight", v=ori_id, to=des_id)

    if len(res[0]) == 0:
        return []

    path = graph.vs[res[0]]["name"]

    if summarise:
        source_target = {
            "source_target": (
                str(source) + "_" + str(target)
                for source, target in zip(path[:-1], path[1:])
            )
        }
        return [pd.DataFrame(source_target)]

    source_target = [
        str(source) + "_" + str(target) for source, target in zip(path[:-1], path[1:])
    ]
    print(source_target)
    roads["source_target"] = roads["source"] + "_" + roads["target"]
    line = roads.loc[roads["source_target"].isin(source_target), ["geometry"]]

    if not len(line):
        return []

    line = line.dissolve()

    line["origin"] = ori_id
    line["destination"] = des_id

    # to get the cost:
    cost = graph.distances(weights="weight", source=ori_id, target=des_id)
    line[weight] = cost[0][0]

    return [line]
