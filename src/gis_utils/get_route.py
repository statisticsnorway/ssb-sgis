import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from igraph import Graph

from .geopandas_utils import gdf_concat
from .od_cost_matrix import _od_cost_matrix


def _get_route(
    graph: Graph,
    origins: GeoDataFrame,
    destinations: GeoDataFrame,
    weight: str,
    roads: GeoDataFrame,
    summarise=False,
    cutoff: int = None,
    destination_count: int = None,
    rowwise=False,
):
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # to get the actual cost, od_cost_matrix has to be run. Doing it first,
    # so that NaNs can be filtered out before the slower path calculation
    od_results = _od_cost_matrix(
        graph=graph,
        origins=origins,
        destinations=destinations,
        weight=weight,
        rowwise=rowwise,
        cutoff=cutoff,
        destination_count=destination_count,
    )

    od_notna = od_results.dropna()

    results = []
    if not rowwise:
        for ori_id, des_id in zip(od_notna["origin"], od_notna["destination"]):
            results = results + _run_get_route(ori_id, des_id, graph, roads, summarise)
    else:
        for ori_id, des_id in zip(origins["temp_idx"], destinations["temp_idx"]):
            if (
                ori_id not in od_notna["origin"]
                or des_id not in od_notna["destination"]
            ):
                continue

            results = results + _run_get_route(ori_id, des_id, graph, roads, summarise)

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

    results = results.merge(od_results, on=["origin", "destination"])[
        ["origin", "destination", weight, "geometry"]
    ].reset_index(drop=True)

    return results


def _run_get_route(ori_id, des_id, graph: Graph, roads: GeoDataFrame, summarise: bool):
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

    roads["source_target"] = roads["source"] + "_" + roads["target"]
    line = roads.loc[roads["source_target"].isin(source_target), ["geometry"]]

    if not len(line):
        return []

    line = line.dissolve()

    line["origin"] = ori_id
    line["destination"] = des_id

    return [line]
