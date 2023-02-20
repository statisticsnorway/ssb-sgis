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
    cost: str,
    roads: GeoDataFrame,
    summarise=False,
    cutoff: int = None,
    destination_count: int = None,
    rowwise=False,
):
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    lines = []
    if rowwise:
        for ori_id, des_id in zip(startpoints["temp_idx"], endpoints["temp_idx"]):
            lines = lines + _run_shortest_path(
                ori_id, des_id, graph, roads, cost, summarise
            )
    else:
        for ori_id in startpoints["temp_idx"]:
            for des_id in endpoints["temp_idx"]:
                lines = lines + _run_shortest_path(
                    ori_id, des_id, graph, roads, cost, summarise
                )

    if summarise:
        #        edges.groupby(["source", "target"])["n"].count()

        edges = pd.concat(lines, ignore_index=True)
        edges = edges.assign(n=1).groupby("source_target")["n"].count()

        roads2 = roads[["geometry", "source", "target"]]
        roads2["source_target"] = roads2.source + "_" + roads2.target

        return roads2.merge(edges, on="source_target", how="inner").drop(
            "source_target", axis=1
        )

    try:
        lines = gdf_concat(lines)
    except Exception:
        raise ValueError(
            f"No paths were found. Try larger search_tolerance or search_factor. "
            f"Or close_network_holes() or remove_isolated()."
        )

    if cutoff:
        lines = lines[lines[cost] < cutoff]

    if destination_count:
        lines = lines.loc[lines.groupby("origin")[cost].idxmin()].reset_index(drop=True)

    lines = lines[["origin", "destination", cost, "geometry"]]

    return lines.reset_index(drop=True)


def _run_shortest_path(
    ori_id, des_id, graph: Graph, roads: GeoDataFrame, cost: str, summarise: bool
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

    line = roads.loc[
        (roads.source.isin(path)) & (roads.target.isin(path)),
        ["geometry"],
    ]

    if not len(line):
        return []

    line = line.dissolve()

    line["origin"] = ori_id
    line["destination"] = des_id

    # for å få costen også
    kost = graph.distances(weights="weight", source=ori_id, target=des_id)
    line[cost] = kost[0][0]

    return [line]
