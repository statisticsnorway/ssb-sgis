import geopandas as gpd
import pandas as pd
import numpy as np

from .geopandas_utils import (
    clean_geoms,
    gdf_concat,
    to_gdf,
)

def shortest_path(
    nw,
    startpoints: gpd.GeoDataFrame,
    endpoints: gpd.GeoDataFrame,
    summarise=False,
    cutoff: int = None,
    destination_count: int = None,
    rowwise=False,
):
    import warnings

    warnings.filterwarnings(
        "ignore", category=RuntimeWarning
    )

    lines = []
    if rowwise:
        for ori_id, des_id in zip(startpoints["temp_idx"], endpoints["temp_idx"]):
            lines = lines + _run_shortest_path(ori_id, des_id, nw, summarise)
    else:
        for ori_id in startpoints["temp_idx"]:
            for des_id in endpoints["temp_idx"]:
                lines = lines + _run_shortest_path(ori_id, des_id, nw, summarise)

    if summarise:
        
#        edges.groupby(["source", "target"])["n"].count()

        edges = pd.concat(lines, ignore_index=True)
        edges = (edges
                .assign(n=1)
                .groupby("source_target")
                ["n"]
                .count()
        )

        roads = nw.network.gdf[["geometry", "source", "target"]]
        roads["source_target"] = roads.source + "_" + roads.target

        return roads.merge(edges, on="source_target", how="inner").drop(
            "source_target", axis=1
        )

    try:
        lines = gdf_concat(lines)
    except Exception:
        raise ValueError(
            f"No paths were found. Try larger search_tolerance or search_factor. Or close_network_holes() or remove_isolated()."
        )

    if cutoff:
        lines = lines[lines[nw.cost] < cutoff]

    if destination_count:
        lines = lines.loc[lines.groupby("origin")[nw.cost].idxmin()].reset_index(
            drop=True
        )

    lines = lines[["origin", "destination", nw.cost, "geometry"]]

    return lines.reset_index(drop=True)


def _run_shortest_path(ori_id, des_id, nw, summarise):

    res = nw.graph.get_shortest_paths(weights="weight", v=ori_id, to=des_id)

    if len(res[0]) == 0:
        return []

    path = nw.graph.vs[res[0]]["name"]

    if summarise:
        source_target = {
            "source_target": (
                str(source) + "_" + str(target)
                for source, target in zip(path[:-1], path[1:])
                )
                }
        return [pd.DataFrame(source_target)]

    line = nw.network.gdf.loc[
        (nw.network.gdf.source.isin(path)) & (nw.network.gdf.target.isin(path)),
        ["geometry"],
    ]

    if not len(line):
        return []

    line = line.dissolve()

    line["origin"] = ori_id
    line["destination"] = des_id

    # for å få costen også
    kost = nw.graph.distances(weights="weight", source=ori_id, target=des_id)
    line[nw.cost] = kost[0][0]

    return [line]