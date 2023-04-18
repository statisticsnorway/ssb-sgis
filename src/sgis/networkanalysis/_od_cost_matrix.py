import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from igraph import Graph
from pandas import DataFrame
from shapely import shortest_line


def _od_cost_matrix(
    graph: Graph,
    origins: GeoDataFrame,
    destinations: GeoDataFrame,
    weight: str,
    *,
    lines: bool = False,
    rowwise: bool = False,
) -> DataFrame | GeoDataFrame:
    assert origins.index.name == "temp_idx"
    assert destinations.index.name == "temp_idx"

    distances: list[list[float]] = graph.distances(
        weights="weight",
        source=origins.index,
        target=destinations.index,
    )

    ori_idx, des_idx, costs = [], [], []
    for i, f_idx in enumerate(origins.index):
        for j, t_idx in enumerate(destinations.index):
            ori_idx.append(f_idx)
            des_idx.append(t_idx)
            costs.append(distances[i][j])

    results = (
        pd.DataFrame(data={"origin": ori_idx, "destination": des_idx, weight: costs})
        .replace([np.inf, -np.inf], np.nan)
        .reset_index(drop=True)
    )

    # calculating all-to-all distances is much faster than looping rowwise,
    # so filtering to rowwise afterwards instead
    if rowwise:
        rowwise_df = DataFrame(
            {
                "origin": origins.index,
                "destination": destinations.index,
            }
        )
        results = rowwise_df.merge(results, on=["origin", "destination"], how="left")

    results["wkt_ori"] = results["origin"].map(origins.geometry)
    results["wkt_des"] = results["destination"].map(destinations.geometry)

    results.loc[results.wkt_ori == results.wkt_des, weight] = 0

    # straight lines between origin and destination
    if lines:
        results["geometry"] = shortest_line(results["wkt_ori"], results["wkt_des"])
        results = gpd.GeoDataFrame(results, geometry="geometry", crs=25833)

    results = results.drop(["wkt_ori", "wkt_des"], axis=1, errors="ignore")

    return results.reset_index(drop=True)
