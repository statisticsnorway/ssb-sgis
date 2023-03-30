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
    if rowwise and len(origins) != len(destinations):
        raise ValueError(
            "'origins' and 'destinations' must have the same length when rowwise=True"
        )

    results = graph.distances(
        weights="weight",
        source=origins["temp_idx"],
        target=destinations["temp_idx"],
    )

    ori_idx, des_idx, costs = [], [], []
    for i, f_idx in enumerate(origins["temp_idx"]):
        for ii, t_idx in enumerate(destinations["temp_idx"]):
            ori_idx.append(f_idx)
            des_idx.append(t_idx)
            costs.append(results[i][ii])

    results = (
        pd.DataFrame(data={"origin": ori_idx, "destination": des_idx, weight: costs})
        .replace([np.inf, -np.inf], np.nan)
        .reset_index(drop=True)
    )

    # calculating all-to-all distances is much faster than looping rowwise,
    # so filtering to rowwise afterwards instead
    if rowwise:
        rowwise_df = DataFrame(
            {"origin": origins["temp_idx"], "destination": destinations["temp_idx"]}
        )
        results = rowwise_df.merge(results, on=["origin", "destination"], how="left")

    wkt_dict_origin = {
        idx: geom.wkt
        for idx, geom in zip(origins["temp_idx"], origins.geometry, strict=True)
    }
    wkt_dict_destination = {
        idx: geom.wkt
        for idx, geom in zip(
            destinations["temp_idx"], destinations.geometry, strict=True
        )
    }
    results["wkt_ori"] = results["origin"].map(wkt_dict_origin)
    results["wkt_des"] = results["destination"].map(wkt_dict_destination)

    results[weight] = np.where(results.wkt_ori == results.wkt_des, 0, results[weight])

    # straight lines between origin and destination
    if lines:
        origin = gpd.GeoSeries.from_wkt(results["wkt_ori"], crs=25833)
        destination = gpd.GeoSeries.from_wkt(results["wkt_des"], crs=25833)
        results["geometry"] = shortest_line(origin, destination)
        results = gpd.GeoDataFrame(results, geometry="geometry", crs=25833)

    results = results.drop(["wkt_ori", "wkt_des"], axis=1, errors="ignore")

    return results.reset_index(drop=True)
