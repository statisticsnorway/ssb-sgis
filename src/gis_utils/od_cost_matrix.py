import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from igraph import Graph
from pandas import DataFrame
from shapely import shortest_line


def od_cost_matrix(
    graph: Graph,
    startpoints: GeoDataFrame,
    endpoints: GeoDataFrame,
    cost: str,
    *,
    lines=False,
    rowwise=False,
    cutoff: int = None,
    destination_count: int = None,
) -> DataFrame | GeoDataFrame:
    """
    It takes a network, a GeoDataFrame of origins and a GeoDataFrame of destinations,
    and returns a GeoDataFrame with the shortest path between each origin and
    destination.

    Args:
      nw: the network object
      startpoints (GeoDataFrame): GeoDataFrame
      endpoints (GeoDataFrame): GeoDataFrame
      lines: If True, the output will be a GeoDataFrame with straight lines between
        origin and destination. Defaults to False.
      rowwise: Defaults to False
      cutoff (int): If you want to limit the maximum cost between origin and
        destination, you can set a cutoff.
      destination_count (int): int = None

    Returns:
      A dataframe with the origin, destination and cost.
    """

    if not rowwise:
        results = graph.distances(
            weights="weight",
            source=startpoints["temp_idx"],
            target=endpoints["temp_idx"],
        )

        ori_idx, des_idx, costs = [], [], []
        for i, f_idx in enumerate(startpoints["temp_idx"]):
            for ii, t_idx in enumerate(endpoints["temp_idx"]):
                ori_idx.append(f_idx)
                des_idx.append(t_idx)
                costs.append(results[i][ii])

    else:
        ori_idx, des_idx, costs = [], [], []
        for f_idx, t_idx in zip(startpoints["temp_idx"], endpoints["temp_idx"]):
            results = graph.distances(weights="weight", source=f_idx, target=t_idx)
            ori_idx.append(f_idx)
            des_idx.append(t_idx)
            costs.append(results[0][0])

    df = pd.DataFrame(data={"origin": ori_idx, "destination": des_idx, cost: costs})

    results = (
        df.replace([np.inf, -np.inf], np.nan)
        .loc[(df[cost] > 0) | (df[cost].isna())]
        .reset_index(drop=True)
    )

    if cutoff:
        results = results[results[cost] < cutoff]

    if destination_count:
        results = results.loc[~results[cost].isna()]
        cost_ranked = results.groupby("origin")[cost].rank()
        results = results.loc[cost_ranked <= destination_count]

    wkt_dict_origin = {
        idx: geom.wkt
        for idx, geom in zip(startpoints["temp_idx"], startpoints.geometry)
    }
    wkt_dict_destination = {
        idx: geom.wkt for idx, geom in zip(endpoints["temp_idx"], endpoints.geometry)
    }
    results["wkt_ori"] = results["origin"].map(wkt_dict_origin)
    results["wkt_des"] = results["destination"].map(wkt_dict_destination)

    results[cost] = np.where(results.wkt_ori == results.wkt_des, 0, results[cost])

    # straight lines between origin and destination
    if lines:
        origin = gpd.GeoSeries.from_wkt(results["wkt_ori"], crs=25833)
        destination = gpd.GeoSeries.from_wkt(results["wkt_des"], crs=25833)
        results["geometry"] = shortest_line(origin, destination)
        results = gpd.GeoDataFrame(results, geometry="geometry", crs=25833)

    results = results.drop(["wkt_ori", "wkt_des"], axis=1, errors="ignore")

    return results.reset_index(drop=True)
