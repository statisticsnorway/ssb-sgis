import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import shortest_line
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame
import networkx as nx

def od_cost_matrix(
    nw,
    startpoints: GeoDataFrame,
    endpoints: GeoDataFrame,
    *,
    lines=False,
    rowwise=False,
    cutoff: int = None,
    destination_count: int = None,
) -> DataFrame | GeoDataFrame:

    """
    """

    cost = nw.cost
    
    if not rowwise:
        # selve avstandsberegningen her:
        results = nw.graph.distances(
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
            results = nw.graph.distances(weights="weight", source=f_idx, target=t_idx)
            ori_idx.append(f_idx)
            des_idx.append(t_idx)
            costs.append(results[0][0])

    df = pd.DataFrame(data={"origin": ori_idx, "destination": des_idx, nw.cost: costs})

    # litt opprydning
    out = (
        df.replace([np.inf, -np.inf], np.nan)
        .loc[(df[cost] > 0) | (df[cost].isna())]
        .reset_index(drop=True)
    )

    if cutoff:
        out = out[out[cost] < cutoff].reset_index(drop=True)

    if destination_count:
        out = out.loc[~out[cost].isna()]
        out = out.loc[out.groupby("origin")[cost].idxmin()].reset_index(drop=True)

    #
    wkt_dict_origin = {
        idd: geom.wkt
        for idd, geom in zip(startpoints["temp_idx"], startpoints.geometry)
    }
    wkt_dict_destination = {
        idd: geom.wkt
        for idd, geom in zip(endpoints["temp_idx"], endpoints.geometry)
    }
    out["wkt_ori"] = out["origin"].map(wkt_dict_origin)
    out["wkt_des"] = out["destination"].map(wkt_dict_destination)
    out[nw.cost] = [
        0 if ori == des else out[nw.cost].iloc[i]
        for i, (ori, des) in enumerate(zip(out.wkt_ori, out.wkt_des))
    ]

    # lag linjer mellom origin og destination
    if lines:
        origin = gpd.GeoSeries.from_wkt(out["wkt_ori"], crs=25833)
        destination = gpd.GeoSeries.from_wkt(out["wkt_des"], crs=25833)
        out["geometry"] = shortest_line(origin, destination)
        out = gpd.GeoDataFrame(out, geometry="geometry", crs=25833)

    out = out.drop(["wkt_ori", "wkt_des"], axis=1, errors="ignore")

    return out.reset_index(drop=True)
