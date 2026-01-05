from collections.abc import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame
from igraph import Graph
from pandas import DataFrame


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

    results = _get_od_df(graph, origins.index, destinations.index, weight)

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

    results["geom_ori"] = results["origin"].map(origins.geometry)
    results["geom_des"] = results["destination"].map(destinations.geometry)

    # straight lines between origin and destination
    if lines:
        results["geometry"] = shapely.shortest_line(
            results["geom_ori"], results["geom_des"]
        )
        results = gpd.GeoDataFrame(results, geometry="geometry", crs=25833)

    results.loc[
        shapely.to_wkb(results["geom_ori"]) == shapely.to_wkb(results["geom_des"]),
        weight,
    ] = 0

    return results.drop(["geom_ori", "geom_des"], axis=1, errors="ignore").reset_index(
        drop=True
    )


def _get_od_df(
    graph: Graph, origins: Iterable[str], destinations: Iterable[str], weight_col: str
) -> pd.DataFrame:
    distances: list[list[float]] = graph.distances(
        weights="weight",
        source=origins,
        target=destinations,
        algorithm="dijkstra",
    )

    costs = np.array(
        [distances[i][j] for j in range(len(destinations)) for i in range(len(origins))]
    )
    costs[(costs == np.inf) | (costs == -np.inf)] = np.nan
    ori_idx = np.array([x for _ in range(len(destinations)) for x in origins])
    des_idx = np.array([x for x in destinations for _ in range(len(origins))])

    return pd.DataFrame(
        data={"origin": ori_idx, "destination": des_idx, weight_col: costs}
    )
