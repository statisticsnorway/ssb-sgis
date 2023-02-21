import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from igraph import Graph

from .geopandas_utils import gdf_concat


def service_area(
    graph: Graph,
    startpoints: GeoDataFrame,
    cost: str,
    roads: GeoDataFrame,
    impedance: int | list[int] | tuple[int],
    id_col: str | None = None,
    dissolve: bool = True,
):
    if not id_col:
        id_col = "origin"

    if isinstance(impedance, (str, int, float)):
        impedance = [float(impedance)]

    # loop through every startpoint and every impedance
    results = []
    for i in startpoints["temp_idx"]:
        result = graph.distances(weights="weight", source=i)

        df = pd.DataFrame(data={"node_id": np.array(graph.vs["name"]), cost: result[0]})

        for imp in impedance:
            indices = df.loc[df[cost] < imp]

            if not len(indices):
                results.append(
                    pd.DataFrame({id_col: [i], cost: [imp], "geometry": np.nan})
                )
                continue

            service_area = roads.loc[roads.target.isin(indices.node_id)]

            if dissolve:
                service_area = (
                    service_area[["geometry"]].dissolve().reset_index(drop=True)
                )

            service_area[id_col] = i
            service_area[cost] = imp
            results.append(service_area)

    return gdf_concat(results)
