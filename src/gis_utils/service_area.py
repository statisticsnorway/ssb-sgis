import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from igraph import Graph

from .geopandas_utils import gdf_concat


def service_area(
    graph: Graph,
    startpoints: GeoDataFrame,
    weight: str,
    roads: GeoDataFrame,
    breaks: int | list[int] | tuple[int],
    id_col: str | None = None,
    dissolve: bool = True,
):
    if not id_col:
        id_col = "origin"

    if isinstance(breaks, (str, int, float)):
        breaks = [float(breaks)]

    # loop through every startpoint and every breaks
    results = []
    for i in startpoints["temp_idx"]:
        result = graph.distances(weights="weight", source=i)

        df = pd.DataFrame(
            data={"node_id": np.array(graph.vs["name"]), weight: result[0]}
        )

        for imp in breaks:
            indices = df.loc[df[weight] < imp]

            if not len(indices):
                results.append(
                    pd.DataFrame({id_col: [i], weight: [imp], "geometry": np.nan})
                )
                continue

            service_area = roads.loc[roads.target.isin(indices.node_id)]

            if dissolve:
                service_area = (
                    service_area[["geometry"]].dissolve().reset_index(drop=True)
                )

            service_area[id_col] = i
            service_area[weight] = imp
            results.append(service_area)

    return gdf_concat(results)
