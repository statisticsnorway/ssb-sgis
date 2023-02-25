import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from igraph import Graph

from .geopandas_utils import gdf_concat


def _service_area(
    graph: Graph,
    origins: GeoDataFrame,
    weight: str,
    lines: GeoDataFrame,
    breaks: int | list[int] | tuple[int],
    dissolve: bool = True,
) -> GeoDataFrame:
    if isinstance(breaks, (str, int, float)):
        breaks = [float(breaks)]

    # loop through every origin and every breaks
    results = []
    for i in origins["temp_idx"]:
        result = graph.distances(weights="weight", source=i)

        df = pd.DataFrame(
            data={"node_id": np.array(graph.vs["name"]), weight: result[0]}
        )

        for imp in breaks:
            indices = df.loc[df[weight] < imp]

            if not len(indices):
                results.append(
                    pd.DataFrame({"origin": [i], weight: [imp], "geometry": np.nan})
                )
                continue

            service_area = lines.loc[lines.target.isin(indices.node_id)]

            if dissolve:
                service_area = (
                    service_area[["geometry"]].dissolve().reset_index(drop=True)
                )

            service_area["origin"] = i
            service_area[weight] = imp
            results.append(service_area)

    return gdf_concat(results)
