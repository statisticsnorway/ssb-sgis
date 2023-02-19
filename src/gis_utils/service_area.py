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

    # loop for hvert startpunkt og hver cost
    service_areas = []
    for i in startpoints["temp_idx"]:
        for imp in impedance:
            result = graph.distances(weights="weight", source=i)

            # lag tabell av resultene og fjern alt over ønsket cost
            df = pd.DataFrame(
                data={"name": np.array(graph.vs["name"]), cost: result[0]}
            )

            df = df[df[cost] < imp]

            if len(df) == 0:
                service_areas.append(
                    pd.DataFrame({id_col: [i], cost: [imp], "geometry": np.nan})
                )
                continue

            # velg ut vegene som er i dataframen vi nettopp lagde.
            # Og dissolve til én rad.
            sa = roads.loc[roads.target.isin(df.name)]

            if dissolve:
                sa = sa[["geometry"]].dissolve().reset_index(drop=True)

            # lag kolonner for id, cost og evt. node-info
            sa[id_col] = i
            sa[cost] = imp
            service_areas.append(sa)

    return gdf_concat(service_areas)
