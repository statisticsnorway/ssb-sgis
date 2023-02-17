import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from .geopandas_utils import gdf_concat


def service_area(
    nw,
    startpoints: gpd.GeoDataFrame,
    impedance,
    id_col=None,  # hvis ikke id-kolonne oppgis, brukes startpunktenes geometri som id
    dissolve=True,
):
    if not id_col:
        id_col = "origin"

    if isinstance(impedance, (str, int, float)):
        impedance = [float(impedance)]

    # loop for hvert startpunkt og hver cost
    service_areas = []
    for i in startpoints["temp_idx"]:
        for imp in impedance:
            if not i in nw.graph.vs()["name"] and not i in nw.graph.vs.indices:
                continue

            # beregn alle coster fra startpunktet
            resultat = nw.graph.distances(weights="weight", source=i)

            # lag tabell av resultatene og fjern alt over ønsket cost
            df = pd.DataFrame(
                data={"name": np.array(nw.graph.vs["name"]), nw.cost: resultat[0]}
            )
            df = df[df[nw.cost] < imp]

            if len(df) == 0:
                service_areas.append(
                    pd.DataFrame({id_col: [i], nw.cost: [imp], "geometry": np.nan})
                )
                continue

            # velg ut vegene som er i dataframen vi nettopp lagde.
            # Og dissolve til én rad.
            sa = nw.network.gdf.loc[nw.network.gdf.target.isin(df.name)]

            if dissolve:
                sa = sa[["geometry"]].dissolve().reset_index(drop=True)

            # lag kolonner for id, cost og evt. node-info
            sa[id_col] = i
            sa[nw.cost] = imp
            service_areas.append(sa)

    return gdf_concat(service_areas)
