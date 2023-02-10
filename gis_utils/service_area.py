import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString

from .geopandas_utils import gdf_concat

def service_area(
    nx,
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
            if not i in nx.graph.vs()["name"] and not i in nx.graph.vs.indices:
                continue

            # beregn alle coster fra startpunktet
            resultat = nx.graph.distances(weights="weight", source=i)

            # lag tabell av resultatene og fjern alt over ønsket cost
            df = pd.DataFrame(
                data={"name": np.array(nx.graph.vs["name"]), nx.cost: resultat[0]}
            )
            df = df[df[nx.cost] < imp]

            if len(df) == 0:
                service_areas.append(
                        pd.DataFrame(
                            {id_col: [i], nx.cost: [imp], "geometry": np.nan}
                        )
                    )
                continue
                service_areas.append(
                    gpd.GeoDataFrame(
                        pd.DataFrame(
                            {id_col: [i], nx.cost: imp, "geometry": LineString()}
                        ),
                        geometry="geometry",
                        crs=25833,
                    )
                )
                continue

            # velg ut vegene som er i dataframen vi nettopp lagde. Og dissolve til én rad.
            sa = nx.network.loc[nx.network.target.isin(df.name)]

            if dissolve:
                sa = sa[["geometry"]].dissolve().reset_index(drop=True)

            # lag kolonner for id, cost og evt. node-info
            sa[id_col] = i
            sa[nx.cost] = imp
            service_areas.append(sa)

    return gdf_concat(service_areas)
