import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString
from networkz.hjelpefunksjoner import gdf_concat
from networkz.id_greier import bestem_ids, lag_midlr_id, map_ids
from networkz.lag_igraph import lag_graf


def service_area(
    G,
    startpunkter: gpd.GeoDataFrame,
    impedance,
    id_kolonne=None,  # hvis ikke id-kolonne oppgis, brukes startpunktenes geometri som id
    dissolve=True,
):
    startpunkter = startpunkter.to_crs(25833)

    startpunkter, id_kolonner = bestem_ids(id_kolonne, startpunkter)

    startpunkter["nz_idx"] = lag_midlr_id(G.noder, startpunkter)

    G2, startpunkter = lag_graf(G, G.kostnad, startpunkter)

    if isinstance(impedance, (str, int, float)):
        impedance = [float(impedance)]

    # loop for hvert startpunkt og hver kostnad
    alle_service_areas = []
    for i in startpunkter["nz_idx"]:
        for imp in impedance:
            if not i in G2.vs()["name"] and not i in G2.vs.indices:
                continue

            # beregn alle kostnader fra startpunktet
            resultat = G2.distances(weights="weight", source=i)

            # lag tabell av resultatene og fjern alt over ønsket kostnad
            df = pd.DataFrame(
                data={"name": np.array(G2.vs["name"]), G.kostnad: resultat[0]}
            )
            df = df[df[G.kostnad] < imp]

            if len(df) == 0:
                alle_service_areas.append(
                    gpd.GeoDataFrame(
                        pd.DataFrame(
                            {id_kolonne: [i], G.kostnad: imp, "geometry": LineString()}
                        ),
                        geometry="geometry",
                        crs=25833,
                    )
                )
                continue

            # velg ut vegene som er i dataframen vi nettopp lagde. Og dissolve til én rad.
            sa = G.nettverk.loc[G.nettverk.target.isin(df.name)]

            if dissolve:
                sa = sa[["geometry"]].dissolve().reset_index(drop=True)

            # lag kolonner for id, kostnad og evt. node-info
            sa[id_kolonne] = i
            sa[G.kostnad] = imp
            alle_service_areas.append(sa)

    alle_service_areas = gdf_concat(alle_service_areas)

    # få tilbake opprinnelige id-er
    alle_service_areas = map_ids(alle_service_areas, id_kolonner, startpunkter)

    return alle_service_areas.reset_index(drop=True)
