import geopandas as gpd
import pandas as pd
import numpy as np
from .hjelpefunksjoner import gdf_concat
from .id_greier import bestem_ids, lag_midlr_id, map_ids
from .lag_igraph import lag_graf


def shortest_path(
    G,
    startpunkter: gpd.GeoDataFrame,
    sluttpunkter: gpd.GeoDataFrame,
    id_kolonne=None,
    cutoff: int = None,
    destination_count: int = None,
    radvis=False,
    tell_opp=False,
):
    import warnings

    warnings.filterwarnings(
        "ignore", category=RuntimeWarning
    )  # ignorer tåpelig advarsel

    startpunkter = startpunkter.to_crs(25833)
    sluttpunkter = sluttpunkter.to_crs(25833)
    G.nettverk = G.nettverk.to_crs(25833)

    startpunkter, sluttpunkter, id_kolonner = bestem_ids(
        id_kolonne, startpunkter, sluttpunkter
    )

    startpunkter["nz_idx"], sluttpunkter["nz_idx"] = lag_midlr_id(
        G.noder, startpunkter, sluttpunkter
    )

    G2, startpunkter, sluttpunkter = lag_graf(G, G.kostnad, startpunkter, sluttpunkter)

    # funksjonen get_shortest_paths() må loopes for hver fra-til-kombinasjon
    def kjor_korteste(fra_id, til_id, G, G2, tell_opp):
        res = G2.get_shortest_paths(weights="weight", v=fra_id, to=til_id)

        if len(res[0]) == 0:
            return []

        path = G2.vs[res[0]]["name"]

        if tell_opp:
            return [
                pd.DataFrame(
                    {
                        "source_target": (
                            str(source) + "_" + str(target)
                            for source, target in zip(path[:-1], path[1:])
                        )
                    }
                )
            ]

        linje = G.nettverk.loc[
            (G.nettverk.source.isin(path)) & (G.nettverk.target.isin(path)),
            ["geometry"],
        ]

        if not len(linje):
            return []

        linje = linje.dissolve()

        linje["fra"] = fra_id
        linje["til"] = til_id

        # for å få kostnaden også
        kost = G2.distances(weights="weight", source=fra_id, target=til_id)
        linje[G.kostnad] = kost[0][0]

        return [linje]

    linjer = []
    if radvis:
        for fra_id, til_id in zip(startpunkter["nz_idx"], sluttpunkter["nz_idx"]):
            linjer = linjer + kjor_korteste(fra_id, til_id, G, G2, tell_opp)
    else:
        for fra_id in startpunkter["nz_idx"]:
            for til_id in sluttpunkter["nz_idx"]:
                linjer = linjer + kjor_korteste(fra_id, til_id, G, G2, tell_opp)

    if tell_opp:
        lenker = pd.concat(linjer, ignore_index=True)
        lenker["antall"] = 1
        lenker = lenker.groupby("source_target").count()["antall"].reset_index()

        veger = G.nettverk[["geometry", "source", "target"]]
        veger["source_target"] = veger.source + "_" + veger.target

        return veger.merge(lenker, on="source_target", how="inner").drop(
            "source_target", axis=1
        )

    try:
        linjer = gdf_concat(linjer)
    except Exception:
        raise ValueError(
            f"Ingen ruter ble funnet. Mulig prøv {'directed=False, ' if G.directed==True else ''}{'fjern_isolerte=True, ' if G.fjern_isolerte==False else ''}{'høyere dist_faktor, ' if G.dist_faktor<15 else ''}"
        )

    linjer = (
        linjer.replace([np.inf, -np.inf], np.nan)
        .loc[(linjer[G.kostnad] > 0) | (linjer[G.kostnad].isna())]
        .reset_index(drop=True)
        .merge(
            startpunkter[["dist_node_start", "nz_idx"]],
            left_on="fra",
            right_on="nz_idx",
            how="left",
        )
        .drop("nz_idx", axis=1)
        .merge(
            sluttpunkter[["dist_node_slutt", "nz_idx"]],
            left_on="til",
            right_on="nz_idx",
            how="left",
        )
        .drop("nz_idx", axis=1)
    )

    linjer = map_ids(linjer, id_kolonner, startpunkter, sluttpunkter)

    if cutoff:
        linjer = linjer[linjer[G.kostnad] < cutoff]

    if destination_count:
        linjer = linjer.loc[linjer.groupby("fra")[G.kostnad].idxmin()].reset_index(
            drop=True
        )

    linjer = linjer[["fra", "til", G.kostnad, "geometry"]]

    return linjer.reset_index(drop=True)
