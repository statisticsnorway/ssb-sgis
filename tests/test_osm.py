#%%
import warnings
import numpy as np
import geopandas as gpd
from time import perf_counter
import sys
sys.path.append("C:/Users/ort/git/ssb-gis-utils")

import gis_utils as gs
import cProfile


def test_osm():

    r = gpd.read_file(r"C:\Users\ort\Downloads\norway-latest-free.shp\gis_osm_roads_free_1.shp")
    r = r.to_crs(25833)
    print(r.columns)
    print(r)

    p = gpd.read_parquet(r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\tilfeldige_adresser_1000.parquet")
    p["idx"] = p.index
    p["idx2"] = p.index

    nw = (
        gs.Network(r)
        .close_network_holes(1.1)
        .remove_isolated()
    )
    nwa = gs.NetworkAnalysis(
        nw, cost="meters", search_factor=100, search_tolerance=1000,
    )
    od = nwa.od_cost_matrix(p, p)
    print(
        f"percent missing undirected, search_factor {nwa.search_factor}:", 
        np.mean(od[nwa.cost].isna()) * 100
        )
    
    nw = (
        gs.DirectedNetwork(r)
        .make_directed_network_osm()
        .remove_isolated()
    )
 
    nwa = gs.NetworkAnalysis(nw, cost="minutes")
    
    for search_factor in [0, 10, 25, 50, 100]:
        nwa.search_factor = search_factor
        od = nwa.od_cost_matrix(p, p)
        print(
            f"percent missing, search_factor {nwa.search_factor}:", 
            np.mean(od[nwa.cost].isna()) * 100
            )

    for search_tolerance in [100, 250, 1000, 10_000]:
        nwa.search_tolerance = search_tolerance
        od = nwa.od_cost_matrix(p, p)
        print(
            f"percent missing, search_factor 100, search_tolerance {nwa.search_tolerance}:", 
            np.mean(od[nwa.cost].isna()) * 100
            )

    od = nwa.od_cost_matrix(p, p, id_col=("idx", "idx2"), lines=True)

    print(nwa.startpoints.points.n_missing.value_counts())

    p1 = nwa.startpoints.points
    p1 = p1.loc[[p1.n_missing.idxmin()]].sample(1).idx.values[0]

    # ved Lindeberg skole, på hovedøya
    gs.qtm(
        od.loc[od.origin == p1], 
        nwa.cost, 
        scheme="quantiles"
        )

    m = nwa.startpoints.points.n_missing.mean()
    display(nwa.startpoints.points.query("n_missing > @m").explore("n_missing", scheme="quantiles"))


def main():
    cProfile.run("test_osm()", sort="cumtime")


if __name__ == "__main__":
    main()