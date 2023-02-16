#%%
import warnings
import numpy as np
import geopandas as gpd

import gis_utils as gs
from pathlib import Path

def test_od_cost_matrix():

    p = gpd.read_parquet(Path(__file__).parent / "testdata" / "random_points.parquet")
    p["idx"] = p.index
    p["idx2"] = p.index
    
    r = gpd.read_parquet(Path(__file__).parent / "testdata" / "roads_oslo_2022.parquet")
    
    nw = (
        gs.DirectedNetwork(r)
        .make_directed_network_norway()
        .remove_isolated()
    )

    nwa = gs.NetworkAnalysis(nw, cost="minutes")

    for search_factor in [0, 10, 25, 50]:
        nwa.search_factor = search_factor
        od = nwa.od_cost_matrix(p, p)
        print(
            f"percent missing, search_factor {nwa.search_factor}:", 
            np.mean(od[nwa.cost].isna()) * 100
            )

    for search_tolerance in [100, 250, 1000]:
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

    gs.qtm(
        od.loc[od.origin == p1], 
        nwa.cost, 
        scheme="quantiles"
        )


def main():
    test_od_cost_matrix()


if __name__ == "__main__":
    main()