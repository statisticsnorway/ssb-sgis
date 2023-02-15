#%%
import warnings
import numpy as np
import geopandas as gpd
from time import perf_counter
import sys
sys.path.append("C:/Users/ort/git/ssb-gis-utils")

import gis_utils as gs
import cProfile


def test_service_area(nwa: gs.NetworkAnalysis):

    p = gpd.read_parquet(r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\tilfeldige_adresser_1000.parquet")
    p["idx"] = p.index
    p["idx2"] = p.index

    sa = nwa.service_area(p.sample(25), impedance=5, dissolve=False)

    print(len(sa))

    sa = sa.drop_duplicates(["source", "target"])

    print(len(sa))
    gs.qtm(sa)

    # many impedances
    sa = nwa.service_area(p.iloc[[0]], impedance=np.arange(1, 11), id_col="idx")
    sa = sa.sort_values("minutes", ascending=False)
    gs.qtm(sa, "minutes", k=10)


def main():
#    r = gpd.read_parquet(r"C:/Users/ort/OneDrive - Statistisk sentralbyrå/data/vegdata/veger_landet_2022.parquet")
    r = gpd.read_parquet(r"C:/Users/ort/OneDrive - Statistisk sentralbyrå/data/vegdata/veger_oslo_og_naboer_2022.parquet")
    
    nw = (
        gs.DirectedNetwork(r)
        .make_directed_network_norway()
        .remove_isolated()
    )

    nwa = gs.NetworkAnalysis(nw, cost="minutes")

    test_service_area(nwa)
#    cProfile.run(f"test_service_area({r})", sort="cumtime")


if __name__ == "__main__":
    main()