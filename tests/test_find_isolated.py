#%%
import warnings
import geopandas as gpd
import numpy as np
from time import perf_counter
import sys
sys.path.append("C:/Users/ort/git/ssb-gis-utils")

import gis_utils as gs


def test_find_isolated():
    
    p = gpd.read_parquet(r"C:/Users/ort/OneDrive - Statistisk sentralbyrå/data/tilfeldige_adresser_1000.parquet")
    p = p.iloc[[0]]

    r = gpd.read_parquet(r"C:/Users/ort/OneDrive - Statistisk sentralbyrå/data/vegdata/veger_oslo_og_naboer_2022.parquet")
    nw = gs.DirectedNetwork(r, cost="meters")

    _time = perf_counter()
    nw = nw.find_isolated()
    print("n", sum(nw.network.isolated==1))
    print("time find_isolated: ", perf_counter()-_time)

    gs.qtm(nw.network.sjoin(gs.buff(p, 1000)), "isolated", cmap="bwr")

    r = gpd.read_parquet(r"C:/Users/ort/OneDrive - Statistisk sentralbyrå/data/vegdata/veger_landet_2022.parquet")
    nw = gs.DirectedNetwork(r, cost="meters")

    _time = perf_counter()
    nw = nw.find_isolated()
    print("n", sum(nw.network.isolated==1))
    print("time find_isolated: ", perf_counter()-_time)

    gs.qtm(nw.network.sjoin(gs.buff(p, 1000)), "isolated", cmap="bwr")


def main():
    test_find_isolated()


if __name__ == "__main__":
    main()

#%%