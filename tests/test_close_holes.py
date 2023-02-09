#%%
import warnings
import geopandas as gpd
from time import perf_counter
import sys
sys.path.append("C:/Users/ort/git/ssb-geopandas-util")

import geopandas_util as gs


def test_close_holes(meters=1.1):
    print("meters:", meters)
    warnings.filterwarnings(action="ignore", category=UserWarning)
    warnings.filterwarnings(action="ignore", category=FutureWarning)

    r = gpd.read_parquet(r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\vegdata\veger_oslo_og_naboer_2021.parquet")
    r = r.loc[r.SPERRING == -1]
    r = r.loc[r.FYLKE_ID.astype(int) == 3]

    p = gpd.read_parquet(r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\tilfeldige_adresser_1000.parquet")
    p = p.iloc[[0]] 
    nw = gs.DirectedNetwork(r)
    
    _time = perf_counter()
    nw = nw.close_network_holes(meters, deadends_only=False)
    print("n", sum(nw.network.hole==1))
    print("time close_network_holes, all roads: ", perf_counter()-_time)

    _time = perf_counter()
    nw = nw.close_network_holes(meters, deadends_only=True)
    print("n", sum(nw.network.hole==1))
    print("time close_network_holes, deadends_only: ", perf_counter()-_time)

    nw = nw.finn_isolerte_nettverk(lengde=10_000, ruteloop_m=2250)
    gs.qtm(nw.network.sjoin(gs.buff(p, 1000)), "isolated", cmap="bwr")


def main():
    test_close_holes(1.1)
    test_close_holes(2)
#    test_close_holes(3)
 #   test_close_holes(4)


if __name__ == "__main__":
    main()