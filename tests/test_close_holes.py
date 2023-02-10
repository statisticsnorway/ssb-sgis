#%%
import warnings
import geopandas as gpd
from time import perf_counter
import sys
sys.path.append("C:/Users/ort/git/ssb-gis-utils")

import gis_utils as gs


def test_close_holes(meters=1.1):
    print("meters:", meters)
    warnings.filterwarnings(action="ignore", category=UserWarning)
    warnings.filterwarnings(action="ignore", category=FutureWarning)

    r = gpd.read_parquet(r"C:/Users/ort/OneDrive - Statistisk sentralbyrå/data/vegdata/veger_landet_2022.parquet")

    p = gpd.read_parquet(r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\tilfeldige_adresser_1000.parquet")
    p = p.iloc[[0]] 
    nw = gs.DirectedNetwork(r)
    
    nw = nw.find_isolated()
    gs.qtm(nw.network.sjoin(gs.buff(p, 1000)), "isolated", cmap="bwr", title="before close_holes")
        
    _time = perf_counter()
    nw = nw.close_network_holes(meters, deadends_only=False)
    print("n", sum(nw.network.hole==1))
    print("time close_network_holes, all roads: ", perf_counter()-_time)

    _time = perf_counter()
    nw = nw.close_network_holes(meters, deadends_only=True)
    print("n", sum(nw.network.hole==1))
    print("time close_network_holes, deadends_only: ", perf_counter()-_time)

    nw = nw.find_isolated()
    gs.qtm(nw.network.sjoin(gs.buff(p, 1000)), "isolated", cmap="bwr", title=f"after close_holes({meters})")


def main():
    test_close_holes(1.1)
    test_close_holes(3)


if __name__ == "__main__":
    main()