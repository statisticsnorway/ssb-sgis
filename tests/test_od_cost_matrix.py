#%%
import warnings
import geopandas as gpd
from time import perf_counter
import sys
sys.path.append("C:/Users/ort/git/ssb-gis-utils")

import gis_utils as gs


def test_od_cost_matrix():

    r = gpd.read_parquet(r"C:/Users/ort/OneDrive - Statistisk sentralbyrå/data/vegdata/veger_landet_2022.parquet")

    p = gpd.read_parquet(r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\tilfeldige_adresser_1000.parquet")

    nw = gs.DirectedNetwork(r)
    
    nw = nw.remove_isolated()
        
    _time = perf_counter()
    od = nw.od_cost_matrix(p, p)
    print("time od_cost_matrix: ", perf_counter()-_time)

    print("percent missing", sum(od[nw.cost].isna()) / len(od) * 100)


def main():
    test_od_cost_matrix()


if __name__ == "__main__":
    main()