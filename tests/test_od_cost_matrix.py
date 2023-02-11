#%%
import warnings
import geopandas as gpd
from time import perf_counter
import sys
sys.path.append("C:/Users/ort/git/ssb-gis-utils")

import gis_utils as gs
import cProfile


def test_od_cost_matrix():

    r = gpd.read_parquet(r"C:/Users/ort/OneDrive - Statistisk sentralbyrå/data/vegdata/veger_landet_2022.parquet")
#    r = gpd.read_parquet(r"C:/Users/ort/OneDrive - Statistisk sentralbyrå/data/vegdata/veger_oslo_og_naboer_2022.parquet")

    p = gpd.read_parquet(r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\tilfeldige_adresser_1000.parquet")
    p["idx"] = p.index
    p["idx2"] = p.index

    p1 = p.sample(1).idx.values[0]

    nw = gs.DirectedNetwork(r)

    nw = nw.make_directed_network_norway()
    
    nw = nw.remove_isolated()
 
    nw = gs.NetworkAnalysis(nw, cost="minutes")
    
    _time = perf_counter()
    od = nw.od_cost_matrix(p, p)
    od = nw.od_cost_matrix(p, p, id_col="idx")
    od = nw.od_cost_matrix(p, p, id_col=("idx", "idx2"),
        lines=True)

#    gs.qtm( 
 #       od.loc[od.origin == p1], 
  #      nw.cost, 
   #     scheme="quantiles"
    #    )
    print("time od_cost_matrix: ", perf_counter()-_time)

    print("percent missing", sum(od[nw.cost].isna()) / len(od) * 100)


def main():
    cProfile.run("test_od_cost_matrix()", sort="cumtime")


if __name__ == "__main__":
    main()