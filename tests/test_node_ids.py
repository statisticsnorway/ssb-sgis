#%%
import warnings
import geopandas as gpd
import numpy as np
from time import perf_counter
import sys
import cProfile
sys.path.append("C:/Users/ort/git/ssb-gis-utils")

import gis_utils as gs

def test_node_ids():

    r = gpd.read_parquet(r"C:/Users/ort/OneDrive - Statistisk sentralbyrå/data/vegdata/veger_oslo_og_naboer_2022.parquet")
    p = gpd.read_parquet(r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\tilfeldige_adresser_1000.parquet")
    p["idx"] = p.index

    nw = gs.DirectedNetwork(r)
    nw = gs.NetworkAnalysis(nw, cost="meters")

    nw.od_cost_matrix(p.sample(5), p.sample(5), id_col="idx")
    
    nw.network = nw.network.close_network_holes(2)
    nw.network = nw.network.get_component_size()
    nw.network = nw.network.remove_isolated()
    nw.network.gdf["kolonne"] = 1
    nw.network.gdf = nw.network.gdf.drop("kolonne", axis=1)

    nw.network.gdf = nw.network.gdf.sjoin(gs.buff(p[["geometry"]].sample(1), 2500)).drop("index_right", axis=1, errors="ignore")

    p = p.sjoin(gs.buff(nw.network.gdf[["geometry"]], 2500)).drop("index_right", axis=1, errors="ignore").drop_duplicates("idx")

    nw.od_cost_matrix(p.sample(5), p.sample(5), id_col="idx")
    nw.od_cost_matrix(p.sample(5), p.sample(5), id_col="idx")
    nw.od_cost_matrix(p.sample(5), p.sample(5), id_col="idx")
    nw.od_cost_matrix(p.sample(5), p.sample(5), id_col="idx")
    nw.od_cost_matrix(p.sample(5), p.sample(5), id_col="idx")


def main():
    cProfile.run("test_node_ids()", sort="cumtime")


if __name__ == "__main__":
    main()

#%%