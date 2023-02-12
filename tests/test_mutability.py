#%%
import warnings
import geopandas as gpd
from time import perf_counter
import sys
sys.path.append("C:/Users/ort/git/ssb-gis-utils")

import gis_utils as gs
import cProfile


def test_mutability():

    r = gpd.read_parquet(r"C:/Users/ort/OneDrive - Statistisk sentralbyrå/data/vegdata/veger_oslo_og_naboer_2022.parquet")

    nw = gs.DirectedNetwork(r)
    
    for i in nw:
        print(i)

    print(len(nw.gdf))
    nw.make_directed_network_norway()
    print(len(nw.gdf))
    nw = nw.remove_isolated()
    print(len(nw.gdf))

    nw = (nw
        .make_directed_network_norway()
        .get_component_size()
        .get_largest_component()
        .close_network_holes(1.1)
        .remove_isolated()
       # .cut_lines(50)
    )

def main():
    cProfile.run("test_mutability()", sort="cumtime")
    

if __name__ == "__main__":
    main()