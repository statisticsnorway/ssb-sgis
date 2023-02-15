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
    nw = gs.DirectedNetwork(r)

    nw = nw.get_component_size()
    gs.qtm(nw.gdf.loc[nw.gdf.component_size != max(nw.gdf.component_size)].sjoin(gs.buff(p, 1000)), "component_size", scheme="quantiles", k=7)

    _time = perf_counter()
    nw = nw.get_largest_component()
    print("n", sum(nw.gdf.connected==0))
    print("time get_largest_component: ", perf_counter()-_time)

    gs.qtm(nw.gdf.sjoin(gs.buff(p, 1000)), "connected", cmap="bwr", scheme="equalinterval")

    r = gpd.read_parquet(r"C:/Users/ort/OneDrive - Statistisk sentralbyrå/data/vegdata/veger_landet_2022.parquet")
    nw = gs.DirectedNetwork(r)

    _time = perf_counter()
    nw = nw.get_largest_component()
    print("n", sum(nw.gdf.connected==0))
    print("time get_largest_component: ", perf_counter()-_time)

    gs.qtm(nw.gdf.sjoin(gs.buff(p, 1000)), "connected", cmap="bwr", scheme="equalinterval")


def main():
    test_find_isolated()


if __name__ == "__main__":
    main()

#%%