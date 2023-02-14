#%%
import warnings
import numpy as np
import geopandas as gpd
from geopandas import GeoDataFrame
from time import perf_counter
import sys
sys.path.append("C:/Users/ort/git/ssb-gis-utils")

import gis_utils as gs
import cProfile


from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

def chop_cmap_frac(cmap: LinearSegmentedColormap, frac: float) -> LinearSegmentedColormap:
    """Chops off the beginning `frac` fraction of a colormap."""
    cmap = plt.get_cmap(cmap)
    cmap_as_array = cmap(np.arange(256))
    cmap_as_array = cmap_as_array[int(frac * len(cmap_as_array)):]
    return LinearSegmentedColormap.from_list(cmap.name + f"_frac{frac}", cmap_as_array)
    

def test_shortest_path(nwa: gs.NetworkAnalysis):

    cmap = chop_cmap_frac("RdPu", 0.2)

    p = gpd.read_parquet(r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\tilfeldige_adresser_1000.parquet")
    p["idx"] = p.index
    p["idx2"] = p.index

    sp = nwa.shortest_path(p.iloc[[0]], p.sample(250), id_col="idx", summarise=True)
    gs.qtm(sp, "n", scheme="naturalbreaks", k=9, cmap=cmap, title="One-to-many: n times each road was used.")

    sp = nwa.shortest_path(
        p.sample(50), p.sample(50), 
        summarise=True
    )

    gs.qtm(sp, "n", scheme="naturalbreaks", k=9, cmap=cmap, title="Many-to-many: n times each road was used.")

    sp = nwa.shortest_path(
        p.sample(25), p.sample(25), 
    )


def main():
    #    r = gpd.read_parquet(r"C:/Users/ort/OneDrive - Statistisk sentralbyrå/data/vegdata/veger_landet_2022.parquet")
    r = gpd.read_parquet(r"C:/Users/ort/OneDrive - Statistisk sentralbyrå/data/vegdata/veger_oslo_og_naboer_2022.parquet")

    nw = (
        gs.DirectedNetwork(r)
        .make_directed_network_norway()
        .remove_isolated()
    )

    nwa = gs.NetworkAnalysis(nw, cost="meters")

    test_shortest_path(nwa)
#    cProfile.run(f"test_shortest_path()", sort="cumtime")


if __name__ == "__main__":
    main()
# %%
