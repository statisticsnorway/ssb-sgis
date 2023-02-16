#%%
import geopandas as gpd
from pathlib import Path

import gis_utils as gs

def test_node_ids():

    p = gpd.read_parquet(Path(__file__).parent / "testdata" / "random_points.parquet")
    p["idx"] = p.index
    p["idx2"] = p.index
    
    r = gpd.read_parquet(Path(__file__).parent / "testdata" / "roads_oslo_2022.parquet")

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
    import cProfile
    cProfile.run("test_node_ids()", sort="cumtime")


if __name__ == "__main__":
    main()

#%%