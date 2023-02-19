# %%
from pathlib import Path
import geopandas as gpd
import gis_utils as gs


def test_node_ids():
    p = gpd.read_parquet(Path(__file__).parent / "testdata" / "random_points.parquet")
    p["idx"] = p.index
    p["idx2"] = p.index

    r = gpd.read_parquet(Path(__file__).parent / "testdata" / "roads_oslo_2022.parquet")

    nw = gs.DirectedNetwork(r)
    rules = gs.NetworkAnalysisRules(cost="meters")
    nwa = gs.NetworkAnalysis(nw, rules=rules)
    
    nwa.od_cost_matrix(p.sample(5), p.sample(5), id_col="idx")

    nwa.network = nwa.network.close_network_holes(2)
    nwa.network = nwa.network.get_component_size()
    nwa.network = nwa.network.remove_isolated()
    nwa.network.gdf["kolonne"] = 1
    nwa.network.gdf = nwa.network.gdf.drop("kolonne", axis=1)

    nwa.network.gdf = nwa.network.gdf.sjoin(
        gs.buff(p[["geometry"]].sample(1), 2500)
    ).drop("index_right", axis=1, errors="ignore")

    p = (
        p.sjoin(gs.buff(nwa.network.gdf[["geometry"]], 2500))
        .drop("index_right", axis=1, errors="ignore")
        .drop_duplicates("idx")
    )

    nwa.od_cost_matrix(p.sample(5), p.sample(5), id_col="idx")
    nwa.od_cost_matrix(p.sample(5), p.sample(5), id_col="idx")
    nwa.od_cost_matrix(p.sample(5), p.sample(5), id_col="idx")
    nwa.od_cost_matrix(p.sample(5), p.sample(5), id_col="idx")
    nwa.od_cost_matrix(p.sample(5), p.sample(5), id_col="idx")


def main():
    import cProfile

    cProfile.run("test_node_ids()", sort="cumtime")


if __name__ == "__main__":
    main()

# %%
