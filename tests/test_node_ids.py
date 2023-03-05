# %%
import sys
from pathlib import Path

import geopandas as gpd


src = str(Path(__file__).parent).strip("tests") + "src"

sys.path.insert(0, src)

import gis_utils as gs


def test_node_ids():
    p = gpd.read_parquet(gs.pointpath)
    p = gs.clean_clip(p, p.geometry.iloc[0].buffer(500))
    p["idx"] = p.index
    p["idx2"] = p.index

    r = gpd.read_parquet(gs.roadpath)
    r = gs.clean_clip(r, p.geometry.iloc[0].buffer(600))

    r, nodes = gs.make_node_ids(r)
    print(nodes)
    r, nodes = gs.make_node_ids(r, wkt=False)
    print(nodes)

    nw = gs.DirectedNetwork(r)
    rules = gs.NetworkAnalysisRules(weight="meters")
    nwa = gs.NetworkAnalysis(nw, rules=rules)

    nwa.od_cost_matrix(p.sample(5), p.sample(5), id_col="idx")

    nwa.network = nwa.network.close_network_holes(2, fillna=0)
    nwa.network = nwa.network.get_component_size()
    nwa.network = nwa.network.remove_isolated()
    nwa.network.gdf["col"] = 1
    nwa.network.gdf = nwa.network.gdf.drop("col", axis=1)

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
    """Check how many times make_node_ids is run."""
    import cProfile

    test_node_ids()
    quit()
    cProfile.run("test_node_ids()", sort="cumtime")


if __name__ == "__main__":
    main()

# %%
