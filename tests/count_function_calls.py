# %%
import cProfile
from pathlib import Path

import geopandas as gpd

import gis_utils as gs


def count_function_calls():
    p = gpd.read_parquet(Path(__file__).parent / "testdata" / "random_points.parquet")
    p = p.iloc[:50]
    p["idx"] = p.index
    p["idx2"] = p.index

    r = gpd.read_parquet(Path(__file__).parent / "testdata" / "roads_oslo_2022.parquet")

    nw = gs.DirectedNetwork(r)

    nw = nw.remove_isolated()
    nw.make_directed_network_norway()

    nw = (
        nw.get_component_size()
        .get_largest_component()
        .close_network_holes(1.1)
        .remove_isolated()
        .cut_lines(250)
    )

    rules = gs.NetworkAnalysisRules(weight="minutes")
    nwa = gs.NetworkAnalysis(nw, rules=rules)
    nwa.network = nwa.network.get_component_size()
    nwa.network = nwa.network.remove_isolated()

    for _ in range(10):
        nwa.od_cost_matrix(p.sample(1), p.sample(1))


def main():
    count_function_calls()
    cProfile.run("count_function_calls()", sort="cumtime")


if __name__ == "__main__":
    main()
