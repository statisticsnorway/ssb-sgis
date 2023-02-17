# %%
import sys
import warnings
from time import perf_counter

import geopandas as gpd


sys.path.append("C:/Users/ort/git/ssb-gis-utils")

import cProfile

import gis_utils as gs


def count_function_calls():
    r = gpd.read_parquet(
        r"C:/Users/ort/OneDrive - Statistisk sentralbyrå/data/vegdata/veger_oslo_og_naboer_2022.parquet"
    )
    p = gpd.read_parquet(
        r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\tilfeldige_adresser_1000.parquet"
    )

    nw = gs.DirectedNetwork(r)

    nw = nw.remove_isolated()
    nw.make_directed_network_norway()

    nw = (
        nw.get_component_size()
        .get_largest_component()
        .close_network_holes(1.1)
        .remove_isolated()
        # .cut_lines(50)
    )

    nwa = gs.NetworkAnalysis(nw, cost="minutes")
    nwa.network = nwa.network.get_component_size()
    nwa.network = nwa.network.remove_isolated()

    for _ in range(10):
        nwa.od_cost_matrix(p.sample(1), p.sample(1))


def main():
    count_function_calls()
    cProfile.run("count_function_calls()", sort="cumtime")


if __name__ == "__main__":
    main()
