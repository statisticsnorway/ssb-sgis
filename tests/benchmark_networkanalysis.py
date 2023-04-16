"""Benchmark network analysis"""

# %%
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


src = str(Path(__file__).parent).strip("tests") + "src"

sys.path.insert(0, src)

import sgis as sg


def not_test_od_cost_matrix(nwa, p):
    od = nwa.od_cost_matrix(p, p)


def not_test_get_route_frequency(nwa, p):
    od_pairs = pd.MultiIndex.from_product([p.index, p.index])
    weight_df_all_10 = pd.DataFrame(index=od_pairs)
    weight_df_all_10["weight"] = 10

    frequencies = nwa.get_route_frequencies(p, p, weight_df=weight_df_all_10)


def not_test_get_route(nwa, p):
    routes = nwa.get_route(p, p)

    # adding this for comparison purposes
    od_pairs = pd.MultiIndex.from_product([p.index, p.index])
    weight_df_all_10 = pd.DataFrame(index=od_pairs)
    weight_df_all_10["weight"] = 10


def not_test_service_area(nwa, p):
    service_areas = nwa.service_area(p, breaks=20, dissolve=True)


def not_test_network_analysis(points_oslo, roads_oslo):
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    pd.options.mode.chained_assignment = None

    split_lines = False

    buffdist = 2000

    p = points_oslo
    p = sg.clean_clip(p, p.geometry.iloc[0].buffer(buffdist))
    p["idx"] = p.index
    p["idx2"] = p.index

    print("number of points", len(p))

    r = roads_oslo
    r = sg.clean_clip(r, p.geometry.loc[0].buffer(buffdist * 1.1))

    ### MAKE THE ANALYSIS CLASS
    nw = sg.DirectedNetwork(r).make_directed_network_norway().remove_isolated()
    rules = sg.NetworkAnalysisRules(
        weight="minutes",
        split_lines=split_lines,
    )
    nwa = sg.NetworkAnalysis(nw, rules=rules)
    print(nwa)

    not_test_od_cost_matrix(nwa, p)
    print(nwa.log.groupby("method")["minutes_elapsed"].mean())

    not_test_get_route_frequency(nwa, p)
    print(nwa.log.groupby("method")["minutes_elapsed"].mean())

    not_test_get_route(nwa, p)
    print(nwa.log.groupby("method")["minutes_elapsed"].mean())

    print("_graph_updated_count", nwa._graph_updated_count)

    not_test_service_area(nwa, p)
    print(nwa.log.groupby("method")["minutes_elapsed"].mean())

    print("_graph_updated_count", nwa._graph_updated_count)


def main():
    roads_oslo = sg.read_parquet_url(
        "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet"
    )
    points_oslo = sg.read_parquet_url(
        "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet"
    )

    for _ in range(1):
        not_test_network_analysis(points_oslo, roads_oslo)


if __name__ == "__main__":
    import cProfile

    cProfile.run("main()", sort="cumtime")
