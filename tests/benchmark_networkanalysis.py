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
    for search_factor in [0, 50]:
        nwa.rules.search_factor = search_factor
        od = nwa.od_cost_matrix(p, p)

    nwa.rules.search_factor = 10

    for search_tolerance in [100, 1000]:
        nwa.rules.search_tolerance = search_tolerance
        od = nwa.od_cost_matrix(p, p)

    print(
        nwa.log[["search_tolerance", "search_factor", "percent_missing", "cost_mean"]]
    )

    od = nwa.od_cost_matrix(p, p, lines=True)

    p = p.sort_index(ascending=True)
    p_rev = p.sort_index(ascending=False)
    od = nwa.od_cost_matrix(p, p_rev, rowwise=True)


def not_test_get_route_frequency(nwa, p):
    frequencies = nwa.get_route_frequencies(p, p)

    # testing weight_df with and without reset_index
    # first, weight_df will have 1 column and multiindex, second, weight_df will have
    # three columns (origin, destination and weight. Column names doesn't matter)
    for truefalse in [0, 1]:
        od_pairs = pd.MultiIndex.from_product([p.index, p.index])
        weight_df_all_10 = pd.DataFrame(index=od_pairs)
        weight_df_all_10["weight"] = 10

        if truefalse:
            weight_df_all_10 = weight_df_all_10.reset_index()

        frequencies = nwa.get_route_frequencies(p, p, weight_df=weight_df_all_10)

    od_pairs = pd.MultiIndex.from_product([p.index, p.index])
    weight_df_one_pair_10 = pd.DataFrame(index=od_pairs)
    weight_df_one_pair_10["weight"] = 1
    weight_df_one_pair_10.loc[(349, 97), "weight"] = 100

    frequencies = nwa.get_route_frequencies(p, p, weight_df=weight_df_one_pair_10)

    frequencies = nwa.get_route_frequencies(p.loc[[349]], p)


def not_test_get_route(nwa, p):
    routes = nwa.get_route(p, p)

    routes = nwa.get_route(p.loc[[349]], p)

    nwa.rules.search_factor = 0
    nwa.rules.split_lines = False

    routes = nwa.get_route(p.loc[[349]], p.loc[[440]])

    nwa.rules.split_lines = True
    routes = nwa.get_route(p.loc[[349]], p.loc[[440]])
    routes = nwa.get_route(p.loc[[349]], p.loc[[440]])

    nwa.rules.split_lines = False
    routes = nwa.get_route(p.loc[[349]], p)

    nwa.rules.split_lines = True
    routes = nwa.get_route(p.loc[[349]], p)

    routes = nwa.get_route(p.loc[[349]], p)

    p = p.sort_index(ascending=True)
    p_rev = p.sort_index(ascending=False)
    routes = nwa.get_route(p, p_rev, rowwise=True)


def not_test_service_area(nwa, p):
    sa = nwa.service_area(p, breaks=5, dissolve=False)

    sa = sa.drop_duplicates(["source", "target"])

    sa = nwa.service_area(p.loc[[349]], breaks=np.arange(1, 11))

    sa = sa.sort_values("minutes", ascending=False)


def not_test_get_k_routes(nwa, p):
    for x in [0, 100]:
        routes = nwa.get_k_routes(
            p.loc[[349]], p.loc[[440]], k=5, drop_middle_percent=x
        )

    n = 0
    for x in [-1, 101]:
        try:
            routes = nwa.get_k_routes(
                p.loc[[349]],
                p.loc[[440]],
                k=5,
                drop_middle_percent=x,
            )
        except ValueError:
            n += 1
            print("drop_middle_percent works as expected", x)

    assert n == 2

    routes = nwa.get_k_routes(p.loc[[349]], p.loc[[440]], k=5, drop_middle_percent=50)

    routes = nwa.get_k_routes(p.loc[[349]], p, k=5, drop_middle_percent=50)


def not_test_network_analysis(points_oslo, roads_oslo):
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    pd.options.mode.chained_assignment = None

    split_lines = False

    p = points_oslo
    p = sg.clean_clip(p, p.geometry.iloc[0].buffer(2500))
    p["idx"] = p.index
    p["idx2"] = p.index

    r = roads_oslo
    r = sg.clean_clip(r, p.geometry.loc[0].buffer(3000))

    ### MAKE THE ANALYSIS CLASS
    nw = sg.DirectedNetwork(r).make_directed_network_norway().remove_isolated()
    rules = sg.NetworkAnalysisRules(
        weight="minutes",
        split_lines=split_lines,
    )
    nwa = sg.NetworkAnalysis(nw, rules=rules)
    print(nwa)

    not_test_od_cost_matrix(nwa, p)
    not_test_get_route_frequency(nwa, p)
    not_test_service_area(nwa, p)
    not_test_get_route(nwa, p)
    not_test_get_k_routes(nwa, p)


def main():
    roads_oslo = sg.read_parquet_url(
        "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet"
    )
    points_oslo = sg.read_parquet_url(
        "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet"
    )

    not_test_network_analysis(points_oslo, roads_oslo)


if __name__ == "__main__":
    import cProfile

    cProfile.run("main()", sort="cumtime")
