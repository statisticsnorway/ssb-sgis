"""Benchmark network analysis."""

# %%
import sys
import warnings
from pathlib import Path

import pandas as pd

src = str(Path(__file__).parent).replace("tests", "") + "src"

sys.path.insert(0, src)

import sgis as sg


def not_test_od_cost_matrix(nwa, p):
    od = nwa.od_cost_matrix(p, p, lines=True)


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

    p = points_oslo.loc[:100]

    ### MAKE THE ANALYSIS CLASS
    nw = (
        sg.get_connected_components(roads_oslo)
        .query("connected == 1")
        .pipe(sg.make_directed_network_norway)
    )
    rules = sg.NetworkAnalysisRules(weight="minutes", directed=True)
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
