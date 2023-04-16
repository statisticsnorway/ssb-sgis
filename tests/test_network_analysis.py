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

    assert not od[nwa.rules.weight].isna().any()
    assert len(od) == len(p) * len(p), len(od)
    assert 2 > od[nwa.rules.weight].mean() > 1, od[nwa.rules.weight].mean()

    assert list(od.columns) == ["origin", "destination", nwa.rules.weight]

    nwa.rules.search_factor = 10

    for search_tolerance in [100, 1000]:
        nwa.rules.search_tolerance = search_tolerance
        od = nwa.od_cost_matrix(p, p)

    print(
        nwa.log[["search_tolerance", "search_factor", "percent_missing", "cost_mean"]]
    )

    od = nwa.od_cost_matrix(p, p, lines=True)
    assert list(od.columns) == [
        "origin",
        "destination",
        nwa.rules.weight,
        "geometry",
    ]
    assert sg.get_geom_type(od) == "line"

    if __name__ == "__main__":
        sg.qtm(od.loc[od.origin == 0], nwa.rules.weight, scheme="quantiles")

    p = p.sort_index(ascending=True)
    p_rev = p.sort_index(ascending=False)
    print(p)
    print(p_rev)
    od = nwa.od_cost_matrix(p, p_rev, rowwise=True)
    print(od)
    assert len(od) == len(p)
    assert not od[nwa.rules.weight].isna().any()

    assert all(nwa.log["percent_missing"] == 0), nwa.log["percent_missing"]
    assert all(nwa.log["cost_mean"] < 3), nwa.log["cost_mean"]
    assert all(nwa.log["cost_mean"] > 1), nwa.log["cost_mean"]


def not_test_get_route_frequency(nwa, p):
    frequencies = nwa.get_route_frequencies(p, p)
    if __name__ == "__main__":
        sg.qtm(frequencies, "frequency")
    assert frequencies["frequency"].max() == 44, frequencies["frequency"].max()

    # testing weight_df with and without reset_index
    # first, weight_df will have 1 column and multiindex, second, weight_df will have
    # three columns (origin, destination and weight. Column names doesn't matter)
    for truefalse in [0, 1]:
        od_pairs = pd.MultiIndex.from_product([p.index, p.index])
        weight_df = pd.DataFrame(index=od_pairs)
        weight_df["weight"] = 10

        no_identical_ods = weight_df.loc[
            weight_df.index.get_level_values(0) != weight_df.index.get_level_values(1)
        ]

        if truefalse:
            no_identical_ods = no_identical_ods.reset_index()

        frequencies = nwa.get_route_frequencies(p, p, weight_df=no_identical_ods)
        assert frequencies["frequency"].max() == 440, frequencies["frequency"].max()

    if __name__ == "__main__":
        sg.qtm(frequencies, "frequency", title="weight_df all = * 10")

    od_pairs = pd.MultiIndex.from_product([p.index, p.index])
    one_pair_100 = pd.DataFrame(index=od_pairs).assign(weight=1)
    one_pair_100.loc[(349, 97), "weight"] = 100

    frequencies = nwa.get_route_frequencies(p, p, weight_df=one_pair_100)
    if __name__ == "__main__":
        sg.qtm(frequencies, "frequency", title="weight_df one = * 10")
    assert frequencies["frequency"].max() == 141, frequencies["frequency"].max()

    # this should give same results
    od_pairs = pd.MultiIndex.from_product([[349], [97]])
    one_pair_100 = pd.DataFrame({"weight": [100]}, index=od_pairs)

    with_default_weight = nwa.get_route_frequencies(
        p, p, weight_df=one_pair_100, default_weight=1
    )
    if __name__ == "__main__":
        sg.qtm(with_default_weight, "frequency", title="weight_df one = * 10")
    assert with_default_weight["frequency"].max() == 141, with_default_weight[
        "frequency"
    ].max()
    assert frequencies.equals(with_default_weight)

    if __name__ == "__main__":
        sg.qtm(frequencies, "frequency", title="weight_df one = * 10")
    assert frequencies["frequency"].max() == 141, frequencies["frequency"].max()

    frequencies = nwa.get_route_frequencies(p.loc[[349]], p)
    if __name__ == "__main__":
        sg.qtm(frequencies)


def not_test_get_route(nwa, p):
    routes = nwa.get_route(p, p)
    assert not routes.isna().any().any(), routes.isna().any()
    assert int(routes.length.mean()) == 1025, routes.length.mean()
    assert list(routes.columns) == [
        "origin",
        "destination",
        nwa.rules.weight,
        "geometry",
    ]

    routes = nwa.get_route(p.loc[[349]], p)

    nwa.rules.search_factor = 0
    nwa.rules.split_lines = False

    routes = nwa.get_route(p.loc[[349]], p.loc[[440]])
    if __name__ == "__main__":
        sg.qtm(routes)
    nwa.rules.split_lines = True
    routes = nwa.get_route(p.loc[[349]], p.loc[[440]])
    if __name__ == "__main__":
        sg.qtm(routes)
    routes = nwa.get_route(p.loc[[349]], p.loc[[440]])
    if __name__ == "__main__":
        sg.qtm(routes)

    nwa.rules.split_lines = False
    routes = nwa.get_route(p.loc[[349]], p)
    if __name__ == "__main__":
        sg.qtm(routes)
    nwa.rules.split_lines = True
    routes = nwa.get_route(p.loc[[349]], p)
    if __name__ == "__main__":
        sg.qtm(routes)

    assert list(routes.columns) == [
        "origin",
        "destination",
        nwa.rules.weight,
        "geometry",
    ]
    routes = nwa.get_route(p.loc[[349]], p)
    if __name__ == "__main__":
        sg.qtm(routes)

    p = p.sort_index(ascending=True)
    p_rev = p.sort_index(ascending=False)
    routes = nwa.get_route(p, p_rev, rowwise=True)
    assert not routes.isna().any().any(), routes.isna().any()


def not_test_service_area(nwa, p):
    sa = nwa.service_area(p, breaks=5, dissolve=False)

    print(len(sa))

    sa = sa.drop_duplicates(["source", "target"])

    print(len(sa))
    if __name__ == "__main__":
        sg.qtm(sa)

    sa = nwa.service_area(p.loc[[349]], breaks=np.arange(1, 11))
    print(sa.columns)
    sa = sa.sort_values("minutes", ascending=False)
    if __name__ == "__main__":
        sg.qtm(sa, "minutes", k=10)
    assert list(sa.columns) == [
        "origin",
        nwa.rules.weight,
        "geometry",
    ]


def not_test_get_k_routes(nwa, p):
    for x in [0, 100]:
        routes = nwa.get_k_routes(
            p.loc[[349]], p.loc[[440]], k=5, drop_middle_percent=x
        )
        if __name__ == "__main__":
            sg.qtm(routes, "k")

    assert list(routes.columns) == [
        "origin",
        "destination",
        nwa.rules.weight,
        "k",
        "geometry",
    ], list(routes.columns)

    n = 0
    for x in [-1, 101]:
        try:
            routes = nwa.get_k_routes(
                p.loc[[349]],
                p.loc[[440]],
                k=5,
                drop_middle_percent=x,
            )
            if __name__ == "__main__":
                sg.qtm(routes, "k")
        except ValueError:
            n += 1
            print("drop_middle_percent works as expected", x)

    assert n == 2

    routes = nwa.get_k_routes(p.loc[[349]], p.loc[[440]], k=5, drop_middle_percent=50)
    print(routes)
    if __name__ == "__main__":
        sg.qtm(routes)

    routes = nwa.get_k_routes(p.loc[[349]], p, k=5, drop_middle_percent=50)
    if __name__ == "__main__":
        sg.qtm(routes)


def test_network_analysis(points_oslo, roads_oslo):
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    pd.options.mode.chained_assignment = None

    split_lines = False

    p = points_oslo
    p = sg.clean_clip(p, p.geometry.iloc[0].buffer(700))
    p["idx"] = p.index
    p["idx2"] = p.index

    r = roads_oslo
    r = sg.clean_clip(r, p.geometry.loc[0].buffer(750))

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

    test_network_analysis(points_oslo, roads_oslo)


if __name__ == "__main__":
    main()

# %%
"""
roads_oslo = sg.read_parquet_url(
    "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet"
)
nw = sg.DirectedNetwork(roads_oslo).make_directed_network_norway().remove_isolated()
rules = sg.NetworkAnalysisRules(weight="minutes")
nwa = sg.NetworkAnalysis(nw, rules=rules)
print(nwa)
"""
