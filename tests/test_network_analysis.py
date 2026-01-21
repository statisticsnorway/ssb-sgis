# %%
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

src = str(Path(__file__).parent).replace("tests", "") + "src"

sys.path.insert(0, src)

import sgis as sg


def not_test_od_cost_matrix(nwa, p):
    for search_factor in [0, 50]:
        nwa.rules.search_factor = search_factor
        od = nwa.od_cost_matrix(p, p)

    assert not od[nwa.rules.weight].isna().any()
    assert len(od) == len(p) * len(p), len(od)
    # assert 2 > od[nwa.rules.weight].mean() > 1, od[nwa.rules.weight].mean()

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

    if 0:
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

    od = nwa.od_cost_matrix(p, p, cutoff=1)
    assert all(od[nwa.rules.weight] <= 1), od[nwa.rules.weight].describe()

    od = nwa.od_cost_matrix(p, p, destination_count=1)
    assert all(od["origin"].value_counts() == 1), od["origin"].value_counts()


def not_test_get_route_frequency(nwa, p):
    frequencies = nwa.get_route_frequencies(p, p)
    if 0:
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
        else:
            no_identical_ods = no_identical_ods["weight"]

        frequencies = nwa.get_route_frequencies(p, p, weight_df=no_identical_ods)
        assert frequencies["frequency"].max() == 440, frequencies["frequency"].max()

    if 0:
        sg.qtm(frequencies, "frequency", title="weight_df all = * 10")

    od_pairs = pd.MultiIndex.from_product([p.index, p.index])
    one_pair_100 = pd.DataFrame(index=od_pairs).assign(weight=1)
    one_pair_100.loc[(349, 97), "weight"] = 100

    frequencies = nwa.get_route_frequencies(p, p, weight_df=one_pair_100)
    if 0:
        sg.qtm(frequencies, "frequency", title="weight_df one = * 10")
    assert frequencies["frequency"].max() == 141, frequencies["frequency"].max()

    # this should give same results
    od_pairs = pd.MultiIndex.from_product([[349], [97]])
    one_pair_100 = pd.DataFrame({"weight": [100]}, index=od_pairs)

    with_default_weight = nwa.get_route_frequencies(
        p, p, weight_df=one_pair_100, default_weight=1
    )
    if 0:
        sg.qtm(with_default_weight, "frequency", title="weight_df one = * 10")
    assert with_default_weight["frequency"].max() == 141, with_default_weight[
        "frequency"
    ].max()
    assert frequencies.equals(with_default_weight)

    if 0:
        sg.qtm(frequencies, "frequency", title="weight_df one = * 10")
    assert frequencies["frequency"].max() == 141, frequencies["frequency"].max()

    # wrong indices should raise error
    od_pairs = pd.MultiIndex.from_product([["wrong_index"], ["wrong_index"]])
    od_pairs = pd.DataFrame({"weight": [1]}, index=od_pairs)

    with pytest.raises(ValueError):
        nwa.get_route_frequencies(p, p, weight_df=od_pairs, default_weight=1)


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
    if 0:
        sg.qtm(routes)
    nwa.rules.split_lines = True
    routes = nwa.get_route(p.loc[[349]], p.loc[[440]])
    if 0:
        sg.qtm(routes)
    routes = nwa.get_route(p.loc[[349]], p.loc[[440]])
    if 0:
        sg.qtm(routes)

    nwa.rules.split_lines = False
    routes = nwa.get_route(p.loc[[349]], p)
    if 0:
        sg.qtm(routes)
    nwa.rules.split_lines = True
    routes = nwa.get_route(p.loc[[349]], p)
    if 0:
        sg.qtm(routes)

    assert list(routes.columns) == [
        "origin",
        "destination",
        nwa.rules.weight,
        "geometry",
    ]
    routes = nwa.get_route(p.loc[[349]], p)
    if 0:
        sg.qtm(routes)

    p = p.sort_index(ascending=True)
    p_rev = p.sort_index(ascending=False)
    routes = nwa.get_route(p, p_rev, rowwise=True)
    assert not routes.isna().any().any(), routes.isna().any()

    nwa.rules.search_tolerance = 1
    routes = nwa.get_route(p.loc[[349]], p)
    assert not len(routes), routes
    nwa.rules.search_tolerance = sg.NetworkAnalysisRules.search_tolerance

    routes = nwa.get_route(p, p, cutoff=1)
    assert (routes[nwa.rules.weight] <= 1).all(), routes[nwa.rules.weight].describe()

    routes = nwa.get_route(p, p, destination_count=1)
    assert (routes["origin"].value_counts() == 1).all(), routes["origin"].value_counts()


def not_test_service_area(nwa, p):
    sa = nwa.service_area(p, breaks=5, dissolve=False)

    print(len(sa))

    sa = sa.drop_duplicates(["source", "target"])

    print(len(sa))
    if 0:
        sg.qtm(sa)

    sa = nwa.service_area(p.loc[[349]], breaks=np.arange(1, 11))
    print(sa.columns)
    sa = sa.sort_values("minutes", ascending=False)
    if 0:
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
        if 0:
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
            if 0:
                sg.qtm(routes, "k")
        except ValueError:
            n += 1
            print("drop_middle_percent works as expected", x)

    assert n == 2

    routes = nwa.get_k_routes(p.loc[[349]], p.loc[[440]], k=5, drop_middle_percent=50)
    print(routes)
    if 0:
        sg.qtm(routes)

    routes = nwa.get_k_routes(p.loc[[349]], p, k=5, drop_middle_percent=50)
    if 0:
        sg.qtm(routes)

    nwa.rules.search_tolerance = 1
    routes = nwa.get_k_routes(p.loc[[349]], p, k=5, drop_middle_percent=50)
    assert not len(routes), routes
    nwa.rules.search_tolerance = sg.NetworkAnalysisRules.search_tolerance

    routes = nwa.get_k_routes(p, p, cutoff=1, k=5, drop_middle_percent=50)
    assert all(routes[nwa.rules.weight] <= 1), routes[nwa.rules.weight].describe()

    routes = nwa.get_k_routes(p, p, destination_count=1, k=5, drop_middle_percent=50)
    assert all(routes["origin"].value_counts() == 1), routes["origin"].value_counts()


def not_test_direction(roads_oslo):
    """Check that a route that should go in separate tunnels, goes in correct tunnels."""
    vippetangen = sg.to_gdf([10.741527, 59.9040595], crs=4326).to_crs(roads_oslo.crs)
    ryen = sg.to_gdf([10.8047522, 59.8949826], crs=4326).to_crs(roads_oslo.crs)

    # direction vippetangen-ryen should be 2-ish meters from this point
    tunnel_fromto = sg.to_gdf([10.7730091, 59.899740], crs=4326).to_crs(roads_oslo.crs)

    # direction ryen-vippetangen should be 2-ish meters from this point
    tunnel_tofrom = sg.to_gdf([10.7724645, 59.899908], crs=4326).to_crs(roads_oslo.crs)

    clipped = sg.clean_clip(roads_oslo, tunnel_fromto.buffer(2000))
    connected_roads = sg.get_connected_components(clipped).query("connected == 1")
    directed_roads = sg.make_directed_network_norway(connected_roads, dropnegative=True)
    rules = sg.NetworkAnalysisRules(directed=True, weight="minutes")
    nwa = sg.NetworkAnalysis(directed_roads, rules=rules)

    route_fromto = nwa.get_route(vippetangen, ryen)
    route_tofrom = nwa.get_route(ryen, vippetangen)

    m = 5
    should_be_within = route_fromto.sjoin_nearest(tunnel_fromto, distance_col="dist")
    assert should_be_within["dist"].max() < m, should_be_within["dist"]
    should_be_within = route_tofrom.sjoin_nearest(tunnel_tofrom, distance_col="dist")
    assert should_be_within["dist"].max() < m, should_be_within["dist"]

    should_not_be_within = route_fromto.sjoin_nearest(
        tunnel_tofrom, distance_col="dist"
    )
    assert should_not_be_within["dist"].max() > m, should_not_be_within["dist"]

    should_not_be_within = route_tofrom.sjoin_nearest(
        tunnel_fromto, distance_col="dist"
    )
    assert should_not_be_within["dist"].max() > m, should_not_be_within["dist"]


def test_network_analysis(points_oslo, roads_oslo):
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    pd.options.mode.chained_assignment = None

    split_lines = False

    rules = sg.NetworkAnalysisRules(
        directed=True,
        weight="minutes",
        split_lines=split_lines,
    )
    connected_roads = sg.get_connected_components(roads_oslo).query("connected == 1")
    directed_roads = sg.make_directed_network_norway(connected_roads, dropnegative=True)
    nwa = sg.NetworkAnalysis(directed_roads, rules=rules, detailed_log=True)
    print(nwa)

    od = nwa.od_cost_matrix(points_oslo.iloc[[0]], points_oslo, lines=True)
    print(od)
    if 0:
        sg.qtm(od, column=nwa.rules.weight, scheme="quantiles")
    assert (times := list(od["minutes"].dropna().astype(int))[:100]) == [
        0,
        13,
        10,
        8,
        14,
        14,
        13,
        4,
        13,
        8,
        5,
        13,
        11,
        12,
        14,
        6,
        6,
        11,
        18,
        16,
        5,
        23,
        7,
        13,
        4,
        5,
        3,
        5,
        3,
        19,
        7,
        13,
        14,
        10,
        4,
        11,
        9,
        12,
        11,
        12,
        15,
        2,
        19,
        3,
        21,
        8,
        5,
        12,
        7,
        12,
        5,
        6,
        22,
        7,
        15,
        8,
        24,
        9,
        10,
        12,
        3,
        9,
        7,
        9,
        6,
        7,
        13,
        10,
        6,
        3,
        13,
        11,
        15,
        16,
        4,
        9,
        10,
        12,
        10,
        5,
        12,
        15,
        13,
        7,
        5,
        21,
        12,
        1,
        13,
        9,
        15,
        13,
        21,
        15,
        12,
        6,
        0,
        14,
        11,
        8,
    ], times

    p = points_oslo
    p = sg.clean_clip(p, p.geometry.iloc[0].buffer(700))
    p["idx"] = p.index
    p["idx2"] = p.index

    r = roads_oslo
    r = sg.clean_clip(r, p.geometry.loc[0].buffer(750))

    connected_roads = sg.get_connected_components(r).query("connected == 1")

    directed_roads = sg.make_directed_network(
        connected_roads,
        dropna=False,
        dropnegative=True,
        direction_col="oneway",
        direction_vals_bft=("B", "FT", "TF"),
        minute_cols=("drivetime_fw", "drivetime_bw"),
    )
    nwa = sg.NetworkAnalysis(directed_roads, rules=rules, detailed_log=True)

    print(nwa)

    not_test_od_cost_matrix(nwa, p)
    not_test_get_route_frequency(nwa, p)
    not_test_service_area(nwa, p)
    not_test_get_route(nwa, p)
    not_test_get_k_routes(nwa, p)
    not_test_direction(roads_oslo)

    rules = {
        "directed": True,
        "weight": "minutes",
        "split_lines": split_lines,
    }
    nwa = sg.NetworkAnalysis(directed_roads, rules=rules, detailed_log=True)
    print(nwa)
    print(nwa.rules)

    not_test_od_cost_matrix(nwa, p)
    not_test_get_route_frequency(nwa, p)
    not_test_service_area(nwa, p)
    not_test_get_route(nwa, p)
    not_test_get_k_routes(nwa, p)
    not_test_direction(roads_oslo)


def main():
    from oslo import points_oslo
    from oslo import roads_oslo

    test_network_analysis(points_oslo(), roads_oslo())


if __name__ == "__main__":
    main()
