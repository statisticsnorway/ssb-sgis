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


def test_network_analysis(points_oslo, roads_oslo):
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    #    warnings.filterwarnings(action="ignore", category=UserWarning)
    pd.options.mode.chained_assignment = None

    split_lines = False

    ### READ FILES

    p = points_oslo
    p = sg.clean_clip(p, p.geometry.iloc[0].buffer(700))
    p["idx"] = p.index
    p["idx2"] = p.index

    r = roads_oslo
    r = sg.clean_clip(r, p.geometry.loc[0].buffer(750))

    def run_analyses(nwa, p):
        x = nwa.get_route_frequencies(p.loc[p.idx == 0], p.sample(7))
        sg.qtm(x, "n")

        ### OD COST MATRIX

        for search_factor in [0, 25, 50]:
            nwa.rules.search_factor = search_factor
            od = nwa.od_cost_matrix(p, p)

        nwa.rules.search_factor = 10

        for search_tolerance in [100, 250, 1000]:
            nwa.rules.search_tolerance = search_tolerance
            od = nwa.od_cost_matrix(p, p)

        print(
            nwa.log[
                ["search_tolerance", "search_factor", "percent_missing", "cost_mean"]
            ]
        )

        assert all(nwa.log["percent_missing"] == 0)
        assert all(nwa.log["cost_mean"] < 3)
        assert all(nwa.log["cost_mean"] > 0)

        od = nwa.od_cost_matrix(p, p, id_col=("idx", "idx2"), lines=True)

        p1 = nwa.origins.gdf
        p1 = p1.loc[[p1.missing.idxmin()]].sample(1).idx.values[0]

        sg.qtm(od.loc[od.origin == p1], nwa.rules.weight, scheme="quantiles")

        od2 = nwa.od_cost_matrix(p, p, destination_count=3)

        assert (od2.groupby("origin")["destination"].count() <= 3).mean() > 0.6

        if len(od2) != len(od):
            assert np.mean(od2[nwa.rules.weight]) < np.mean(od[nwa.rules.weight])

        od = nwa.od_cost_matrix(p, p, cutoff=5)
        assert (od[nwa.rules.weight] <= 5).all()

        od = nwa.od_cost_matrix(p, p, rowwise=True)
        assert len(od) == len(p)

        ### GET ROUTE

        sp = nwa.get_route(p, p, id_col="idx")

        sp = nwa.get_route(p.iloc[[0]], p, id_col="idx")

        i = 1
        nwa.rules.search_factor = 0
        nwa.rules.split_lines = False

        sp = nwa.get_route(p.iloc[[0]], p.iloc[[i]], id_col="idx")
        sg.qtm(sp)
        nwa.rules.split_lines = True
        sp = nwa.get_route(p.iloc[[0]], p.iloc[[i]], id_col="idx")
        sg.qtm(sp)
        sp = nwa.get_route(p.iloc[[0]], p.iloc[[i]], id_col="idx")
        sg.qtm(sp)

        nwa.rules.split_lines = False
        sp = nwa.get_route(p.iloc[[0]], p, id_col="idx")
        sg.qtm(sp)
        nwa.rules.split_lines = True
        sp = nwa.get_route(p.iloc[[0]], p, id_col="idx")
        sg.qtm(sp)

        sp = nwa.get_route(p.iloc[[0]], p, id_col="idx")
        sg.qtm(sp)

        ### GET ROUTE FREQUENCIES
        print(len(p))
        print(len(p))
        print(len(p))
        sp = nwa.get_route_frequencies(p.iloc[[0]], p)
        sg.qtm(sp)

        ### SERVICE AREA

        sa = nwa.service_area(p, breaks=5, dissolve=False)

        print(len(sa))

        sa = sa.drop_duplicates(["source", "target"])

        print(len(sa))
        sg.qtm(sa)

        sa = nwa.service_area(p.iloc[[0]], breaks=np.arange(1, 11), id_col="idx")
        print(sa.columns)
        sa = sa.sort_values("minutes", ascending=False)
        sg.qtm(sa, "minutes", k=10)

        ### GET K ROUTES

        i = 1

        for x in [0, 50, 100]:
            sp = nwa.get_k_routes(
                p.iloc[[0]], p.iloc[[i]], k=5, drop_middle_percent=x, id_col="idx"
            )
            sg.qtm(sp, "k")

        n = 0
        for x in [-1, 101]:
            try:
                sp = nwa.get_k_routes(
                    p.iloc[[0]], p.iloc[[i]], k=5, drop_middle_percent=x, id_col="idx"
                )
                sg.qtm(sp, "k")
            except ValueError:
                n += 1
                print("drop_middle_percent works as expected", x)

        assert n == 2

        i += 1
        sp = nwa.get_k_routes(
            p.iloc[[0]], p.iloc[[i]], k=5, drop_middle_percent=50, id_col="idx"
        )
        print(sp)
        sg.qtm(sp)

        sp = nwa.get_k_routes(p.iloc[[0]], p, k=5, drop_middle_percent=50, id_col="idx")
        sg.qtm(sp)

    ### MAKE THE ANALYSIS CLASS
    nw = (
        sg.DirectedNetwork(r)
        .make_directed_network_norway(minute_cols=("drivetime_fw", "drivetime_bw"))
        .close_network_holes(1.1, fillna=0, deadends_only=False)
        .get_largest_component()
    )
    sg.qtm(nw.gdf, "connected", scheme="equalinterval")
    sg.qtm(nw.gdf, "hole")
    print(nw.gdf.hole.value_counts())

    nw = nw.remove_isolated()

    rules = sg.NetworkAnalysisRules(
        weight="minutes",
        split_lines=split_lines,
    )

    nwa = sg.NetworkAnalysis(nw, rules=rules)
    print(nwa)

    run_analyses(nwa, p)

    nw = sg.DirectedNetwork(r).make_directed_network_norway().remove_isolated()

    rules = sg.NetworkAnalysisRules(
        weight="minutes",
        split_lines=split_lines,
    )

    nwa = sg.NetworkAnalysis(nw, rules=rules)
    print(nwa)

    run_analyses(nwa, p)


def main():
    roads_oslo = sg.read_parquet_url(
        "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/roads_oslo_2022.parquet"
    )
    points_oslo = sg.read_parquet_url(
        "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet"
    )
    """
    roads_oslo = gpd.read_parquet(
        r"C:/Users/ort/git/ssb-sgis/tests/testdata/roads_eidskog_2022.parquet"
    )
    points_oslo = gpd.read_parquet(
        r"C:/Users/ort/git/ssb-sgis/tests/testdata/points_eidskog.parquet"
    )
    """
    test_network_analysis(points_oslo, roads_oslo)


if __name__ == "__main__":
    # import cProfile
    # cProfile.run("main()", sort="cumtime")
    main()
