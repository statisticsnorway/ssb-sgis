# %%
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


src = str(Path(__file__).parent).strip("tests") + "src"

import sys


sys.path.insert(0, src)

import gis_utils as gs


def test_network_analysis():
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    #    warnings.filterwarnings(action="ignore", category=UserWarning)
    pd.options.mode.chained_assignment = None

    split_lines = True

    ### READ FILES

    p = gpd.read_parquet(gs.pointpath)
    p = gs.clean_clip(p, p.geometry.iloc[0].buffer(500))
    p["idx"] = p.index
    p["idx2"] = p.index
    print(p.idx)

    r = gpd.read_parquet(gs.roadpath)
    r = gs.clean_clip(r, p.geometry.iloc[0].buffer(700))

    ### MAKE THE ANALYSIS CLASS

    nw = gs.DirectedNetwork(r).remove_isolated().make_directed_network_norway()
    rules = gs.NetworkAnalysisRules(weight="minutes", split_lines=split_lines)
    nwa = gs.NetworkAnalysis(nw, rules=rules)
    print(nwa)

    ### OD COST MATRIX

    for search_factor in [0, 25, 50]:
        nwa.rules.search_factor = search_factor
        od = nwa.od_cost_matrix(p, p)

    nwa.rules.search_factor = 10

    for search_tolerance in [100, 250, 1000]:
        nwa.rules.search_tolerance = search_tolerance
        od = nwa.od_cost_matrix(p, p)

    print(
        nwa.log[["search_tolerance", "search_factor", "percent_missing", "cost_mean"]]
    )

    od = nwa.od_cost_matrix(p, p, id_col=("idx", "idx2"), lines=True)

    print(nwa.origins.gdf.missing.value_counts())

    p1 = nwa.origins.gdf
    p1 = p1.loc[[p1.missing.idxmin()]].sample(1).idx.values[0]

    gs.qtm(od.loc[od.origin == p1], nwa.rules.weight, scheme="quantiles")

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
    gs.qtm(sp)
    nwa.rules.split_lines = True
    sp = nwa.get_route(p.iloc[[0]], p.iloc[[i]], id_col="idx")
    gs.qtm(sp)

    nwa.rules.split_lines = False
    sp = nwa.get_route(p.iloc[[0]], p, id_col="idx")
    gs.qtm(sp)
    nwa.rules.split_lines = True
    sp = nwa.get_route(p.iloc[[0]], p, id_col="idx")
    gs.qtm(sp)

    sp = nwa.get_route(p.iloc[[0]], p, id_col="idx")
    gs.qtm(sp)

    ### GET K ROUTES

    i = 1

    for x in [0, 50, 100]:
        sp = nwa.get_k_routes(
            p.iloc[[0]], p.iloc[[i]], k=5, drop_middle_percent=x, id_col="idx"
        )
        gs.qtm(sp)

    for x in [-1, 101]:
        try:
            sp = nwa.get_k_routes(
                p.iloc[[0]], p.iloc[[i]], k=5, drop_middle_percent=x, id_col="idx"
            )
            gs.qtm(sp)
        except ValueError:
            print("get_k_routes works as expected", x)

    i += 1
    sp = nwa.get_k_routes(
        p.iloc[[0]], p.iloc[[i]], k=5, drop_middle_percent=50, id_col="idx"
    )
    print(sp)
    gs.qtm(sp)

    sp = nwa.get_k_routes(p.iloc[[0]], p, k=5, drop_middle_percent=50, id_col="idx")
    gs.qtm(sp)

    ### GET ROUTE FREQUENCIES
    print(len(p))
    print(len(p))
    print(len(p))
    print(len(p))
    sp = nwa.get_route_frequencies(p.iloc[[0]], p)
    gs.qtm(sp)

    ### SERVICE AREA

    sa = nwa.service_area(p, breaks=5, dissolve=False)

    print(len(sa))

    sa = sa.drop_duplicates(["source", "target"])

    print(len(sa))
    gs.qtm(sa)

    sa = nwa.service_area(p.iloc[[0]], breaks=np.arange(1, 11), id_col="idx")
    sa = sa.sort_values("minutes", ascending=False)
    gs.qtm(sa, "minutes", k=10)


def main():
    test_network_analysis()
    import cProfile

    # cProfile.run("test_network_analysis()", sort="cumtime")


if __name__ == "__main__":
    main()
