# %%
import cProfile
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np


src = str(Path(__file__).parent).strip("tests") + "src"

import sys


sys.path.insert(0, src)

import gis_utils as gs


def not_test_network_analysis():
    warnings.filterwarnings(action="ignore", category=FutureWarning)

    ### READ FILES

    p = gpd.read_parquet(Path(__file__).parent / "testdata" / "random_points.parquet")
    p["idx"] = p.index
    p["idx2"] = p.index

    r = gpd.read_parquet(Path(__file__).parent / "testdata" / "roads_oslo_2022.parquet")

    ### MAKE THE ANALYSIS CLASS

    nw = (
        gs.Network(r)
        .get_largest_component()
        .close_network_holes(1.1)
        .remove_isolated()
        .cut_lines(250)
    )

    if (l := max(nw.gdf.length)) > 250 + 1:
        raise ValueError(f"cut_lines did not cut lines. max line length: {l}")

    nw = gs.DirectedNetwork(r).make_directed_network_norway().remove_isolated()
    rules = gs.NetworkAnalysisRules(weight="minutes", split_lines=True)
    nwa = gs.NetworkAnalysis(nw, rules=rules)

    ### OD COST MATRIX

    for search_factor in [0, 25, 50]:
        nwa.rules.search_factor = search_factor
        od = nwa.od_cost_matrix(p, p)

    nwa.rules.search_factor = 10

    for search_tolerance in [100, 250, 1000]:
        nwa.rules.search_tolerance = search_tolerance
        od = nwa.od_cost_matrix(p, p)

    print(nwa.log[["search_tolerance", "search_factor", "percent_missing"]])

    od = nwa.od_cost_matrix(p, p, id_col=("idx", "idx2"), lines=True)

    print(nwa.origins.gdf.n_missing.value_counts())

    p1 = nwa.origins.gdf
    p1 = p1.loc[[p1.n_missing.idxmin()]].sample(1).idx.values[0]

    gs.qtm(od.loc[od.origin == p1], nwa.rules.weight, scheme="quantiles")

    od2 = nwa.od_cost_matrix(p, p, destination_count=3)
    print(od2.groupby("origin")["destination"].count().value_counts())
    print(od2)
    print(od2.groupby("origin")["destination"].count())
    assert (od2.groupby("origin")["destination"].count() <= 3).mean() > 0.95
    if len(od2) != len(od):
        assert np.mean(od2[nwa.rules.weight]) < np.mean(od[nwa.rules.weight])

    od = nwa.od_cost_matrix(p, p, cutoff=5)
    assert (od[nwa.rules.weight] <= 5).all()

    od = nwa.od_cost_matrix(p, p, rowwise=True)
    assert len(od) == len(p)

    ### SHORTEST PATH

    sp = nwa.get_route(p.iloc[[0]], p, id_col="idx", summarise=True)
    gs.qtm(sp)

    sp = nwa.get_route(
        p,
        p,
    )
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
    not_test_network_analysis()


#    cProfile.run("not_test_network_analysis()", sort="cumtime")


if __name__ == "__main__":
    main()