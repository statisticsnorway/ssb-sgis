# %%
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np


src = str(Path(__file__).parent).strip("tests") + "src"

import sys


sys.path.append(src)

import gis_utils as gs


def test_network_analysis():
    warnings.filterwarnings(action="ignore", category=UserWarning)
    warnings.filterwarnings(action="ignore", category=FutureWarning)

    ### READ FILES

    p = gpd.read_parquet(Path(__file__).parent / "testdata" / "random_points.parquet")
    p = gs.clean_clip(p, p.geometry.iloc[0].buffer(500))
    p["idx"] = p.index
    p["idx2"] = p.index

    r = gpd.read_parquet(Path(__file__).parent / "testdata" / "roads_oslo_2022.parquet")
    r = gs.clean_clip(r, p.geometry.iloc[0].buffer(600))

    ### MAKE THE ANALYSIS CLASS

    nw = (
        gs.Network(r)
        .get_largest_component()
        .close_network_holes(1.1)
        .remove_isolated()
        .cut_lines(250)
    )

    nw = gs.DirectedNetwork(r).make_directed_network_norway().remove_isolated()
    rules = gs.NetworkAnalysisRules(cost="minutes")
    nwa = gs.NetworkAnalysis(nw, rules=rules)

    ### OD COST MATRIX

    for search_factor in [0, 25, 50]:
        nwa.rules.search_factor = search_factor
        od = nwa.od_cost_matrix(p, p)
        print(
            f"percent missing, search_factor {nwa.rules.search_factor}:",
            np.mean(od[nwa.rules.cost].isna()) * 100,
            "len",
            len(od),
        )

    for search_tolerance in [100, 250, 1000]:
        nwa.rules.search_tolerance = search_tolerance
        od = nwa.od_cost_matrix(p, p)
        print(
            f"percent missing, search_factor 100, "
            f"search_tolerance {nwa.rules.search_tolerance}:",
            np.mean(od[nwa.rules.cost].isna()) * 100,
            "len",
            len(od),
        )

    od = nwa.od_cost_matrix(p, p, id_col=("idx", "idx2"), lines=True)

    print(nwa.startpoints.gdf.n_missing.value_counts())

    p1 = nwa.startpoints.gdf
    p1 = p1.loc[[p1.n_missing.idxmin()]].sample(1).idx.values[0]

    gs.qtm(od.loc[od.origin == p1], nwa.rules.cost, scheme="quantiles")

    od2 = nwa.od_cost_matrix(p, p, destination_count=3)
    assert (od2.groupby("origin")["destination"].count() <= 3).all()
    if len(od2) != len(od):
        assert np.mean(od2[nwa.rules.cost]) < np.mean(od[nwa.rules.cost])

    od = nwa.od_cost_matrix(p, p, cutoff=5)
    assert (od[nwa.rules.cost] <= 5).all()

    od = nwa.od_cost_matrix(p, p, rowwise=True)
    assert len(od) == len(p)

    ### SHORTEST PATH

    sp = nwa.shortest_path(p.iloc[[0]], p, id_col="idx", summarise=True)
    gs.qtm(sp)

    sp = nwa.shortest_path(
        p,
        p,
    )
    gs.qtm(sp)

    ### SERVICE AREA

    sa = nwa.service_area(p, impedance=5, dissolve=False)

    print(len(sa))

    sa = sa.drop_duplicates(["source", "target"])

    print(len(sa))
    gs.qtm(sa)

    sa = nwa.service_area(p.iloc[[0]], impedance=np.arange(1, 11), id_col="idx")
    sa = sa.sort_values("minutes", ascending=False)
    gs.qtm(sa, "minutes", k=10)


def main():
    test_network_analysis()


if __name__ == "__main__":
    main()
