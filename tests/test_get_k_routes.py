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


def not_test_network_analysis():
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    #    warnings.filterwarnings(action="ignore", category=UserWarning)
    pd.options.mode.chained_assignment = None

    split_lines = True

    ### READ FILES

    p = gpd.read_parquet(Path(__file__).parent / "testdata" / "random_points.parquet")
    #  p = gs.clean_clip(p, p.geometry.iloc[0].buffer(3000))
    p["idx"] = p.index
    p["idx2"] = p.index

    r = gpd.read_parquet(Path(__file__).parent / "testdata" / "roads_oslo_2022.parquet")
    #   r = gs.clean_clip(r, p.geometry.iloc[0].buffer(5000))

    ### MAKE THE ANALYSIS CLASS

    nw = gs.DirectedNetwork(r).remove_isolated().make_directed_network_norway()
    rules = gs.NetworkAnalysisRules(weight="minutes", split_lines=split_lines)
    nwa = gs.NetworkAnalysis(nw, rules=rules)

    ### GET ROUTE
    for i in range(10):
        print(i)
        p1 = p.sample(1)
        try:
            k = nwa.get_k_routes(
                p.iloc[[0]], p1, k=5, id_col="idx", drop_middle_percent=50
            )
            gs.qtm(k, nwa.rules.weight)
            k = nwa.get_k_routes(
                p.iloc[[0]], p1, k=5, id_col="idx", drop_middle_percent=25
            )
            gs.qtm(k, nwa.rules.weight)
            k = nwa.get_k_routes(
                p.iloc[[0]], p1, k=5, id_col="idx", drop_middle_percent=10
            )
            gs.qtm(k, nwa.rules.weight)
        except Exception as e:
            raise e
            print(e)


def main():
    not_test_network_analysis()
    import cProfile

    # cProfile.run("not_test_network_analysis()", sort="cumtime")


if __name__ == "__main__":
    main()
