# %%
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


src = str(Path(__file__).parent).strip("tests") + "src"

sys.path.insert(0, src)

import gis_utils as gs


def test_split_lines():
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    #    warnings.filterwarnings(action="ignore", category=UserWarning)
    pd.options.mode.chained_assignment = None

    ### READ FILES

    points = gpd.read_parquet(gs.pointpath)

    r = gpd.read_parquet(gs.roadpath)

    r = gs.clean_clip(r, points.geometry.loc[0].buffer(700))
    points = gs.clean_clip(points, points.geometry.loc[0].buffer(700))

    ### MAKE THE ANALYSIS CLASS
    nw = gs.DirectedNetwork(r).make_directed_network_norway().remove_isolated()
    rules = gs.NetworkAnalysisRules(
        weight="minutes",
    )

    nwa = gs.NetworkAnalysis(nw, rules=rules, detailed_log=False)
    print(nwa)

    nwa.rules.split_lines = False

    od = nwa.od_cost_matrix(points, points)
    sp1 = nwa.get_route(points.loc[[97]], points.loc[[135]])
    sp1["split_lines"] = "Not splitted"

    nwa.rules.split_lines = True

    od = nwa.od_cost_matrix(points, points)
    print(nwa.log[["method", "cost_mean", "percent_missing"]])
    # repeat to see if something dodgy happens
    for _ in range(3):
        sp2 = nwa.get_route(points.loc[[97]], points.loc[[135]])
    sp2["split_lines"] = "Splitted"

    gs.qtm(gs.gdf_concat([sp1, sp2]), column="split_lines", cmap="bwr")


def main():
    test_split_lines()


# import cProfile
# cProfile.run("test_network_analysis()", sort="cumtime")


if __name__ == "__main__":
    main()
# %%
""
