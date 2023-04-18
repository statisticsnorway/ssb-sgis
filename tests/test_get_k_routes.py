# %%
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd


src = str(Path(__file__).parent).strip("tests") + "src"


sys.path.insert(0, src)

import sgis as sg


def not_test_network_analysis(roads_oslo, points_oslo):
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    #    warnings.filterwarnings(action="ignore", category=UserWarning)
    pd.options.mode.chained_assignment = None

    split_lines = False

    ### READ FILES

    points = points_oslo
    p = points.iloc[[0]]

    r = roads_oslo
    #    r = sg.clean_clip(r, p.buffer(1000))
    #   points = sg.clean_clip(points, p.buffer(800))

    ### MAKE THE ANALYSIS CLASS

    nw = (
        sg.get_connected_components(roads_oslo)
        .query("connected == 1")
        .pipe(sg.make_directed_network_norway)
    )
    rules = sg.NetworkAnalysisRules(weight="minutes", split_lines=split_lines)
    nwa = sg.NetworkAnalysis(nw, rules=rules)

    ### GET ROUTE
    for i in range(10):
        print(i)
        p1 = points.sample(1)
        try:
            k = nwa.get_k_routes(p, p1, k=5, drop_middle_percent=50)
            sg.qtm(k, nwa.rules.weight)
            k = nwa.get_k_routes(p, p1, k=5, drop_middle_percent=25)
            sg.qtm(k, nwa.rules.weight)
            k = nwa.get_k_routes(p, p1, k=5, drop_middle_percent=10)
            sg.qtm(k, nwa.rules.weight)
        except Exception as e:
            raise e


def main():
    from oslo import points_oslo, roads_oslo

    not_test_network_analysis(roads_oslo(), points_oslo())


if __name__ == "__main__":
    main()
