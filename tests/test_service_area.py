# %%
import sys
import timeit
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


src = str(Path(__file__).parent).strip("tests") + "src"

sys.path.insert(0, src)

import sgis as sg


def test_service_area(points_oslo, roads_oslo):
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    #    warnings.filterwarnings(action="ignore", category=UserWarning)
    pd.options.mode.chained_assignment = None

    p = points_oslo
    p = sg.clean_clip(p, p.geometry.iloc[0].buffer(700))
    p["idx"] = p.index
    p["idx2"] = p.index

    point = p.iloc[[1]]

    r = roads_oslo
    r = sg.clean_clip(r, p.geometry.loc[0].buffer(750))
    r100 = sg.clean_clip(r, point.buffer(310))

    rules = sg.NetworkAnalysisRules(
        directed=False,
        weight="meters",
    )

    nw = sg.get_connected_components(r).query("connected == 1")
    nwa = sg.NetworkAnalysis(nw, rules=rules)

    x = nwa.service_area(point, breaks=30, dissolve=False)
    assert all(x.geometry.isna()) and len(x) == 1
    x = nwa.service_area(point, breaks=125, dissolve=False)
    if __name__ == "__main__":
        sg.qtm(r100, x, point, title="undirected 125")
    assert len(x) == 8

    x = nwa.precice_service_area(point, breaks=30, dissolve=False)
    if __name__ == "__main__":
        sg.qtm(r100, x, point, title="undirected 30, precice")
    assert len(x) == 4

    x = nwa.precice_service_area(point, breaks=125, dissolve=False)
    if __name__ == "__main__":
        sg.qtm(r100, x, point, title="undirected 125, precice")
    assert len(x) == 19

    nwa.rules.split_lines = True
    x = nwa.precice_service_area(point, breaks=125, dissolve=False)
    if __name__ == "__main__":
        sg.qtm(
            r100,
            x,
            point,
            title="undirected 125, precice,\n split_lines=True",
        )
    assert len(x) == 10, x

    rules.split_lines = False

    nw = sg.make_directed_network_norway(nw, dropnegative=True)
    rules = sg.NetworkAnalysisRules(
        directed=True,
        weight="meters",
    )
    nwa = sg.NetworkAnalysis(nw, rules=rules)

    x = nwa.service_area(point, breaks=30, dissolve=False)
    assert all(x.geometry.isna()) and len(x) == 1
    x = nwa.service_area(point, breaks=125, dissolve=False)
    assert len(x) == 3
    if __name__ == "__main__":
        sg.qtm(r100, x, point, title="directed 125")

    x = nwa.precice_service_area(point, breaks=30, dissolve=False)
    assert len(x) == 2
    if __name__ == "__main__":
        sg.qtm(r100, x, point, title="directed 30, precice")
    x = nwa.precice_service_area(point, breaks=125, dissolve=False)
    assert len(x) == 7
    if __name__ == "__main__":
        sg.qtm(r100, x, point, title="directed 125, precice")

    nwa.rules.split_lines = True
    x = nwa.precice_service_area(point, breaks=125, dissolve=False)
    if __name__ == "__main__":
        sg.qtm(
            r100,
            x,
            point,
            title="directed 125, precice,\n split_lines=True",
        )
    assert len(x) == 7, x


def not_test_service_area(points_oslo, roads_oslo):
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    warnings.filterwarnings(action="ignore", category=UserWarning)
    pd.options.mode.chained_assignment = None

    p = points_oslo
    p = sg.clean_clip(p, p.geometry.iloc[0].buffer(1000))
    p["idx"] = p.index
    p["idx2"] = p.index

    r = roads_oslo
    r = sg.clean_clip(r, p.geometry.loc[0].buffer(2200))

    rules = sg.NetworkAnalysisRules(directed=True, weight="meters")

    nw = sg.get_connected_components(r).query("connected == 1")
    nwa = sg.NetworkAnalysis(nw, rules=rules)

    precice = nwa.service_area(p, breaks=(100, 300, 500))
    not_precice = nwa.service_area(p, breaks=(100, 300, 500))
    print(precice.length.sum(), not_precice.length.sum())

    def _precice():
        nwa.service_area(p, breaks=(100, 300, 500))

    def _not_precice():
        nwa.service_area(p, breaks=(100, 300, 500))

    print("_not_precice", timeit.timeit(lambda: _not_precice(), number=1))
    print("_precice", timeit.timeit(lambda: _precice(), number=1))
    print("_not_precice", timeit.timeit(lambda: _not_precice(), number=1))
    print("_precice", timeit.timeit(lambda: _precice(), number=1))

    def _precice():
        nwa.service_area(p, breaks=(1000, 3000, 5000))

    def _not_precice():
        nwa.service_area(p, breaks=(1000, 3000, 5000))

    print("_not_precice", timeit.timeit(lambda: _not_precice(), number=1))
    print("_precice", timeit.timeit(lambda: _precice(), number=1))
    print("_not_precice", timeit.timeit(lambda: _not_precice(), number=1))
    print("_precice", timeit.timeit(lambda: _precice(), number=1))


def main():
    from oslo import points_oslo, roads_oslo

    roads_oslo = roads_oslo()
    points_oslo = points_oslo()

    test_service_area(points_oslo, roads_oslo)
    not_test_service_area(points_oslo, roads_oslo)


if __name__ == "__main__":
    import cProfile

    # cProfile.run("main()", sort="cumtime")
    main()
