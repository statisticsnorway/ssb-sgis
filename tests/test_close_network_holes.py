# %%
import warnings
from pathlib import Path
from time import perf_counter

import geopandas as gpd

import sgis as sg


def test_close_network_holes(roads_oslo, points_oslo):
    warnings.filterwarnings(action="ignore", category=UserWarning)
    warnings.filterwarnings(action="ignore", category=FutureWarning)

    r = roads_oslo
    p = points_oslo

    p = p.iloc[[0]]

    r = sg.clean_clip(r, p.buffer(600))

    nw = sg.Network(r)

    nw = nw.get_largest_component()
    len_now = len(nw.gdf)

    for meters in [1.1, 3, 10]:
        _time = perf_counter()
        nw = nw.close_network_holes(meters, fillna=0, deadends_only=False)
        print("n", sum(nw.gdf.hole == 1))
        print("time close_network_holes, all roads: ", perf_counter() - _time)

        _time = perf_counter()
        nw = nw.close_network_holes(meters, fillna=0, deadends_only=True)
        print("n", sum(nw.gdf.hole == 1))
        print("time close_network_holes, deadends_only: ", perf_counter() - _time)

    nw = nw.get_largest_component()
    assert len(nw.gdf) != len_now


def main():
    from oslo import points_oslo, roads_oslo

    test_close_network_holes(roads_oslo(), points_oslo())


if __name__ == "__main__":
    main()
