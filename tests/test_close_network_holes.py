# %%
import warnings
from pathlib import Path
from time import perf_counter

import geopandas as gpd

import gis_utils as gs


def test_close_network_holes():
    warnings.filterwarnings(action="ignore", category=UserWarning)
    warnings.filterwarnings(action="ignore", category=FutureWarning)

    p = gpd.read_parquet(Path(__file__).parent / "testdata" / "random_points.parquet")
    p = p.iloc[[0]]

    r = gpd.read_parquet(Path(__file__).parent / "testdata" / "roads_oslo_2022.parquet")
    r = gs.clean_clip(r, p.buffer(600))

    nw = gs.Network(r)

    nw = nw.get_largest_component()
    len_now = len(nw.gdf)

    for meters in [1.1, 3, 10]:
        _time = perf_counter()
        nw = nw.close_network_holes(meters, deadends_only=False)
        print("n", sum(nw.gdf.hole == 1))
        print("time close_network_holes, all roads: ", perf_counter() - _time)

        _time = perf_counter()
        nw = nw.close_network_holes(meters, deadends_only=True)
        print("n", sum(nw.gdf.hole == 1))
        print("time close_network_holes, deadends_only: ", perf_counter() - _time)

    nw = nw.get_largest_component()
    assert len(nw.gdf) != len_now


def main():
    test_close_network_holes()


if __name__ == "__main__":
    main()
