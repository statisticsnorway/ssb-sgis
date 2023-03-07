# %%
from pathlib import Path

import geopandas as gpd


src = str(Path(__file__).parent).strip("tests") + "src"

import sys


sys.path.insert(0, src)

import gis_utils as gs


def not_test_explore():
    roads = gpd.read_parquet(gs.roadpath)
    points = gpd.read_parquet(gs.pointpath)

    p = points.iloc[[0]]
    roads = roads[["geometry"]]
    roads["meters"] = roads.length
    # roads["meters"] = [str(int(x / 30)) for x in roads["meters"]]
    roads = roads.sjoin(p.buffer(500).to_frame()).drop("index_right", axis=1)
    points = points.sjoin(p.buffer(500).to_frame())
    points["geometry"] = points.buffer(8)
    roads["geometry"] = roads.buffer(3)
    # points["meters"] = ""
    roads.name = "roads"
    points.name = "points"
    x = gs.Explore(roads, points, p, ("roads", "points", "p"))
    x = gs.Explore(roads, points)
    assert x._is_categorical
    x.explore()
    x.clipmap(p.buffer(100))
    x.samplemap()

    r1 = roads.sjoin(p.buffer(100).to_frame())
    r2 = roads.sjoin(p.buffer(200).to_frame())
    r3 = roads.sjoin(p.buffer(300).to_frame())
    x = gs.Explore(r1, r2, r3, "meters")
    x.explore()
    x.clipmap(p.buffer(100))
    x.samplemap()

    gs.explore(roads, points, "meters")
    gs.clipmap(roads, p.buffer(100), column="meters")
    gs.samplemap(roads, column="meters")


def main():
    not_test_explore()


if __name__ == "__main__":
    main()
