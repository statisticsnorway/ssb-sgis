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
    roads["km"] = roads.length / 1000
    points["meters"] = points.length
    points["km"] = points.length / 1000
    roads = roads.sjoin(p.buffer(500).to_frame()).drop("index_right", axis=1)
    points = points.sjoin(p.buffer(500).to_frame())
    points["geometry"] = points.buffer(8)
    roads["geometry"] = roads.buffer(3)
    roads.name = "roads"
    points.name = "points"
    r1 = roads.sjoin(p.buffer(300).to_frame())
    r2 = roads.sjoin(p.buffer(200).to_frame())
    r3 = roads.sjoin(p.buffer(100).to_frame())

    gs.clipmap(r1, r2, r3, p.buffer(100), "meters", ("r100", "r200", "r300"))
    gs.samplemap(r1, r2, r3, "meters", ("r100", "r200", "r300"))

    print("static mapping finished")

    gs.explore(roads, points, "meters", ("roads", "points"), cmap="plasma")

    x = gs.Explore(roads, points, p, "meters", ("roads", "points", "p"))
    assert not x._is_categorical
    x = gs.Explore(roads, points)
    assert x._is_categorical
    x.explore("meters")
    x.clipmap(p.buffer(100))
    x.samplemap()

    x = gs.Explore(r1, r2, r3, column="meters", labels=("r100", "r200", "r300"))
    x.clipmap(p.buffer(100))
    x.samplemap()
    r3.loc[0, "meters"] = None
    x = gs.Explore(r1, r2, r3, labels=("r100", "r200", "r300"))
    x.explore()
    x.explore("meters", cmap="inferno")
    x.samplemap(cmap="magma")
    x.samplemap(100, "km")
    x.clipmap(p.buffer(100), cmap="RdPu")
    x.clipmap(p.buffer(100), "meters")


def main():
    not_test_explore()


if __name__ == "__main__":
    main()
