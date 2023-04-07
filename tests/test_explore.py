# %%
from pathlib import Path

import geopandas as gpd
import numpy as np


src = str(Path(__file__).parent).strip("tests") + "src"

import sys


sys.path.insert(0, src)

import sgis as sg


def not_test_explore(points_oslo, roads_oslo):
    roads = roads_oslo
    points = points_oslo

    p = points.iloc[[0]]
    roads = roads[["geometry"]]
    roads["km"] = roads.length / 1000
    points["km"] = points.length / 1000
    roads = roads.sjoin(p.buffer(500).to_frame()).drop("index_right", axis=1)
    points = points.sjoin(p.buffer(500).to_frame())
    points["geometry"] = points.buffer(8)
    roads["geometry"] = roads.buffer(3)

    r1 = roads.clip(p.buffer(300))
    r2 = roads.clip(p.buffer(200))
    r3 = roads.clip(p.buffer(100))

    sg.explore(r1, r2, r3)
    sg.explore(r1, r2, r3, "meters")

    for yesno in [1, 0]:
        sg.samplemap(
            r1,
            r2,
            r3,
            "length",
            labels=("r100", "r200", "r300"),
            cmap="plasma",
            explore=yesno,
            size=100,
        )

    sg.clipmap(r1, r2, r3, "meters", mask=p.buffer(100), explore=True)
    for yesno in [1, 0]:
        sg.clipmap(
            r1, r2, r3, "area", cmap="inferno", mask=p.buffer(100), explore=yesno
        )

    for yesno in [1, 0]:
        sg.samplemap(r1, roads_oslo, sample_from_first=yesno, size=50)
    monopoly = sg.to_gdf(r1.unary_union.convex_hull, crs=r1.crs)

    for _ in range(5):
        sg.samplemap(
            monopoly,
            r1,
            roads_oslo,
            size=30,
        )

    sg.clipmap(r1, r2, r3, "meters", mask=p.buffer(100))

    sg.samplemap(r1, r2, r3, "meters", labels=("r100", "r200", "r300"), cmap="plasma")

    sg.explore(roads, points, "meters")

    roads_mcat = roads.assign(
        meters_cat=lambda x: (x.length / 50).astype(int).astype(str)
    )
    points_mcat = points.assign(
        meters_cat=lambda x: (x.length / 50).astype(int).astype(str)
    )

    sg.explore(roads_mcat, points_mcat, "meters_cat")
    sg.qtm(roads_mcat, points_mcat, "meters_cat")


def main():
    from oslo import points_oslo, roads_oslo

    not_test_explore(points_oslo(), roads_oslo())


if __name__ == "__main__":
    import cProfile

    main()
    # cProfile.run("main()", sort="cumtime")
