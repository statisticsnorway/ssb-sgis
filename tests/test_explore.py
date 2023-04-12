# %%
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


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
    donut = p.assign(geometry=lambda x: x.buffer(150).difference(x.buffer(50)))
    lines = roads.clip(donut)
    roads["geometry"] = roads.buffer(3)

    r300 = roads.clip(p.buffer(300))
    r200 = roads.clip(p.buffer(200))
    r100 = roads.clip(p.buffer(100))

    print(
        "when 1 gdf and categorical column, the gdf should be split into categories"
        " that can be toggled on/off:"
    )
    r300["category"] = np.random.choice([*"abc"], len(r300))
    sg.explore(r300, "category")

    print("when multiple gdfs and no column, should be one color per gdf:")
    sg.explore(r300, r200, r100)
    print("when numeric column, should be same color scheme:")
    sg.explore(r300, r200, r100, "meters")

    for explore in [1, 0]:
        sg.samplemap(
            r300,
            r200,
            r100,
            "length",
            labels=("r30000", "r20000", "r10000"),
            cmap="plasma",
            explore=explore,
            size=100,
        )

    sg.clipmap(r300, r200, r100, "meters", mask=p.buffer(100), explore=True)
    for explore in [1, 0]:
        sg.clipmap(
            r300,
            r200,
            r100,
            "area",
            cmap="inferno",
            mask=p.buffer(100),
            explore=explore,
        )

    for sample_from_first in [1, 0]:
        sg.samplemap(r300, roads_oslo, sample_from_first=sample_from_first, size=50)
    monopoly = sg.to_gdf(r300.unary_union.convex_hull, crs=r300.crs)

    for _ in range(5):
        sg.samplemap(
            monopoly,
            r300,
            roads_oslo,
            size=30,
        )

    sg.clipmap(r300, r200, r100, "meters", mask=p.buffer(100))

    sg.samplemap(
        r300, r200, r100, "meters", labels=("r30000", "r20000", "r10000"), cmap="plasma"
    )

    sg.explore(roads, points, "meters")

    roads_mcat = roads.assign(
        meters_cat=lambda x: (x.length / 50).astype(int).astype(str)
    )
    points_mcat = points.assign(
        meters_cat=lambda x: (x.length / 50).astype(int).astype(str)
    )

    sg.explore(roads_mcat, points_mcat, "meters_cat")
    sg.qtm(roads_mcat, points_mcat, "meters_cat")

    # createing a geometry collection
    r100 = pd.concat([r100, lines], ignore_index=True).dissolve()
    sg.explore(r300, r200, r100, "meters")


def main():
    from oslo import points_oslo, roads_oslo

    not_test_explore(points_oslo(), roads_oslo())


if __name__ == "__main__":
    import cProfile

    main()
    # cProfile.run("main()", sort="cumtime")
