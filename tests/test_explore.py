# %%
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

src = str(Path(__file__).parent).strip("tests") + "src"
testdata = str(Path(__file__).parent.parent) + "/tests/testdata/raster"

import sys

sys.path.insert(0, src)

import sgis as sg

path_sentinel = testdata + "/sentinel2"


def test_torch():

    torch_dataset = sg.torchgeo.Sentinel2(path_sentinel, res=10)

    sg.explore(torch_dataset, "value")


def not_test_center(r300, r200, r100, p):
    for center in [
        (263206.184457095, 6651199.528012605),
        "point (263206.184457095 6651199.528012605)",
        {"geometry": [(263206.184457095, 6651199.528012605)]},
        sg.to_gdf("point (263206.184457095 6651199.528012605)", crs=r300.crs),
        sg.to_gdf("point (263206.184457095 6651199.528012605)", crs=r300.crs).buffer(
            100
        ),
    ]:
        sg.explore(
            r300,
            r200,
            r100,
            "length",
            cmap="plasma",
            center=center,
            size=100,
            show_in_browser=False,
        )


def test_explore(points_oslo, roads_oslo):
    roads = roads_oslo.copy()
    points = points_oslo.copy()

    p = points.iloc[[0]]
    roads = roads[["geometry"]]
    roads["km"] = roads.length / 1000
    roads["cat"] = np.random.choice([*"abc"], len(roads))
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

    sg.explore(r300, "meters", r100, bygdoy=7000)

    sg.explore(r300, r100, center_4326=(10.75966535, 59.92945927, 1000))
    sg.explore(r300, r100, center=(10.75966535, 59.92945927, 1000), crs=4326)

    sg.clipmap(r300, r200, "meters", show_in_browser=False)
    sg.explore(r300, r200, bygdoy=1, size=10_000, show_in_browser=False)
    not_test_center(r300, r200, r100, p)

    sg.explore(
        r300,
        {"r200": r200, "r100": r100},
        bygdoy=1,
        size=10_000,
        show_in_browser=False,
        show=False,
    )

    sg.explore(
        r300,
        **{"r200": r200, "r100": r100},
        bygdoy=1,
        size=10_000,
        show_in_browser=False,
        show=False,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        sg.explore(
            **{"r200": r200, "r100": r100},
            show_in_browser=False,
        )

    print("when multiple gdfs and no column, should be one color per gdf:")
    sg.explore(r300, r200, r100, show_in_browser=False)
    print("when numeric column, should be same color scheme:")
    sg.explore(r300, r200, r100, "meters", scheme="quantiles", show_in_browser=False)
    sg.explore(*(r300, r200, r100), "meters", scheme="quantiles", show_in_browser=False)

    sg.clipmap(r300, r200, r100, "meters", mask=p.buffer(100), show_in_browser=False)
    sg.clipmap(
        r300,
        r200,
        r100,
        "area",
        cmap="inferno",
        mask=p.buffer(100),
        show_in_browser=False,
    )

    for sample_from_first in [1, 0]:
        sg.samplemap(
            r300,
            roads_oslo,
            sample_from_first=sample_from_first,
            size=50,
            show_in_browser=False,
        )
    monopoly = sg.to_gdf(r300.unary_union.convex_hull, crs=r300.crs)

    for _ in range(5):
        sg.samplemap(
            monopoly,
            r300,
            roads_oslo,
            size=30,
            show_in_browser=False,
        )

    sg.clipmap(r300, r200, r100, "meters", mask=p.buffer(100), show_in_browser=False)

    sg.samplemap(
        r300,
        r200,
        r100,
        "meters",
        labels=("r30000", "r20000", "r10000"),
        cmap="plasma",
        show_in_browser=False,
    )

    sg.explore(roads, points, "meters", show_in_browser=False)

    roads_mcat = roads.assign(
        meters_cat=lambda x: (x.length / 50).astype(int).astype(str)
    )
    points_mcat = points.assign(
        meters_cat=lambda x: (x.length / 50).astype(int).astype(str)
    )

    sg.explore(roads_mcat, points_mcat, "meters_cat", show_in_browser=False)
    sg.qtm(roads_mcat, points_mcat, "meters_cat")

    print("creating a geometry collection")
    r100 = pd.concat([r100, lines], ignore_index=True).dissolve()
    sg.explore(r300, r200, r100, "meters", show_in_browser=False)

    print("only one unique value per gdf")
    r300["col"] = 30323.32032
    r200["col"] = 232323.32032
    r100["col"] = 12243433.3223
    sg.explore(r300, r200, r100, "col", show_in_browser=False)


def not_test_explore(points_oslo, roads_oslo):
    roads = roads_oslo.copy()
    points = points_oslo.copy()

    p = points.iloc[[0]]
    roads = roads[["geometry"]]
    roads["km"] = roads.length / 1000
    roads["cat"] = np.random.choice([*"abc"], len(roads))
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

    sg.explore(points_oslo, center="akersveien 26")


def main():
    from oslo import points_oslo
    from oslo import roads_oslo

    test_explore(points_oslo(), roads_oslo())
    not_test_explore(points_oslo(), roads_oslo())


if __name__ == "__main__":

    main()
    # cProfile.run("main()", sort="cumtime")
