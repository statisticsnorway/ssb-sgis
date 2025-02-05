# %%
import datetime
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import union_all

src = str(Path(__file__).parent).replace("tests", "") + "src"
testdata = str(Path(__file__).parent.parent) + "/tests/testdata/raster"

import sys

sys.path.insert(0, src)


import sgis as sg

path_sentinel = testdata + "/sentinel2"


def test_image_collection():

    collection = sg.Sentinel2Collection(path_sentinel, res=10, level="L2A")
    e = sg.Explore(collection)
    e.explore()

    # selecting one random image
    img = collection[0]  # ["B04"]
    msk = sg.to_gdf(img.bounds, crs=img.crs).centroid.buffer(150)

    for image in collection:
        e = sg.Explore(image)
        e.explore()
        assert len(e.rasters), e.rasters
        for band in image:
            e = sg.Explore(band)
            e.explore()
            assert len(e.rasters), e.rasters

    sg.explore(collection.sample_images(2))

    sg.explore(collection.sample_images(2))

    sg.explore(collection.sample_tiles(1))
    sg.explore(collection)
    sg.explore(collection, mask=msk)

    e = sg.Explore(msk, collection)

    e.explore()
    assert len(e.rasters), e.rasters

    e = sg.Explore(collection)
    e.explore()
    assert len(e.rasters), e.rasters


def inner_test_center(r300, r200, r100, p):

    explorer = sg.explore(
        sg.to_gdf([10.51125371, 0], 25833),
        center=(6.02240883, 62.47416834, 300),
    )
    assert not len(explorer), len(explorer)

    explorer = sg.explore(
        sg.to_gdf("POINT (306.022 62.474)", 25833),
        center=(6.02240883, 62.47416834, 300),
    )

    assert not len(explorer), len(explorer)

    for center in [
        (263206.184457095, 6651199.528012605, 100),
        # (6651199.528012605, 263206.184457095, 100),
        (10.76100918, 59.92997799, 100),
        (59.92997799, 10.76100918, 100),
        "point (263206.184457095 6651199.528012605)",
        {"geometry": [(263206.184457095, 6651199.528012605)]},
        sg.to_gdf("point (263206.184457095 6651199.528012605)", crs=r300.crs),
        sg.to_gdf("point (263206.184457095 6651199.528012605)", crs=r300.crs).buffer(
            100
        ),
    ]:
        explorer = sg.explore(
            r300,
            r200,
            r100,
            "length",
            cmap="plasma",
            center=center,
            size=100,
            show_in_browser=False,
        )
        assert explorer, (explorer, center)
        assert len(explorer) == 3, (len(explorer), explorer)


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

    e = sg.explore(
        r300,
        "meters",
        r100,
        bygdoy=7000,
        wms=sg.NorgeIBilderWms(
            years=[2022, 2023, 2024], not_contains="sentinel", show=-1
        ),
    )
    assert isinstance(next(iter(e.wms)), sg.NorgeIBilderWms)
    e = sg.explore(
        r300,
        "meters",
        r100,
        bygdoy=7000,
        wms=sg.NorgeIBilderWms(
            years=[2022, 2023, 2024], not_contains="sentinel", show=0
        ),
    )
    assert isinstance(next(iter(e.wms)), sg.NorgeIBilderWms)
    e = sg.explore(
        r300,
        "meters",
        r100,
        bygdoy=7000,
        wms=sg.NorgeIBilderWms(
            years=[2022, 2023, 2024], not_contains="sentinel", show=[1, 2]
        ),
    )
    assert isinstance(next(iter(e.wms)), sg.NorgeIBilderWms)

    inner_test_center(r300, r200, r100, p)

    sg.explore(r300, "meters", r100, bygdoy=7000)

    # sg.explore(r300, r100, center_4326=(10.75966535, 59.92945927, 1000))
    sg.explore(r300, r100, center=(10.75966535, 59.92945927, 1000))
    sg.explore(r300, r100, center=(10.75966535, 59.92945927, 1000), crs=4326)

    sg.clipmap(r300, r200, "meters", show_in_browser=False)
    sg.explore(r300, r200, bygdoy=1, size=10_000, show_in_browser=False)

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
    monopoly = sg.to_gdf(union_all(r300.geometry.values).convex_hull, crs=r300.crs)

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


def not_test_wms_json():
    print(
        "IMPORTANT: if you run this function, make sure to change the global variable JSON_YEARS in wms.py"
    )
    wms = sg.NorgeIBilderWms(years=range(1999, datetime.datetime.now().year + 1))
    wms.load_tiles()
    try:
        os.remove(sg.maps.norge_i_bilder_wms.JSON_PATH)
    except FileNotFoundError:
        pass
    with open(sg.maps.norge_i_bilder_wms.JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(
            [
                {
                    key: value if key != "bbox" else value.wkt
                    for key, value in tile.items()
                }
                for tile in wms.tiles
            ],
            file,
            ensure_ascii=False,
        )


def main():

    from oslo import points_oslo
    from oslo import roads_oslo

    # not_test_wms_json()

    test_explore(points_oslo(), roads_oslo())
    test_image_collection()
    # not_test_explore(points_oslo(), roads_oslo())


if __name__ == "__main__":

    main()
    import cProfile

    cProfile.run("main()", sort="cumtime")
