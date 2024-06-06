# %%
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

src = str(Path(__file__).parent).replace("tests", "") + "src"
testdata = str(Path(__file__).parent.parent) + "/tests/testdata/raster"

import sys

sys.path.insert(0, src)


import sgis as sg

path_sentinel = testdata + "/sentinel2"

# np.set_printoptions(linewidth=400)

if 0:
    for p, bbox in zip(
        [
            # "S2A_MSIL2A_20230624T104621_N0509_R051_T32VPM_20230624T170454.SAFE",
            "S2B_MSIL2A_20170826T104019_N0208_R008_T32VNM_20221207T150454.SAFE",
            "S2B_MSIL2A_20230606T103629_N0509_R008_T32VNM_20230606T121204.SAFE",
        ],
        [
            # (sg.to_gdf([11.0771105, 59.9483914], crs=4326).to_crs(25833).buffer(1500)),
            (
                sg.to_gdf([10.28173443, 60.16616654], crs=4326)
                .to_crs(25833)
                .buffer(1500)
            ),
            (
                sg.to_gdf([10.28173443, 60.16616654], crs=4326)
                .to_crs(25833)
                .buffer(1500)
            ),
        ],
        strict=False,
    ):
        paths = sg.helpers.get_all_files(
            f"C:/Users/ort/OneDrive - Statistisk sentralbyrå/data/sentinel2/{p}"
        )
        img = sg.Sentinel2Image(
            f"C:/Users/ort/OneDrive - Statistisk sentralbyrå/data/sentinel2/{p}",
        )
        for path in paths:

            if "tif" not in path:
                continue
            raster = sg.Raster.from_path(path)
            try:
                centroid
            except NameError:
                centroid = raster.centroid
            out = (
                Path(r"C:\Users\ort\git\ssb-sgis\tests\testdata\raster\sentinel2")
                / f"{p}"
            ) / (Path(raster.name).stem + "_clipped.tif")
            print(path)
            print(out)
            import os

            os.makedirs(out.parent, exist_ok=True)

            print(bbox)
            raster = raster.load().clip(
                bbox,  # boundless=False, masked=False
            )
            print(raster.bounds)
            print(raster.values)
            print(raster.values.shape)
            print(np.sum(np.where(raster.values != 0, 1, 0)))

            # sg.explore(raster.to_gdf())
            raster.write(out)
            print(raster.shape)
            # print(raster)
            # sg.explore(raster.to_gdf(), "value")
if 0:
    SENTINEL2_FILENAME_REGEX = r"""
        ^(?P<tile>T\d{2}[A-Z]{3})
        _(?P<date>\d{8}T\d{6})
        _(?P<band>B[018][\dA])
        (?:_(?P<resolution>\d+)m)?
        .*
        \..*$
    """
    path_sentinel = r"C:\Users\ort\git\ssb-sgis\tests\testdata\raster\sentinel2\S2A_MSIL2A_20230601T104021_N0509_R008_T32VNM_20230601T215503.SAFE"

    cube = sg.DataCube.from_root(
        path_sentinel, res=10, filename_regex=SENTINEL2_FILENAME_REGEX
    )

    band2 = cube["B02"].load(indexes=1)
    band3 = cube["B03"].load(indexes=1)
    band4 = cube["B04"].load(indexes=1)

    sg.torchgeo.Sentinel2.filename_regex = SENTINEL2_FILENAME_REGEX
    ds = sg.torchgeo.Sentinel2(path_sentinel)
    ds.plot(ds[ds.bounds])

    print(band2.values.shape)
    print(band3.values.shape)
    print(band4.values.shape)
    # sg.explore(band2.to_gdf(), "value")

    arr = np.array([band2.values, band3.values, band4.values])
    arr = arr.reshape(arr.shape[1], arr.shape[2], arr.shape[0])

    print(arr.shape)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    import torch

    image = torch.clamp(torch.tensor(arr) / 10000, min=0, max=1)

    print(image)

    ax.imshow(image)
    ax.axis("off")

    # Use bilinear interpolation (or it is displayed as bicubic by default).
    # plt.imshow(arr, interpolation="nearest")
    plt.show()

    import folium

    folium.raster_layers.ImageOverlay(arr, bounds=raster.bounds)


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

    # sg.explore(r300, r100, center_4326=(10.75966535, 59.92945927, 1000))
    sg.explore(r300, r100, center=(10.75966535, 59.92945927, 1000))
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

    test_image_collection()
    # test_torch()
    test_explore(points_oslo(), roads_oslo())
    # not_test_explore(points_oslo(), roads_oslo())


if __name__ == "__main__":

    main()
    # cProfile.run("main()", sort="cumtime")
