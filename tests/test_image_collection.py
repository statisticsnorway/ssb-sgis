# %%


from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from geopandas import GeoSeries
from lightning.pytorch import Trainer
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon
from torch.utils.data import DataLoader
from torchgeo.datamodules import InriaAerialImageLabelingDataModule
from torchgeo.datasets import stack_samples
from torchgeo.datasets.utils import BoundingBox
from torchgeo.samplers import RandomBatchGeoSampler
from torchgeo.samplers import RandomGeoSampler
from torchgeo.trainers import SemanticSegmentationTask

src = str(Path(__file__).parent).replace("tests", "") + "src"
testdata = str(Path(__file__).parent.parent) + "/tests/testdata/raster"

import sys

sys.path.insert(0, src)


import sgis as sg

path_sentinel = testdata + "/sentinel2"
path_singleband = testdata + "/dtm_10.tif"
path_two_bands = testdata + "/dtm_10_two_bands.tif"


def test_zonal():
    r = sg.Band(path_singleband, indexes=1, res=None).load()
    gdf = sg.make_grid(r.bounds, 100, crs=r.crs)

    sg.explore(sg.to_gdf(r.bounds, r.crs), gdf)

    gdf.index = [np.random.choice([*"abc"]) for _ in range(len(gdf))]

    zonal_stats = r.zonal(gdf, aggfunc=[sum, np.mean, "median"], array_func=None)
    print(zonal_stats)
    print(gdf)


def test_gradient():
    arr = np.array(
        [
            [100, 100, 100, 100, 100],
            [100, 110, 110, 110, 100],
            [100, 110, 120, 110, 100],
            [100, 110, 110, 110, 100],
            [100, 100, 100, 100, 100],
        ]
    )
    print(arr)

    # creating a simple Band with a resolution of 10 (50 / width or height).
    band = sg.Band(arr, crs=None, bounds=(0, 0, 50, 50), res=10)
    gradient = band.get_gradient(copy=True)
    assert np.max(gradient.values) == 1, gradient.values

    degrees = band.get_gradient(degrees=True, copy=True)

    assert np.max(degrees.values) == 45, np.max(degrees.values)

    # r = sg.Band(path_singleband, indexes=1, res=None).load(nodata=0)
    # assert int(np.min(r.values)) == 0, np.min(r.values)
    # degrees = r.get_gradient(degrees=True, copy=True)
    # assert int(np.max(degrees.values)) == 75, np.max(degrees.values)
    # gradient = r.get_gradient(copy=True)
    # assert int(np.max(gradient.values)) == 3, np.max(gradient.values)

    # r = sg.Band(path_two_bands, indexes=1).load(nodata=0)
    # assert r.shape == (101, 101), r.shape
    # gradient = r.get_gradient(copy=True)
    # assert int(np.max(gradient.values)) == 3, np.max(gradient.values)

    # degrees = r.get_gradient(degrees=True, copy=True)
    # assert int(np.max(degrees.values)) == 75, np.max(degrees.values)

    # r = sg.Band(path_two_bands, indexes=(1, 2))
    # assert r.shape == (2, 101, 101), r.shape

    # degrees = r.load().get_gradient(degrees=True)

    # assert int(np.nanmax(degrees.values)) == 75, int(np.nanmax(degrees.values))
    # assert len(degrees.shape) == 3
    # gdf = degrees.to_gdf()
    # if __name__ == "__main__":
    #     sg.explore(gdf[gdf["indexes"] == 1], "value")
    #     sg.explore(gdf[gdf["indexes"] == 2], "value")

    # max_ = int(np.nanmax(r.values))
    # gradient = r.get_gradient(copy=True)
    # gradient.plot()
    # assert max_ == int(np.nanmax(r.values))
    # assert int(np.nanmax(gradient.values)) == 6, int(np.nanmax(gradient.values))


def test_with_mosaic():
    collection = sg.Sentinel2Collection(path_sentinel, level=None, res=10, processes=2)
    assert len(collection) == 3, collection
    assert list(collection.dates) == list(sorted(collection.dates)), collection.dates

    mosaic = sg.Sentinel2CloudlessCollection(
        path_sentinel, level=None, res=10, processes=2
    )
    sg.explore(mosaic)
    assert len(mosaic) == 1, mosaic

    concated = sg.concat_image_collections([mosaic, collection])
    assert len(concated) == 4, concated

    concated = mosaic | collection
    assert len(concated) == 4, concated


def test_concat_image_collections():
    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10, processes=2)

    new_collection = sg.concat_image_collections(
        [collection[[i]] for i in range(len(collection))]
    )
    assert len(new_collection) == len(collection)
    for k, v in new_collection.__dict__.items():
        if "path" in k:
            continue
        if k in ["_df", "_images", "_all_filepaths"]:
            continue
        assert v == getattr(
            collection, k
        ), f"different value for '{k}': {v} and {getattr(collection, k)}"


def x():
    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)

    sg.explore(collection)

    merged = collection.groupby(["tile", "date"]).merge_by_band()

    sg.explore(merged)

    ndvi = merged.get_ndvi()

    sg.explore(ndvi)


def test_ndvi_and_explore():
    """Running ndvi and checking how it's plotted with explore."""
    n = 3000

    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)

    e = sg.explore(collection)
    assert e.rasters
    # assert (x := [x["label"] for x in e.raster_data]) == [
    #     f"{img.tile}_{img.date[:8]}" for img in collection
    # ], x

    collection = collection.filter(bands=collection.ndvi_bands)

    for (tile_id,), tile_collection in collection.groupby("tile"):

        for img in tile_collection:
            assert img.tile == tile_id
            ndvi = img.get_ndvi()

            assert (ndvi.cmap) == "Greens"

            e = sg.explore(ndvi)
            assert e.rasters
            assert (x := list(sorted([x["label"] for x in e.raster_data]))) == [
                "ndvi",
            ], x

            e = sg.explore(ndvi.get_n_largest(n), ndvi.get_n_smallest(n), img)
            assert e.rasters
            assert e["labels"] == [
                f"largest_{n}",
                f"smallest_{n}",
            ], e["labels"]

            # assert (x := list(sorted([x["label"] for x in e.raster_data]))) == [
            #     f"{img.tile}_{img.date[:8]}"
            # ], x

            new_img = sg.Image([ndvi], res=10)

            e = sg.explore(new_img)
            assert e.rasters
            assert (x := list(sorted([x["label"] for x in e.raster_data]))) == [
                "new_img",
            ], x

            e = sg.explore(sg.Image([ndvi], res=10))
            assert e.rasters
            # cannot find an object name since the image is created within the explore constructor
            assert (x := list(sorted([x["label"] for x in e.raster_data]))) == [
                "Image(0)",
            ], x


def test_bbox():
    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10, processes=2)

    no_imgs = collection.filter(bounds=Point(0, 0))  # bounds=Point(0, 0))
    assert not len(no_imgs), no_imgs

    centroid = collection[0].centroid
    imgs = collection.filter(bounds=centroid)  # bounds=centroid)
    assert len(imgs) == 2, imgs

    centroid = collection[1].centroid
    imgs = collection.filter(bounds=centroid)  # bounds=centroid)
    assert len(imgs) == 2, imgs

    centroid = collection[2].centroid
    imgs = collection.filter(bounds=centroid)  # bounds=centroid)
    assert len(imgs) == 1, imgs


def test_sample():
    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)

    size = 200

    # low buffer resolution means the area won't be exactly this
    circle_area_should_be = int((size**2) * 3.14159265359)

    e = sg.explore(collection.sample(1, size=size))
    assert (x := e.raster_data[0]["arr"].shape) == (40, 40, 3), x

    bbox = sg.to_gdf(collection.unary_union, collection.crs)

    e = sg.samplemap(collection, bbox, size=size)
    assert len(e.raster_data) in [1, 2], e.raster_data

    assert (x := e.raster_data[0]["arr"].shape) == (40, 40, 3), x
    assert (x := int(e._gdfs[0].dissolve().area.sum())) <= circle_area_should_be, (
        x,
        circle_area_should_be,
    )

    print("as gdfs")
    e = sg.explore(
        collection.sample(1, size=size).load_bands().to_gdfs(),
        column="value",
        return_explorer=True,
    )
    square_area_should_be = int(size * 2 * size * 2)
    assert (x := int(e._gdfs[0].dissolve().area.sum())) == square_area_should_be, (
        x,
        square_area_should_be,
    )

    sample = collection.sample(15, size=size)
    print("as images")
    e = sg.explore(sample)

    assert len(sample) >= 15, len(sample)
    for img in sample:
        e = sg.explore(img)
        assert e.rasters
        for band in img:
            arr = band.load().values
            assert arr.shape <= (40, 40), arr.shape

    sample = collection.sample_images(2)
    assert len(sample) == 2, sample

    sample = collection.sample_images(1)
    assert len(sample) == 1, sample

    sample = collection.sample_tiles(2)
    assert isinstance(sample, sg.Sentinel2Collection), type(sample)
    assert len(sample) == 3
    assert isinstance(sample[0], sg.Image)
    assert isinstance(sample[1], sg.Image)
    assert isinstance(sample[2], sg.Image)
    assert sample[0].date.startswith("2017")
    assert sample[1].date.startswith("2023")
    assert sample[2].date.startswith("2023")


def test_indexing():
    wrong_collection = sg.Sentinel2Collection(path_sentinel, level="L1C", res=10)
    assert not len(wrong_collection), len(wrong_collection)
    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)

    img = collection[0]
    assert isinstance(img, sg.Image)
    assert img.date.startswith("2017"), img.date
    assert "B02" in img
    assert "B00" not in img
    assert ["B02", "B03", "B04"] in img

    img = collection[1]
    assert isinstance(img, sg.Image)
    assert img.date.startswith("2023"), img.date

    img = collection[2]
    assert isinstance(img, sg.Image)
    assert img.date.startswith("2023"), img.date

    band = collection[0]["B02"]
    assert isinstance(band, sg.Band)
    assert band.date.startswith("2017"), band.date

    arr = collection[0]["B02"].load().values
    assert isinstance(arr, np.ndarray)

    s2a = collection[["s2a" in img.name.lower() for img in collection]]
    assert isinstance(s2a, sg.ImageCollection)
    assert len(s2a) == 1, s2a

    s2b = collection[["s2b" in img.name.lower() for img in collection]]
    assert isinstance(s2b, sg.ImageCollection)
    assert len(s2b) == 2, s2b

    assert isinstance((x := collection[[0, -1]]), sg.ImageCollection), x
    assert len(x := collection[[0, -1]]) == 2, x
    assert isinstance((x := collection[0][["B02", "B03"]]), sg.Image), x
    assert isinstance((x := collection[0][["B02"]]), sg.Image), x
    assert isinstance((x := collection[0]["B02"]), sg.Band), x

    assert isinstance(collection[0]["B02"].load().values, np.ndarray)

    bounds = GeoSeries([collection[0].unary_union]).bounds
    torchgeo_bbox = BoundingBox(
        minx=bounds.minx[0],
        miny=bounds.miny[0],
        maxx=bounds.maxx[0],
        maxy=bounds.maxy[0],
        mint=collection.mint,
        maxt=collection.maxt,
    )
    torchgeo_dict = collection[torchgeo_bbox]
    assert isinstance(torchgeo_dict, dict), torchgeo_dict
    assert isinstance(torchgeo_dict["image"], torch.Tensor), torchgeo_dict
    assert isinstance(torchgeo_dict["bbox"], BoundingBox), torchgeo_dict

    torchgeo_dict = collection[[torchgeo_bbox, torchgeo_bbox]]
    assert isinstance(torchgeo_dict, dict), type(torchgeo_dict)
    assert isinstance(torchgeo_dict["image"], torch.Tensor), torchgeo_dict["image"]
    assert isinstance(torchgeo_dict["bbox"][0], BoundingBox), torchgeo_dict["bbox"]


def test_sorting():
    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)

    largest_date = ""
    for img in collection:
        assert img.date > largest_date
        largest_date = img.date

    assert collection[0].date.startswith("2017"), img.date
    assert collection[1].date.startswith("2023"), img.date
    assert collection[2].date.startswith("2023"), img.date

    band = collection[0]["B02"]
    assert isinstance(band, sg.Band)
    assert band.date.startswith("2017"), band.date

    band = collection[1]["B02"]
    assert isinstance(band, sg.Band)
    assert band.date.startswith("2023"), band.date


def test_merge():
    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)

    # loop through each individual image to check the maths
    for ((tile, date), date_group), (img) in zip(
        collection.groupby(["tile", "date"]), collection, strict=True
    ):
        assert date == img.date
        assert tile == img.tile
        assert len(date_group) == 1

        # get 2d array with mean/median values of all bands in the image
        medianed = date_group.merge(method="median").values
        meaned = date_group.merge(method="mean").values
        assert meaned.shape == (300, 300)
        assert medianed.shape == (300, 300)

        # reading all bands as 3d array and taking mean/median manually
        arr = np.array([band.load().values for band in img])
        assert arr.shape in [
            (12, 300, 300),
            (13, 300, 300),
        ], arr.shape
        manually_meaned = np.mean(arr, axis=0).astype(int)
        assert manually_meaned.shape == (300, 300)

        manually_medianed = np.median(arr, axis=0).astype(int)
        assert manually_medianed.shape == (300, 300)

        assert int(np.mean(meaned)) == int(np.mean(manually_meaned))
        assert int(np.mean(medianed)) == int(np.mean(manually_medianed))

    merged_by_band = collection.merge_by_band()
    assert len(merged_by_band) == 13, len(merged_by_band)

    grouped_by_year_merged_by_band = collection.groupby("year").merge_by_band(
        method="mean"
    )
    for img in grouped_by_year_merged_by_band:
        for band in img:
            assert np.issubdtype(band.values.dtype, np.integer)

    grouped_by_year_merged_by_band = collection.groupby("year").merge_by_band(
        method="median"
    )
    for img in grouped_by_year_merged_by_band:
        for band in img:
            assert np.issubdtype(band.values.dtype, np.integer)

    assert isinstance(grouped_by_year_merged_by_band, sg.ImageCollection), type(
        grouped_by_year_merged_by_band
    )
    assert len(grouped_by_year_merged_by_band) == 2, len(grouped_by_year_merged_by_band)

    for img in grouped_by_year_merged_by_band:
        for band in img:
            assert band.band_id.startswith("B") or band.band_id == "SCL", band.band_id

    for img in grouped_by_year_merged_by_band:
        for band in img:
            arr = band.load(bounds=sg.to_shapely(band.bounds).buffer(-150)).values
            shape = arr.shape
            arr = band.load(bounds=sg.to_shapely(band.bounds).buffer(-150)).values
            assert arr.shape[0] < shape[0], (arr.shape, shape)
            assert arr.shape[1] < shape[1], (arr.shape, shape)

    sg.explore(merged_by_band)
    sg.explore(grouped_by_year_merged_by_band)

    merged_by_year = collection.groupby("year").merge()
    assert isinstance(merged_by_year, sg.Image), type(merged_by_year)
    assert len(merged_by_year) == 2, len(merged_by_year)

    sg.explore(merged_by_year)
    sg.explore(grouped_by_year_merged_by_band)
    sg.explore(merged_by_band)
    df = merged_by_band.to_gdf()
    assert (bounds := tuple(int(x) for x in df.total_bounds)) == (
        569631,
        6657859,
        657439,
        6672106,
    ), bounds
    assert len(merged_by_band) == (13), len(merged_by_band)

    merged_mean = collection.merge(method="mean")
    assert (merged_mean.values.shape) == (
        2612,
        4789,
    ), merged_mean.values.shape

    merged_median = collection.merge(method="median")
    assert (merged_median.values.shape) == (
        2612,
        4789,
    ), merged_median.values.shape


def test_date_ranges():
    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)

    assert len(collection) == 3

    imgs = collection.filter(date_ranges=(None, "20240101"))
    assert len(imgs) == 3, len(imgs)

    imgs = collection.filter(date_ranges=("20170101", "20180101"))
    assert len(imgs) == 1, len(imgs)

    imgs = collection.filter(date_ranges=("20230101", "20240101"))
    assert len(imgs) == 2, len(imgs)

    imgs = collection.filter(date_ranges=("20200101", "20220101"))
    assert len(imgs) == 0, len(imgs)

    imgs = collection.filter(date_ranges=("20170101", "20240101"))
    assert len(imgs) == 3, len(imgs)

    imgs = collection.filter(date_ranges=((None, "20180101"), ("20230101", None)))
    assert len(imgs) == 3, len(imgs)

    imgs = collection.filter(date_ranges=((None, "20180101"), ("20240101", None)))
    assert len(imgs) == 1, len(imgs)

    imgs = collection.filter(date_ranges=((None, "20170101"), ("20240101", None)))
    assert len(imgs) == 0, len(imgs)

    imgs = collection.filter(
        date_ranges=(("20170101", "20180101"), ("20230101", "20240101"))
    )
    assert len(imgs) == 3, len(imgs)


def test_groupby():

    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)

    assert isinstance(collection, Iterable)

    assert len(n := collection.groupby("date")) == 3, (n, len(n))
    assert len(n := collection.groupby("year")) == 2, (n, len(n))
    assert len(n := collection.groupby("month")) == 2, (n, len(n))
    assert len(n := collection.groupby("tile")) == 2, (n, len(n))
    assert len(n := collection.groupby("band_id")) == 13, (n, len(n))
    assert len(n := collection.groupby(["band_id", "date"])) == 38, (n, len(n))

    bands_should_be = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B09",
        "B11",
        "B12",
        "B8A",
        "SCL",
    ]

    for should_be, ((band_id,), subcollection) in zip(
        bands_should_be, collection.groupby("band_id"), strict=False
    ):
        assert isinstance(band_id, str), band_id
        assert band_id.startswith("B") or band_id.startswith("S"), band_id
        assert band_id == should_be, (band_id, should_be)
        assert isinstance(subcollection, sg.ImageCollection), type(subcollection)
        for img in subcollection:
            assert len(img.band_ids) == 1, img.band_ids
            assert len(img.file_paths) == 1
            assert isinstance(img.file_paths[0], str), img.file_paths
            assert img.band_ids[0] == band_id, (band_id, img.band_ids)

        largest_date = ""
        for img in subcollection:
            assert img.date > largest_date
            largest_date = img.date

    bands_should_be2 = sorted(bands_should_be + bands_should_be)

    # band_ids should appear twice in a row since there are two tiles
    for should_be, ((band_id, tile), subcollection) in zip(
        bands_should_be2, collection.groupby(["band_id", "tile"]), strict=False
    ):
        assert isinstance(tile, str)
        assert tile.startswith("T")
        assert isinstance(band_id, str), band_id
        assert band_id.startswith("B") or band_id.startswith("S"), band_id
        assert band_id == should_be, (band_id, should_be)
        assert isinstance(subcollection, sg.ImageCollection), type(subcollection)
        for img in subcollection:
            assert len(img.band_ids) == 1
            assert img.band_ids[0] == band_id, (band_id, img.band_ids)
            assert img.tile == tile
        largest_date = ""
        for img in subcollection:
            assert img.date > largest_date
            largest_date = img.date

    for (tile, band_id), subcollection in collection.groupby(["tile", "band_id"]):
        assert isinstance(tile, str)
        assert tile.startswith("T")
        assert isinstance(band_id, str), band_id
        assert band_id.startswith("B") or band_id.startswith("S"), band_id
        assert isinstance(subcollection, sg.ImageCollection), type(subcollection)
        for img in subcollection:
            assert len(img) == 1
            assert len(img.bands) == 1
            assert len(img.band_ids) == 1
            assert img.band_ids[0] == band_id
            assert img.tile == tile
        largest_date = ""
        for img in subcollection:
            assert img.date > largest_date
            largest_date = img.date

    for (date,), subcollection in collection.groupby("date"):
        assert isinstance(date, str), date
        assert date.startswith("20")
        assert isinstance(subcollection, sg.ImageCollection), type(subcollection)
        assert len(subcollection.band_ids) in [12, 13], subcollection.band_ids
        for img in subcollection:
            assert isinstance(img, sg.Image), type(img)
            assert img.date == date
            for band in img:
                assert band.date == date
                assert isinstance(band, sg.Band), type(band)
            for band_id in img.band_ids:
                assert isinstance(band_id, str), band_id
                arr = img[band_id]
                assert isinstance(arr, sg.Band), type(arr)
        largest_date = ""
        for img in subcollection:
            assert img.date > largest_date, (img.date, largest_date, subcollection, img)
            largest_date = img.date

    for (year,), subcollection in collection.groupby("year"):
        assert isinstance(year, str), year
        assert year.startswith("20")
        assert len(year) == 4, year
        for img in subcollection:
            assert img.year == year
            for band in img:
                assert band.year == year

        largest_date = ""
        for img in subcollection:
            assert img.date > largest_date, (img.date, largest_date, subcollection, img)
            largest_date = img.date

    for (month,), subcollection in collection.groupby("month"):
        merged = subcollection.merge_by_band()
        assert isinstance(month, str), month
        assert month.startswith("20")
        assert len(month) == 6, month
        for img in subcollection:
            assert img.month == month
            for band in img:
                assert band.month == month

        largest_date = ""
        for img in subcollection:
            assert img.date > largest_date, (img.date, largest_date, subcollection, img)
            largest_date = img.date


def test_regexes():
    """Regex search should work even if some patterns give no matches."""
    orig_img_regexes = list(sg.Sentinel2Collection.image_regexes)
    orig_file_regexes = list(sg.Sentinel2Collection.filename_regexes)

    sg.Sentinel2Collection.image_regexes = ["no_match"] + list(
        reversed(sg.Sentinel2Collection.image_regexes)
    )
    sg.Sentinel2Collection.filename_regexes = ["no_match"] + list(
        reversed(sg.Sentinel2Collection.filename_regexes)
    )

    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)
    assert len(collection.images) == 3, len(collection.images)

    collection.image_regexes = orig_img_regexes
    collection.filename_regexes = orig_file_regexes
    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)
    assert len(collection.images) == 3, len(collection.images)


def test_cloud():
    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)
    assert isinstance(collection, sg.ImageCollection), type(collection)
    assert len(collection) == 3, len(collection)

    collection2 = collection.filter(max_cloud_coverage=20)
    collection3 = collection.filter(max_cloud_coverage=40)
    collection4 = collection.filter(max_cloud_coverage=5)
    assert len(collection) == 3, len(collection)
    assert len(collection2) == 1, len(collection2)
    assert len(collection3) == 3, len(collection3)
    assert len(collection4) == 1, len(collection4)

    cloud_coverage_should_be = [36, 0, 25]
    for cloud_coverage, img in zip(cloud_coverage_should_be, collection, strict=False):
        assert cloud_coverage == int(img.cloud_coverage_percentage), (
            cloud_coverage,
            int(img.cloud_coverage_percentage),
            img.path,
        )

    for img in collection[[0, -1]]:
        cloud_arr = img.get_cloud_band()
        # assert np.sum(cloud_arr)
        assert isinstance(cloud_arr, sg.Band), cloud_arr
        assert cloud_arr.values.shape == (300, 300), cloud_arr.values.shape
        cloud_polys = img.get_cloud_band().to_gdf().geometry
        sg.explore(cloud_polys)
        assert isinstance(cloud_polys, GeoSeries), type(cloud_polys)


def test_iteration():

    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)
    assert isinstance(collection, sg.Sentinel2Collection), type(collection)
    assert len(collection.images) == 3, len(collection.images)

    assert len(collection.file_paths) == 38, len(collection.file_paths)
    assert len(collection) == 3, len(collection)

    assert isinstance(collection.unary_union, MultiPolygon), collection.unary_union

    # one of the imgs has no SCL band
    # n_bands = [13, 12, 13]
    # for n, img in zip(n_bands, collection, strict=False):
    for img in collection:
        assert isinstance(img, sg.Sentinel2Image), type(img)
        assert img.band_ids, img.band_ids
        assert all(x is not None for x in img.band_ids), img.band_ids
        assert img.bands, img.bands
        assert all(isinstance(x, sg.Band) for x in img.bands), img.bands
        assert all(isinstance(x, sg.Band) for x in img), img.__iter__()
        assert img.name, img.name

        assert img.date, img.date
        assert img.tile, img.tile
        assert img.level, img.level

        assert isinstance(img.unary_union, Polygon), img.unary_union

        assert img.file_paths, img.file_paths
        assert img.date.startswith("20"), img.date
        assert img.tile.startswith("T"), img.tile
        assert img.name.startswith("S2"), img.name
        assert isinstance(img.bounds, tuple)
        assert all(x for x in img.bounds)
        assert img.cloud_coverage_percentage, img.cloud_coverage_percentage
        assert img.crs
        assert img.centroid
        assert img.level == "L2A"
        # assert list(sorted(img.band_ids)) == list(
        #     sorted(sg.raster.sentinel_config.SENTINEL2_L2A_BANDS)
        # ), img.band_ids

        # arr = img.read()
        # assert isinstance(arr, np.ndarray), arr
        # assert (arr.shape) == (n, 300, 300), (i, n, arr.shape)

        # # without SCL band, always 12 bands
        # arr = img[img.l2a_bands].read()
        # assert isinstance(arr, np.ndarray), arr
        # assert (arr.shape) == (12, 300, 300), (i, arr.shape)

        # arr = img[["B02", "B03", "B04"]].read()
        # assert isinstance(arr, np.ndarray), arr
        # assert (arr.shape) == (3, 300, 300), arr.shape

        arr = img["B02"].load().values
        assert isinstance(arr, np.ndarray), arr
        assert (arr.shape) == (300, 300), arr.shape

        for band in img:
            assert isinstance(band, sg.Band), band
            arr = band.load().values
            assert isinstance(arr, np.ndarray), arr
            assert (arr.shape) == (300, 300), arr.shape

        for band_id, file_path in zip(img.band_ids, img.file_paths, strict=True):
            assert isinstance(band_id, str), band_id
            assert isinstance(file_path, str), type(file_path)
            raster = img[band_id]
            assert raster.band_id is not None, raster.band_id
            raster = img[file_path]
            assert raster.band_id is not None, raster.band_id


def test_iteration_base_image_collection():

    collection = sg.ImageCollection(path_sentinel, level=None, res=10)
    assert isinstance(collection, sg.ImageCollection), type(collection)

    assert len(collection.images) == 4, len(collection.images)
    assert len(collection) == 4, len(collection)
    # assert len(collection.explode()) == 38, len(collection.explode())
    assert len(collection.file_paths) == 54, len(collection.file_paths)

    for img in collection:
        assert isinstance(img, sg.Image), type(img)
        assert img.band_ids, img.band_ids
        assert img.bands, img.bands
        assert all(isinstance(x, sg.Band) for x in img.bands), img.bands
        assert all(isinstance(x, sg.Band) for x in img), img.__iter__()
        assert img.name, img.name

        assert img.file_paths, img.file_paths
        assert isinstance(img.bounds, tuple)
        assert all(x for x in img.bounds)
        assert img.crs
        assert img.centroid

        # arr = img.read()
        # assert isinstance(arr, np.ndarray), arr
        # assert (
        #     (arr.shape) == (13, 300, 300)
        #     or (arr.shape) == (12, 300, 300)
        #     or (arr.shape) == (16, 200, 200)
        # ), arr.shape

        for band in img:
            assert isinstance(band, sg.Band), band
            arr = band.load().values
            assert isinstance(arr, np.ndarray), arr
            assert (arr.shape) == (300, 300) or (arr.shape) == (200, 200), arr.shape


def test_convertion():
    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=100)
    band = collection[0]["B02"]

    arr = band.load().values
    _from_array = sg.Band(arr, res=band.res, crs=band.crs, bounds=band.bounds)
    assert (shape := _from_array.values.shape) == (30, 30), shape

    gdf = band.to_gdf(column="val")
    from_gdf = sg.Band(gdf, res=band.res)
    e = sg.explore(from_gdf.to_gdf(), band.to_gdf(), gdf)
    assert len(e._gdfs) == 3
    assert all(len(gdf) == 889 for gdf in e._gdfs), [len(gdf) for gdf in e._gdfs]
    assert all(int(gdf.area.sum()) == 9_000_000 for gdf in e._gdfs), [
        int(gdf.area.sum()) for gdf in e._gdfs
    ]
    assert (shape := from_gdf.values.shape) == (30, 30), shape


def test_torch():

    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)

    collection = collection.filter(
        bands=[
            band_id
            for band_id in collection.band_ids
            if band_id != collection.cloud_band
        ]
    )
    assert len(collection) == 3
    for img in collection:
        assert len(img) == 12, len(img)

    sampler = RandomGeoSampler(collection, size=16, length=10)
    dataloader = DataLoader(
        collection, batch_size=2, sampler=sampler, collate_fn=stack_samples
    )

    # Training loop
    for batch in dataloader:
        imgs = batch["image"]  # list of imgs
        boxes = batch["boxes"]  # list of boxes
        labels = batch["labels"]  # list of labels
        masks = batch["masks"]  # list of masks
        print(imgs)
        assert len(imgs.shape) == 4, imgs.shape

    sampler = RandomBatchGeoSampler(collection, size=16, length=10, batch_size=10)
    dataloader = DataLoader(
        collection, batch_size=2, sampler=sampler, collate_fn=stack_samples
    )

    # Training loop
    for batch in dataloader:
        imgs = batch["image"]  # list of imgs
        boxes = batch["boxes"]  # list of boxes
        labels = batch["labels"]  # list of labels
        masks = batch["masks"]  # list of masks
        print(imgs)
        assert len(imgs.shape) == 4, imgs.shape

    return

    datamodule = InriaAerialImageLabelingDataModule(
        # root=path_sentinel,
        batch_size=64,
        num_workers=6,
    )
    task = SemanticSegmentationTask(
        model="unet",
        backbone="resnet50",
        weights=True,
        in_channels=3,
        num_classes=2,
        loss="ce",
        ignore_index=None,
        lr=0.1,
        patience=6,
    )
    trainer = Trainer(default_root_dir=path_sentinel)

    trainer.fit(model=task, datamodule=datamodule)


def main():

    test_merge()
    test_zonal()
    test_gradient()
    test_groupby()
    test_concat_image_collections()
    test_iteration_base_image_collection()
    test_sample()
    test_with_mosaic()
    test_convertion()
    test_iteration()
    test_indexing()
    test_bbox()
    test_regexes()
    test_date_ranges()
    test_cloud()
    test_torch()
    test_ndvi_and_explore()


if __name__ == "__main__":
    main()
#     import cProfile

#     cProfile.run(
#         """
# main()
#                  """,
#         sort="cumtime",
#     )

# %%
