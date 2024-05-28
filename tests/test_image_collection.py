# %%
from typing import Iterable
import numpy as np
from pathlib import Path
from shapely import box
from shapely.geometry import Point, MultiPolygon
from geopandas import GeoSeries

src = str(Path(__file__).parent).replace("tests", "") + "src"
testdata = str(Path(__file__).parent.parent) + "/tests/testdata/raster"

import sys

sys.path.insert(0, src)


import sgis as sg

path_sentinel = testdata + "/sentinel2"


def test_bbox():
    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10, processes=2)

    no_images = collection.filter(bbox=Point(0, 0))  # bbox=Point(0, 0))
    assert not len(no_images), no_images

    centroid = collection[0].centroid
    images = collection.filter(bbox=centroid)  # bbox=centroid)
    assert len(images) == 3, images


def test_sample():
    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)

    change_sample = collection.sample_tiles(2)
    assert isinstance(change_sample, sg.Sentinel2Collection), type(change_sample)
    assert len(change_sample) == 3
    assert isinstance(change_sample[0], sg.Image)
    assert isinstance(change_sample[1], sg.Image)
    assert isinstance(change_sample[2], sg.Image)
    assert change_sample[0].date.startswith("2017")
    assert change_sample[1].date.startswith("2023")
    assert change_sample[2].date.startswith("2023")


def test_indexing():
    wrong_collection = sg.Sentinel2Collection(path_sentinel, level="L1C", res=10)
    assert not len(wrong_collection), len(wrong_collection)
    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)

    image = collection[0]
    assert isinstance(image, sg.Image)
    assert image.date.startswith("2017"), image.date
    assert "B02" in image
    assert "B00" not in image
    assert ["B02", "B03", "B04"] in image

    image = collection[1]
    assert isinstance(image, sg.Image)
    assert image.date.startswith("2023"), image.date

    image = collection[2]
    assert isinstance(image, sg.Image)
    assert image.date.startswith("2023"), image.date

    band = collection[0]["B02"]
    assert isinstance(band, sg.Band)
    assert band.date.startswith("2017"), band.date

    arr = collection[0]["B02"].load()
    assert isinstance(arr, np.ndarray)

    assert isinstance((x := collection[[0, -1]]), sg.ImageCollection), x
    assert len(x := collection[[0, -1]]) == 2, x
    assert isinstance((x := collection[0][["B02", "B03"]]), sg.Image), x
    assert isinstance((x := collection[0][["B02"]]), sg.Image), x
    assert isinstance((x := collection[0]["B02"]), sg.Band), x

    assert isinstance(collection[0]["B02"].load(), np.ndarray)


def test_sorting():
    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)

    largest_date = ""
    for image in collection:
        assert image.date > largest_date
        largest_date = image.date

    assert collection[0].date.startswith("2017"), image.date
    assert collection[1].date.startswith("2023"), image.date
    assert collection[2].date.startswith("2023"), image.date

    band = collection[0]["B02"]
    assert isinstance(band, sg.Band)
    assert band.date.startswith("2017"), band.date

    band = collection[1]["B02"]
    assert isinstance(band, sg.Band)
    assert band.date.startswith("2023"), band.date


def test_aggregate():
    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)

    single_band_collection = collection.filter(bands=["B02"])

    for (tile_id,), tile in single_band_collection.groupby("tile"):
        assert tile_id.startswith("T"), tile_id
        assert len(tile.band_ids) == 1, tile.band_ids
        for img in tile:
            print("img")
            print(img.load())

    agged = single_band_collection.aggregate_dates()
    assert (agged.shape) == (1, 250, 250), agged.shape
    print("agged")
    print(agged)

    rbg = collection.filter(bands=collection.rbg_bands)
    assert list(sorted(rbg.band_ids)) == list(sorted(collection.rbg_bands)), list(
        sorted(rbg.band_ids)
    )

    agged = rbg.aggregate_dates()
    assert (agged.shape) == (3, 250, 250), agged.shape


def test_date_ranges():
    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)

    assert len(collection) == 3

    images = collection.filter(date_ranges=(None, "20240101"))
    assert len(images) == 3, len(images)

    images = collection.filter(date_ranges=("20170101", "20180101"))
    assert len(images) == 1, len(images)

    images = collection.filter(date_ranges=("20230101", "20240101"))
    assert len(images) == 2, len(images)

    images = collection.filter(date_ranges=("20200101", "20220101"))
    assert len(images) == 0, len(images)

    images = collection.filter(date_ranges=("20170101", "20240101"))
    assert len(images) == 3, len(images)

    images = collection.filter(date_ranges=((None, "20180101"), ("20230101", None)))
    assert len(images) == 3, len(images)

    images = collection.filter(date_ranges=((None, "20180101"), ("20240101", None)))
    assert len(images) == 1, len(images)

    images = collection.filter(date_ranges=((None, "20170101"), ("20240101", None)))
    assert len(images) == 0, len(images)

    images = collection.filter(
        date_ranges=(("20170101", "20180101"), ("20230101", "20240101"))
    )
    assert len(images) == 3, len(images)


def test_groupby():

    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)

    assert isinstance(collection, Iterable)

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
        bands_should_be, collection.groupby("band")
    ):
        assert isinstance(band_id, str), band_id
        assert band_id.startswith("B") or band_id.startswith("S"), band_id
        assert band_id == should_be, (band_id, should_be)
        assert isinstance(subcollection, sg.ImageCollection), type(subcollection)
        for img in subcollection:
            assert len(img.band_ids) == 1
            assert len(img.file_paths) == 1
            assert isinstance(img.file_paths[0], str), img.file_paths
            assert img.band_ids[0] == band_id, (band_id, img.band_ids)

    bands_should_be2 = sorted(bands_should_be + bands_should_be)

    # band_ids should appear twice in a row since there are two tiles
    for should_be, ((band_id, tile), subcollection) in zip(
        bands_should_be2, collection.groupby(["band", "tile"])
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

    for (tile, band_id), subcollection in collection.groupby(["tile", "band"]):
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

    for (date,), subcollection in collection.groupby("date"):
        assert isinstance(date, str), date
        assert date.startswith("20")
        assert isinstance(subcollection, sg.ImageCollection), type(subcollection)
        assert len(subcollection.band_ids) in [12, 13], subcollection._df[
            "band_filename"
        ].explode()
        assert subcollection._df["band_filename"].notna().all(), subcollection._df[
            "band_filename"
        ].explode()
        for img in subcollection:
            assert isinstance(img, sg.Image), type(img)
            assert img.date == date
            for band in img:
                assert isinstance(band, sg.Band), type(band)
            for band_id in img.band_ids:
                assert isinstance(band_id, str), band_id
                arr = img[band_id]
                assert isinstance(arr, sg.Band), type(arr)

    assert len(n := collection.groupby("date")) == 3, (n, len(n))
    assert len(n := collection.groupby("tile")) == 2, (n, len(n))
    assert len(n := collection.groupby("band")) == 13, (n, len(n))
    assert len(n := collection.groupby(["band", "date"])) == 38, (n, len(n))


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
    assert len(collection._df["band_filename"].explode()) == 38, collection._df[
        "band_filename"
    ].explode()

    collection.image_regexes = orig_img_regexes
    collection.filename_regexes = orig_file_regexes
    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)
    assert len(collection.images) == 3, len(collection.images)
    assert len(collection._df["band_filename"].explode()) == 38, collection._df[
        "band_filename"
    ].explode()


def test_cloud():
    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)
    assert isinstance(collection, sg.ImageCollection), type(collection)
    assert len(collection) == 3, len(collection)

    collection2 = collection.filter(max_cloud_cover=20)
    collection3 = collection.filter(max_cloud_cover=40)
    collection4 = collection.filter(max_cloud_cover=5)
    assert len(collection) == 3, len(collection)
    assert len(collection2) == 1, len(collection2)
    assert len(collection3) == 3, len(collection3)
    assert len(collection4) == 1, len(collection4)

    cloud_cover_should_be = [36, 0, 25]
    for cloud_cover, image in zip(cloud_cover_should_be, collection):
        assert cloud_cover == int(image.cloud_cover_percentage), (
            cloud_cover,
            int(image.cloud_cover_percentage),
            image.path,
        )

    for image in collection[[0, -1]]:
        cloud_arr = image.get_cloud_array()
        # assert np.sum(cloud_arr)
        assert isinstance(cloud_arr, np.ndarray), cloud_arr
        assert cloud_arr.shape == (250, 250), cloud_arr.shape
        cloud_polys = image.get_cloud_polygons()
        sg.explore(cloud_polys)
        assert isinstance(cloud_polys, GeoSeries), type(cloud_polys)


def test_iteration():

    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)
    assert isinstance(collection, sg.ImageCollection), type(collection)

    assert len(collection.images) == 3, len(collection.images)
    assert len(collection.file_paths) == 38, len(collection.file_paths)
    assert len(collection._df) == 3, len(collection._df)
    assert len(collection._df["band_filename"].explode()) == 38, collection._df[
        "band_filename"
    ].explode()
    assert len(collection) == 3, len(collection)

    collection.df = collection.df.iloc[[0]]
    assert len(collection) == 1, len(collection)
    assert len(collection.images) == 1, len(collection.images)

    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)

    # one of the images has no SCL band
    n_bands = [13, 12, 13]
    for i, (n, image) in enumerate(zip(n_bands, collection)):
        assert len(image._df["band_filename"]) == n, (
            i,
            n,
            list(image._df["band_filename"]),
        )
        assert isinstance(image, sg.Image), type(image)
        assert image.band_ids, image.band_ids
        assert all(x is not None for x in image.band_ids), image.band_ids
        assert image.bands, image.bands
        assert all(isinstance(x, sg.Band) for x in image.bands), image.bands
        assert all(isinstance(x, sg.Band) for x in image), image.__iter__()
        assert image.name, image.name

        assert image.date, image.date
        assert image.tile, image.tile
        assert image.level, image.level

        assert image.file_paths, image.file_paths
        assert image.date.startswith("20"), image.date
        assert image.tile.startswith("T"), image.tile
        assert image.name.startswith("S2"), image.name
        assert isinstance(image.bounds, tuple)
        assert all(x for x in image.bounds)
        assert image.cloud_cover_percentage, image.cloud_cover_percentage
        assert image.crs
        assert image.centroid
        assert image.level == "L2A"
        # assert list(sorted(image.band_ids)) == list(
        #     sorted(sg.raster.sentinel_config.SENTINEL2_L2A_BANDS)
        # ), image.band_ids

        arr = image.load()
        assert isinstance(arr, np.ndarray), arr
        assert (arr.shape) == (n, 250, 250), (i, n, arr.shape)

        # without SCL band, always 12 bands
        arr = image[image.l2a_bands].load()
        assert isinstance(arr, np.ndarray), arr
        assert (arr.shape) == (12, 250, 250), (i, arr.shape)

        arr = image[["B02", "B03", "B04"]].load()
        assert isinstance(arr, np.ndarray), arr
        assert (arr.shape) == (3, 250, 250), arr.shape

        arr = image["B02"].load()
        assert isinstance(arr, np.ndarray), arr
        assert (arr.shape) == (250, 250), arr.shape

        for band in image:
            assert isinstance(band, sg.Band), band
            arr = band.load()
            assert isinstance(arr, np.ndarray), arr
            assert (arr.shape) == (250, 250), arr.shape

        for band_id, file_path in zip(image.band_ids, image.file_paths, strict=True):
            assert isinstance(band_id, str), band_id
            assert isinstance(file_path, str), type(file_path)
            raster = image[band_id]
            assert raster.band_id is not None, raster.band_id
            raster = image[file_path]
            assert raster.band_id is not None, raster.band_id


def main():
    test_indexing()
    test_aggregate()
    test_bbox()
    test_iteration()
    test_groupby()
    test_regexes()
    test_cloud()
    test_date_ranges()
    test_sample()


if __name__ == "__main__":
    main()
    import cProfile

    cProfile.run(
        """
main()
                 """,
        sort="cumtime",
    )

# %%
