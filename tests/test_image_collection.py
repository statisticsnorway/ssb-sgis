# %%
from typing import Iterable
import numpy as np
from pathlib import Path
from shapely import box
from shapely.geometry import Point

src = str(Path(__file__).parent).replace("tests", "") + "src"
testdata = str(Path(__file__).parent.parent) + "/tests/testdata/raster"

import sys

sys.path.insert(0, src)


import sgis as sg

path_sentinel = testdata + "/sentinel2"


def test_bbox():
    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10, processes=2)

    no_images = collection.filter_bounds(Point(0, 0)).get_images()  # bbox=Point(0, 0))
    assert not len(no_images), no_images

    centroid = box(*collection.get_images()[0].bounds).centroid
    images = collection.filter_bounds(centroid).get_images()  # bbox=centroid)
    assert len(images) == 3, images


def test_sample():
    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)

    # x: list[sg.Tile] = collection.sample_tiles(2)
    # x: sg.Tile = collection.sample_tile()
    # x: list[list[sg.Image]] = [
    #     tile.sample_images(5) for tile in collection.sample_tiles(3)
    # ]
    # x: list[sg.Image] = [
    #     collection.sample_tile().sample_images(5) for _ in range(len(3))
    # ]
    # x: list[sg.Image] = collection.sample_tile().sample_images(10)
    # x: list[sg.Image] = collection[0].sample_images(10)
    # x: list[sg.Image] = collection.sample_images(10)

    change_sample = collection.sample_tiles(2)
    assert isinstance(change_sample, list), type(change_sample)
    assert len(change_sample) == 2
    assert isinstance(change_sample[0], sg.Image)
    assert isinstance(change_sample[1], sg.Image)
    assert change_sample[0].date.startswith("2017")
    assert change_sample[1].date.startswith("2023")

    assert len(collection.sample_tiles(1)) == 1
    assert len(collection.sample_tiles(1)[0][[0, -1]]) == 2
    assert isinstance(collection.sample_tiles(1)[0][[0, -1]], list)
    assert isinstance(collection.sample_tiles(1)[0], sg.Tile)
    assert isinstance(collection.sample_tiles(1)[0][0], sg.Image)

    assert len(collection.sample_images(2)) == 2

    for tile in collection:
        assert len(tile.sample_images(1)) == 1
        assert len(tile.sample_images(2)) == 2


def test_indexing():
    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)

    largest_date = ""
    for image in collection:
        assert image.date > largest_date
        largest_date = image.date

    image = collection[0]
    assert isinstance(image, sg.Image)
    assert image.date.startswith("2017"), image.date

    band = collection[0]["B02"]
    assert isinstance(band, sg.Band)
    assert band.date.startswith("2023"), band.date

    arr = collection[0]["B02"]
    assert isinstance(arr, np.ndarray)

    assert isinstance(collection[0][[0, -1]], list)
    assert isinstance(collection[0][[0]], list)
    assert isinstance(collection[0][[0]][0], sg.Image)

    assert isinstance(collection[0][0]["B02"], np.ndarray)


def test_sorting():
    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)

    assert list(collection.image_paths) == list(sorted(collection.image_paths))

    assert isinstance(image, sg.Image)
    assert image.date.startswith("2017"), image.date

    band = collection[0]["B02"]
    assert isinstance(band, sg.Band)
    assert band.date.startswith("2023"), band.date

    arr = collection[0]["B02"]
    assert isinstance(arr, np.ndarray)

    assert isinstance(collection[0][[0, -1]], list)
    assert isinstance(collection[0][[0]], list)
    assert isinstance(collection[0][[0]][0], sg.Image)

    assert isinstance(collection[0][0]["B02"], np.ndarray)


def test_aggregate():
    collection = sg.Sentinel2Collection(path_sentinel, res=10)

    rbg = collection.filter(bands=[collection.rbg_bands])
    for tile_id, tile in rbg.groupby("tile"):
        assert tile_id.startswith("T"), tile_id
        for img in tile:
            print(img.load())

    agged = rbg.aggregate_dates()
    print(agged)
    print(agged.shape)
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

    for date, subcollection in collection.groupby("date"):
        assert isinstance(date, str)
        assert date.startswith("20")
        assert isinstance(subcollection, sg.ImageCollection), type(subcollection)
        assert len(subcollection.band_ids) == 12, subcollection._df["band"]
        assert subcollection._df["band"].notna().all(), subcollection._df["band"]
        for image in subcollection:
            assert isinstance(image, sg.Image), type(image)
            for band in image:
                assert isinstance(band, sg.Band), type(band)
            for band_id in image.band_ids:
                assert isinstance(band_id, str), type(band_id)
                arr = image[band_id]
                assert isinstance(arr, sg.Band), type(arr)

    for tile, subcollection in collection.groupby("tile"):
        assert isinstance(tile, str)
        assert tile.startswith("T")
        assert isinstance(subcollection, sg.ImageCollection), type(subcollection)

    for (tile, band), subcollection in collection.groupby(["tile", "band"]):
        assert isinstance(tile, str)
        assert tile.startswith("T")
        assert isinstance(band, str)
        assert band.startswith("B")
        assert isinstance(subcollection, sg.ImageCollection), type(subcollection)

    assert len(n := collection.groupby("date")) == 3, (n, len(n))
    assert len(n := collection.groupby("tile")) == 2, (n, len(n))
    assert len(n := collection.groupby("band")) == 12, (n, len(n))
    assert len(n := collection.groupby(["band", "date"])) == 36, (n, len(n))


def test_iteration():

    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)
    assert isinstance(collection, sg.ImageCollection), type(collection)

    assert len(collection.images) == 3, len(collection.images)
    assert len(collection.file_paths) == 36, len(collection.file_paths)
    assert len(collection._df) == 3, len(collection._df)
    assert len(collection._df.explode("band_filename")) == 36
    assert len(collection) == 3, len(collection)

    collection.df = collection.df.iloc[[0]]
    assert len(collection) == 1, len(collection)
    assert len(collection.images) == 1, len(collection.images)

    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)

    for image in collection:
        assert len(image._df) == 1, len(image._df)
        assert len(image._df.explode("band_filename")) == 12, image._df.explode(
            "band_filename"
        )
        assert isinstance(image, sg.Image), type(image)
        # assert (n := len(image.data)) == 12, n
        assert image.band_ids, image.band_ids
        assert all(x is not None for x in image.band_ids), image.band_ids
        assert image.bands, image.bands
        assert all(isinstance(x, sg.Band) for x in image.bands), image.bands
        assert all(isinstance(x, sg.Band) for x in image), image.__iter__()
        assert image.name, image.name
        assert image.tile, image.tile
        assert image.date, image.date
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
        assert list(sorted(image.band_ids)) == list(
            sorted(sg.raster.sentinel_config.SENTINEL2_L2A_BANDS)
        ), image.band_ids

        arr = image.load()
        assert isinstance(arr, np.ndarray), arr
        assert (arr.shape) == (12, 250, 250), arr.shape

        arr = image["B02"].load()
        assert isinstance(arr, np.ndarray), arr
        assert (arr.shape) == (250, 250), arr.shape

        arr = image[["B02", "B03", "B04"]].load()
        assert isinstance(arr, np.ndarray), arr
        assert (arr.shape) == (3, 250, 250), arr.shape

        for band in image:
            assert isinstance(band, sg.Band), band
            arr = band.load()
            assert isinstance(arr, np.ndarray), arr
            assert (arr.shape) == (250, 250), arr.shape

        for band_id, file_path in zip(image.band_ids, image.file_paths, strict=True):
            assert isinstance(band_id, str), type(band_id)
            assert isinstance(file_path, str), type(file_path)
            raster = image[band_id]
            assert raster.band_id is not None, raster.band_id
            raster = image[file_path]
            assert raster.band_id is not None, raster.band_id


def main():
    test_iteration()
    test_groupby()
    test_date_ranges()
    test_indexing()
    test_bbox()
    test_aggregate()
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
