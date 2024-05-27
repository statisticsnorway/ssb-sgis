# %%
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
    collection = sg.Sentinel2Collection(path_sentinel, processes=1)

    no_images = collection.filter_bounds(Point(0, 0)).get_images()  # bbox=Point(0, 0))
    assert not len(no_images), no_images

    centroid = box(*collection.get_images()[0].bounds).centroid
    images = collection.filter_bounds(centroid).get_images()  # bbox=centroid)
    assert len(images) == 3, images


def test_sample():
    collection = sg.Sentinel2Collection(path_sentinel)

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
    collection = sg.Sentinel2Collection(path_sentinel)
    # assert isinstance(collection[0][0]["B02"], sg.Raster)

    assert len(collection.get_tiles()) == 2
    assert isinstance(collection.get_tiles(), list)
    assert isinstance(collection[0], sg.Tile)

    image = collection[0][0]
    assert isinstance(image, sg.Image)
    assert image.date.startswith("2017"), image.date

    image = collection[0][-1]
    assert isinstance(image, sg.Image)
    assert image.date.startswith("2023"), image.date

    assert isinstance(collection[0][[0, -1]], list)
    assert isinstance(collection[0][[0]], list)
    assert isinstance(collection[0][[0]][0], sg.Image)

    assert isinstance(collection[0][0]["B02"], np.ndarray)


def test_date_ranges():
    collection = sg.Sentinel2Collection(path_sentinel)

    images = collection.get_images()
    assert len(images) == 3, len(images)

    images = collection.get_images(date_ranges=(None, "20240101"))
    assert len(images) == 3, len(images)

    images = collection.get_images(date_ranges=("20230101", "20240101"))
    assert len(images) == 2, len(images)

    images = collection.get_images(date_ranges=("20170101", "20180101"))
    assert len(images) == 1, len(images)

    images = collection.get_images(date_ranges=("20200101", "20220101"))
    assert len(images) == 0, len(images)

    images = collection.get_images(date_ranges=("20170101", "20240101"))
    assert len(images) == 3, len(images)

    images = collection.get_images(date_ranges=((None, "20180101"), ("20230101", None)))
    assert len(images) == 3, len(images)

    images = collection.get_images(date_ranges=((None, "20180101"), ("20240101", None)))
    assert len(images) == 1, len(images)

    images = collection.get_images(date_ranges=((None, "20170101"), ("20240101", None)))
    assert len(images) == 0, len(images)

    images = collection.get_images(
        date_ranges=(("20170101", "20180101"), ("20230101", "20240101"))
    )
    assert len(images) == 3, len(images)


def test_groupby():

    collection = sg.Sentinel2Collection(path_sentinel)

    for subcollection in collection.groupby("date"):
        assert isinstance(subcollection, sg.TileCollection), type(subcollection)
        assert len(subcollection.dates) == 1, subcollection._df["date"]
        assert len(subcollection.band_ids) == 12, subcollection._df["band"]
        assert subcollection._df["band"].notna().all(), subcollection._df["band"]
        for image in subcollection:
            assert isinstance(image, sg.Image), type(image)
            for band in image:
                assert isinstance(image, sg.Band), type(image)

    for subcollection in collection.groupby("tile"):
        assert isinstance(subcollection, sg.TileCollection), type(subcollection)
        assert len(subcollection.tiles) == 1

    for subcollection in collection.groupby(["tile", "band"]):
        assert isinstance(subcollection, sg.TileCollection), type(subcollection)
        assert len(subcollection.dates) in [1, 2], subcollection._df["date"]
        assert len(subcollection.tiles) == 1, subcollection._df["tile"]

    assert len(n := collection.groupby("date")) == 3, (n, len(n))
    assert len(n := collection.groupby("tile")) == 2, (n, len(n))
    assert len(n := collection.groupby("band")) == 12, (n, len(n))
    assert len(n := collection.groupby(["band", "date"])) == 36, (n, len(n))


def test_iteration():

    collection = sg.Sentinel2Collection(path_sentinel)
    assert isinstance(collection, sg.TileCollection), type(collection)
    assert len(collection.file_paths) == 41, len(collection.file_paths)
    # assert len(collection) == 2, collection
    assert (n := len(collection.image_paths)) == 3, n

    for image in collection.get_images():

        assert isinstance(image, sg.Image)

    print(collection.group_paths_by_date())
    print(collection.group_paths_by_band())

    # arr = collection.aggregate_dates()
    # print(arr.shape)

    for tile in collection:

        assert isinstance(tile, sg.Tile), type(tile)
        assert (n := len(tile.image_paths)) in [1, 2], n

        print(tile.group_paths_by_band())

        # arr = tile.aggregate_dates()
        # assert isinstance(arr, np.ndarray), arr
        # assert len(arr.shape) == 3, arr.shape

        for image in tile.get_images():
            assert isinstance(image, sg.Image), type(image)
            # assert (n := len(image.data)) == 12, n
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

            # arr = image.load(bands=["B02", "B03", "B04"])
            # assert isinstance(arr, np.ndarray), arr
            # assert (arr.shape) == (3, 250, 250), arr.shape

            arr = image[["B02", "B03", "B04"]].load()
            assert isinstance(arr, np.ndarray), arr
            assert (arr.shape) == (3, 250, 250), arr.shape

            arr = image.load(bands=["B02", "B03", "B04"])
            assert isinstance(arr, np.ndarray), arr
            assert (arr.shape) == (3, 250, 250), arr.shape

            # assert image.resolutions == (
            #     sg.raster.sentinel_config.SENTINEL2_L2A_BANDS
            # ), image.resolutions

            for band in image:
                arr = band.load()
                assert isinstance(arr, np.ndarray), arr
                assert len(arr.shape) == 2, arr.shape

            for band_id, file_path in image.items():
                assert isinstance(band_id, str), type(band_id)
                assert isinstance(file_path, str), type(file_path)
                raster = image[band_id]
                assert raster.band is not None, raster.band
                assert raster.indexes == 1, raster.indexes
                assert len(raster.shape) == 2, raster.shape
                raster = image[file_path]
                assert raster.band is not None, raster.band
                assert raster.indexes == 1, raster.indexes
                assert len(raster.shape) == 2, raster.shape

            for raster in image.get_bands():
                assert isinstance(raster, sg.raster.raster.Raster), type(raster)
                assert raster.band is not None, raster.band
                assert raster.indexes == 1, raster.indexes
                assert len(raster.shape) == 2, raster.shape

    for image in collection.get_images():
        assert isinstance(image, sg.Image), type(image)
        # assert (n := len(image.data)) == 12, n
        for raster in image.get_bands():
            assert isinstance(raster, sg.raster.raster.Raster), type(raster)
            assert raster.band is not None, raster.band
            assert raster.indexes == 1, raster.indexes
            assert len(raster.shape) == 2, raster.shape


def main():
    test_groupby()
    test_indexing()
    test_date_ranges()
    test_iteration()
    test_bbox()
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
