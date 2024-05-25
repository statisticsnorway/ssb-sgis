# %%
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
    collection = sg.raster.image_collection.Sentinel2(path_sentinel)

    no_images = collection.get_images(bbox=Point(0, 0))
    assert not len(no_images), no_images

    centroid = box(*collection.get_images()[0].bounds).centroid
    images = collection.get_images(bbox=centroid)
    assert len(images) == 2, images


def test_sample():
    collection = sg.raster.image_collection.Sentinel2(path_sentinel)

    change_sample = collection.sample_change()
    assert isinstance(change_sample, list)
    assert len(change_sample) == 2
    assert isinstance(change_sample[0], sg.raster.image_collection.Image)
    assert isinstance(change_sample[1], sg.raster.image_collection.Image)
    assert change_sample[0].date.startswith("2018")
    assert change_sample[1].date.startswith("2023")

    assert len(collection.sample_tiles(1)) == 1
    assert len(collection.sample_tiles(1)[0][[0, -1]]) == 2
    assert isinstance(collection.sample_tiles(1)[0][[0, -1]], list)
    assert isinstance(
        collection.sample_tiles(1)[0], sg.raster.image_collection.ImageTile
    )
    assert isinstance(
        collection.sample_tiles(1)[0][0], sg.raster.image_collection.Image
    )

    assert len(collection.sample_images(2)) == 2

    for tile in collection.get_tiles():
        assert len(tile.sample_images(1)) == 1
        assert len(tile.sample_images(2)) == 2


def test_indexing():
    collection = sg.raster.image_collection.Sentinel2(path_sentinel)
    assert isinstance(collection.get_tiles()[0][0]["B02"], sg.Raster)

    assert len(collection.get_tiles()) == 1
    assert isinstance(collection.get_tiles(), list)
    assert isinstance(collection.get_tiles()[0], sg.raster.image_collection.ImageTile)

    image = collection.get_tiles()[0][0]
    assert isinstance(image, sg.raster.image_collection.Image)
    assert image.date.startswith("2018"), image.date
    image = collection.get_tiles()[0][-1]
    assert isinstance(image, sg.raster.image_collection.Image)

    assert image.date.startswith("2023"), image.date

    assert isinstance(collection.get_tiles()[0][[0, -1]], list)
    assert isinstance(collection.get_tiles()[0][[0]], list)
    assert isinstance(
        collection.get_tiles()[0][[0]][0], sg.raster.image_collection.Image
    )

    assert isinstance(collection.get_tiles()[0][0]["B02"], sg.Raster)


def test_iteration():

    collection = sg.raster.image_collection.Sentinel2(path_sentinel)
    assert isinstance(collection, sg.raster.image_collection.ImageCollection), type(
        collection
    )
    assert len(collection.get_tiles()) == 1, collection.get_tiles()
    assert (n := len(collection.image_paths)) == 2, n

    images = collection.get_images()
    assert len(images) == 2, len(images)

    for tile in collection.get_tiles():
        assert isinstance(tile, sg.raster.image_collection.ImageTile), type(tile)
        assert (n := len(tile.image_paths)) == 2, n

        for image in tile.get_images():
            assert isinstance(image, sg.raster.image_collection.Image), type(image)
            assert (n := len(image.data)) == 12, n
            assert image.date.startswith("20")
            assert image.tile.startswith("T")
            assert image.name.startswith("T")
            assert image.bounds
            assert image.crs
            assert image.centroid
            assert image.level == "L2A"
            assert image.bands == list(
                sg.raster.sentinel2_config.SENTINEL2_L2A_BANDS
            ), image.bands
            assert image.resolutions == list(
                sg.raster.sentinel2_config.SENTINEL2_L2A_RESOLUTIONS
            ), image.resolutions

            for raster in image:
                assert isinstance(raster, sg.raster.raster.Raster), type(raster)
                assert raster.band is not None, raster.band
                assert raster.indexes == 1, raster.indexes
                assert len(raster.shape) == 2, raster.shape

    for image in collection.get_images():
        assert isinstance(image, sg.raster.image_collection.Image), type(image)
        assert (n := len(image.data)) == 12, n
        for raster in image:
            assert isinstance(raster, sg.raster.raster.Raster), type(raster)
            assert raster.band is not None, raster.band
            assert raster.indexes == 1, raster.indexes
            assert len(raster.shape) == 2, raster.shape


if __name__ == "__main__":
    test_indexing()
    test_iteration()
    test_sample()
    test_bbox()
