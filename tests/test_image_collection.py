# %%


from pathlib import Path
from collections.abc import Iterable

import numpy as np
from geopandas import GeoSeries
from shapely.geometry import Point
from torchgeo.datasets.utils import BoundingBox
from shapely.geometry import MultiPolygon, Polygon
import torch
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
from torchgeo.datamodules import InriaAerialImageLabelingDataModule
from torchgeo.datasets import stack_samples
from torchgeo.samplers import RandomBatchGeoSampler, RandomGeoSampler
from torchgeo.trainers import SemanticSegmentationTask


src = str(Path(__file__).parent).replace("tests", "") + "src"
testdata = str(Path(__file__).parent.parent) + "/tests/testdata/raster"

import sys

sys.path.insert(0, src)


import sgis as sg

path_sentinel = testdata + "/sentinel2"


def concat_image_collections():
    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10, processes=2)

    new_collection = sg.concat_image_collections(
        [collection[[i]] for i in range(len(collection))]
    )
    assert len(new_collection) == len(collection)
    for k, v in new_collection.__dict__.items():
        if k == "path":
            assert v is None
            continue
        if k in ["_df", "_images", "_all_filepaths"]:
            continue
        assert v == getattr(
            collection, k
        ), f"different value for '{k}': {v} and {getattr(collection, k)}"


def test_ndvi_and_explore():
    """Running ndvi and checking how it's plotted with explore."""
    n = 3000

    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)

    e = sg.Explore(collection)

    e.explore()
    assert e.rasters
    assert (x := [x["label"] for x in e.raster_data]) == [
        f"{img.tile}_{img.date[:8]}" for img in collection
    ], x

    collection = collection.filter(bands=collection.ndvi_bands)

    for (tile_id,), tile_collection in collection.groupby("tile"):

        for img in tile_collection:
            ndvi = img.get_ndvi()

            assert (ndvi.cmap) == "Greens"

            e = sg.Explore(ndvi)
            e.explore()
            assert e.rasters
            assert (x := list(sorted([x["label"] for x in e.raster_data]))) == [
                "ndvi",
            ], x

            e = sg.Explore(ndvi.get_n_largest(n), ndvi.get_n_smallest(n), img)
            e.explore()
            assert e.rasters
            assert e["labels"] == [
                f"largest_{n}",
                f"smallest_{n}",
            ], e["labels"]

            assert (x := list(sorted([x["label"] for x in e.raster_data]))) == [
                f"{img.tile}_{img.date[:8]}"
            ], x

            new_img = sg.Image([ndvi], res=10)

            e = sg.Explore(new_img)
            e.explore()
            assert e.rasters
            assert (x := list(sorted([x["label"] for x in e.raster_data]))) == [
                "new_img",
            ], x

            e = sg.Explore(sg.Image([ndvi], res=10))
            e.explore()
            assert e.rasters
            # cannot find an object name since the image is created within the Explore constructor
            assert (x := list(sorted([x["label"] for x in e.raster_data]))) == [
                "Image(0)",
            ], x


def test_bbox():
    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10, processes=2)

    no_imgs = collection.filter(bbox=Point(0, 0))  # bbox=Point(0, 0))
    assert not len(no_imgs), no_imgs

    centroid = collection[0].centroid
    imgs = collection.filter(bbox=centroid)  # bbox=centroid)
    assert len(imgs) == 2, imgs

    centroid = collection[1].centroid
    imgs = collection.filter(bbox=centroid)  # bbox=centroid)
    assert len(imgs) == 2, imgs

    centroid = collection[2].centroid
    imgs = collection.filter(bbox=centroid)  # bbox=centroid)
    assert len(imgs) == 1, imgs


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
        assert len(date_group) == 1

        # get 2d array with mean/median values of all bands in the image
        meaned = date_group.merge(method="mean").values
        medianed = date_group.merge(method="median").values
        assert meaned.shape == (300, 300)
        assert medianed.shape == (300, 300)

        # reading all bands as 3d array and taking mean/median manually
        arr = img.read()
        assert arr.shape in [
            (12, 300, 300),
            (13, 300, 300),
        ], arr.shape
        manually_meaned = np.mean(arr, axis=0)
        assert manually_meaned.shape == (300, 300)

        manually_medianed = np.median(arr, axis=0)
        assert manually_medianed.shape == (300, 300)

        assert int(np.mean(meaned)) == int(np.mean(manually_meaned))
        assert int(np.mean(medianed)) == int(np.mean(manually_medianed))

        # same as above
        arr2 = []
        for band in img:
            arr2.append(band.load().values)
        arr2 = np.array(arr2)
        assert arr2.shape in [
            (12, 300, 300),
            (13, 300, 300),
        ], arr2.shape

        manually_meaned2 = np.mean(arr2, axis=0)
        assert manually_meaned2.shape == (300, 300)

        manually_medianed2 = np.median(arr2, axis=0)
        assert manually_medianed2.shape == (300, 300)

        assert int(np.mean(meaned)) == int(np.mean(manually_meaned2))
        assert int(np.mean(medianed)) == int(np.mean(manually_medianed2))

    merged_by_band = collection.merge_by_band()
    df = merged_by_band.to_gdf()
    assert (bounds := tuple(int(x) for x in df.total_bounds)) == (
        569631,
        6657859,
        657439,
        6672106,
    ), bounds
    assert (merged_by_band.values.shape) == (
        13,
        2612,
        4789,
    ), merged_by_band.values.shape

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
        bands_should_be, collection.groupby("band"), strict=False
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
        bands_should_be2, collection.groupby(["band", "tile"]), strict=False
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
    for cloud_cover, img in zip(cloud_cover_should_be, collection, strict=False):
        assert cloud_cover == int(img.cloud_cover_percentage), (
            cloud_cover,
            int(img.cloud_cover_percentage),
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
    assert len(collection._df) == 3, len(collection._df)
    assert len(collection._df["band_filename"].explode()) == 38, collection._df[
        "band_filename"
    ].explode()
    assert len(collection) == 3, len(collection)

    collection.df = collection.df.iloc[[0]]
    assert len(collection) == 1, len(collection)
    assert len(collection.images) == 1, len(collection.images)

    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)

    assert isinstance(collection.unary_union, MultiPolygon), collection.unary_union

    # one of the imgs has no SCL band
    n_bands = [13, 12, 13]
    for i, (n, img) in enumerate(zip(n_bands, collection, strict=False)):
        assert len(img._df["band_filename"]) == n, (
            i,
            n,
            list(img._df["band_filename"]),
        )
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
        assert img.cloud_cover_percentage, img.cloud_cover_percentage
        assert img.crs
        assert img.centroid
        assert img.level == "L2A"
        # assert list(sorted(img.band_ids)) == list(
        #     sorted(sg.raster.sentinel_config.SENTINEL2_L2A_BANDS)
        # ), img.band_ids

        arr = img.read()
        assert isinstance(arr, np.ndarray), arr
        assert (arr.shape) == (n, 300, 300), (i, n, arr.shape)

        # without SCL band, always 12 bands
        arr = img[img.l2a_bands].read()
        assert isinstance(arr, np.ndarray), arr
        assert (arr.shape) == (12, 300, 300), (i, arr.shape)

        arr = img[["B02", "B03", "B04"]].read()
        assert isinstance(arr, np.ndarray), arr
        assert (arr.shape) == (3, 300, 300), arr.shape

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

    collection = sg.ImageCollection(path_sentinel, level="L2A", res=10)
    assert isinstance(collection, sg.ImageCollection), type(collection)

    assert len(collection.images) == 3, len(collection.images)
    assert len(collection.file_paths) == 38, len(collection.file_paths)
    assert len(collection._df) == 3, len(collection._df)
    assert len(collection) == 3, len(collection)

    collection.df = collection.df.iloc[[0]]
    assert len(collection) == 1, len(collection)
    assert len(collection.images) == 1, len(collection.images)

    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)

    # one of the imgs has no SCL band
    n_bands = [13, 12, 13]
    for i, (n, img) in enumerate(zip(n_bands, collection, strict=False)):
        assert len(img._df["band_filename"]) == n, (
            i,
            n,
            list(img._df["band_filename"]),
        )
        assert isinstance(img, sg.Image), type(img)
        assert img.band_ids, img.band_ids
        assert all(x is not None for x in img.band_ids), img.band_ids
        assert img.bands, img.bands
        assert all(isinstance(x, sg.Band) for x in img.bands), img.bands
        assert all(isinstance(x, sg.Band) for x in img), img.__iter__()
        assert img.name, img.name

        assert img.date, img.date
        assert img.tile, img.tile
        assert img.level, img.level

        assert img.file_paths, img.file_paths
        assert img.date.startswith("20"), img.date
        assert img.tile.startswith("T"), img.tile
        assert img.name.startswith("S2"), img.name
        assert isinstance(img.bounds, tuple)
        assert all(x for x in img.bounds)
        assert img.crs
        assert img.centroid
        assert img.level == "L2A"
        # assert list(sorted(img.band_ids)) == list(
        #     sorted(sg.raster.sentinel_config.SENTINEL2_L2A_BANDS)
        # ), img.band_ids

        arr = img.read()
        assert isinstance(arr, np.ndarray), arr
        assert (arr.shape) == (n, 300, 300), (i, n, arr.shape)

        # without SCL band, always 12 bands
        arr = img[img.l2a_bands].read()
        assert isinstance(arr, np.ndarray), arr
        assert (arr.shape) == (12, 300, 300), (i, arr.shape)

        arr = img[["B02", "B03", "B04"]].read()
        assert isinstance(arr, np.ndarray), arr
        assert (arr.shape) == (3, 300, 300), arr.shape

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
        print(boxes)
        print(labels)
        print(masks)
        print(imgs)
        assert len(imgs.shape) == 4, imgs.shape

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
    test_iteration()
    test_indexing()
    test_bbox()
    test_groupby()
    test_regexes()
    test_date_ranges()
    test_sample()
    test_cloud()
    test_iteration_base_image_collection()
    concat_image_collections()
    test_ndvi_and_explore()
    test_torch()


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
