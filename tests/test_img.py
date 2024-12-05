# %%
import os
import platform
import re
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyproj
import pytest
from pyproj.exceptions import CRSError
from rasterio.errors import RasterioIOError
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon
from sklearn.ensemble import RandomForestRegressor

src = str(Path(__file__).parent).replace("tests", "") + "src"
testdata = str(Path(__file__).parent.parent) + "/tests/testdata/raster"

metadata_df_path = str(Path(testdata) / "sentinel2_metadata.parquet")

import sys

sys.path.insert(0, src)


import sgis as sg

path_sentinel = testdata + "/sentinel2"
path_singleband = testdata + "/dtm_10.tif"
path_two_bands = testdata + "/dtm_10_two_bands.tif"


skip_if_github_and_not_linux = pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true" and platform.system().lower() != "linux",
    reason="Skipping test because it's not running in GitHub Actions",
)


def print_function_name(func):
    def wrapper(*args, **kwargs):
        print(f'Calling function: {func.__name__} {args or ""} {kwargs or ""}')
        result = func(*args, **kwargs)
        print(f"Function {func.__name__} finished.")
        return result

    return wrapper


@print_function_name
def test_zonal():

    r = sg.Band(path_singleband, res=None).load()
    gdf = sg.make_grid(r.bounds, 100, crs=r.crs)

    sg.explore(sg.to_gdf(r.bounds, r.crs), gdf)

    gdf.index = [np.random.choice([*"abc"]) for _ in range(len(gdf))]

    zonal_stats = r.zonal(gdf, aggfunc=[sum, np.mean, "median"], array_func=None)
    print(zonal_stats)
    print(gdf)


def test_buffer():

    arr = np.zeros((50, 50))
    arr[10, 10] = 1
    arr[20, 20] = 1
    arr[20, 10] = 1
    arr[10, 20] = 1

    band = sg.Band(arr, crs=25833, bounds=(0, 0, 50, 50))

    buffered = band.copy().buffer(10)

    should_equal = np.array(
        [[1 for _ in range(31)] + [0 for _ in range(19)] for _ in range(31)]
        + [[0 for _ in range(50)] for _ in range(19)]
    )
    assert buffered.values.shape == should_equal.shape, (
        buffered.values.shape,
        should_equal.shape,
    )
    assert np.array_equal(buffered.values, should_equal), sg.explore(
        buffered,
        should_equal=sg.Band(should_equal, crs=25833, bounds=(0, 0, 50, 50), res=10),
    )

    buffered2 = buffered.copy().buffer(-10)

    should_equal = np.array(
        [[0 for _ in range(50)] for _ in range(10)]
        + [
            [0 for _ in range(10)] + [1 for _ in range(11)] + [0 for _ in range(29)]
            for _ in range(11)
        ]
        + [[0 for _ in range(50)] for _ in range(29)]
    )
    assert buffered.values.shape == should_equal.shape, (
        buffered.values.shape,
        should_equal.shape,
    )

    assert np.array_equal(buffered2.values, should_equal), sg.explore(
        buffered2,
        should_equal=sg.Band(should_equal, crs=25833, bounds=(0, 0, 50, 50), res=10),
    )

    sg.explore(band, buffered, buffered2)


@print_function_name
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
    band = sg.Band(arr, crs=None, bounds=(0, 0, 50, 50))
    gradient = band.gradient(copy=True)
    assert np.max(gradient.values) == 1, gradient.values

    degrees = band.gradient(degrees=True, copy=True)

    assert np.max(degrees.values) == 45, np.max(degrees.values)


@print_function_name
def test_with_mosaic():

    mosaic = sg.Sentinel2CloudlessCollection(
        path_sentinel, level=None, res=10, processes=2
    )
    for img in mosaic:
        assert isinstance(img, sg.Sentinel2CloudlessImage), type(img)
        for band in img:
            assert isinstance(band, sg.Sentinel2CloudlessBand), type(band)
            print(img.filename_regexes)
            assert band.band_id is not None
    sg.explore(mosaic)
    assert len(mosaic) == 1, mosaic

    collection = sg.Sentinel2Collection(path_sentinel, level=None, res=10, processes=2)
    assert len(collection) == 3, collection
    assert list(collection.date) == list(sorted(collection.date)), collection.date

    concated = sg.concat_image_collections([mosaic, collection])
    assert len(concated) == 4, concated


@print_function_name
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


@print_function_name
def demo():

    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)

    sg.explore(collection)

    merged = collection.groupby(["tile", "date"]).merge_by_band()

    sg.explore(merged)

    ndvi = merged.ndvi()

    sg.explore(ndvi)


@print_function_name
def test_explore():

    collection = sg.Sentinel2Collection(
        path_sentinel, level="L2A", res=10
    ).sort_images()

    e = sg.explore(collection)
    assert e.rasters
    assert (x := [x["label"] for x in e.raster_data]) == [
        img.name for img in collection
    ], x


@print_function_name
def test_single_banded():

    collection = sg.ImageCollection(
        Path(testdata) / "ndvi", level=None, res=10
    ).sort_images()

    assert len(collection) == 2, len(collection)
    image_names = list(sorted([img.name for img in collection]))
    assert image_names == ["copies", "ndvi"], image_names

    single_banded = collection.explode().sort_images()
    assert len(single_banded) == 6, len(collection)
    print([img.name for img in single_banded])
    print([img.date for img in single_banded])
    image_names = [img.name for img in single_banded]
    assert image_names == [
        "ndvi_T32VNM20170826.tif",
        "ndvi_T32VNM20170826_copy.tif",
        "ndvi_T32VNM20230606.tif",
        "ndvi_T32VNM20230606_copy.tif",
        "ndvi_T32VPM20230624.tif",
        "ndvi_T32VPM20230624_copy.tif",
    ], image_names

    band_names = [band.name for img in single_banded for band in img]
    assert band_names == image_names, band_names


@skip_if_github_and_not_linux
@print_function_name
def test_plot_pixels():

    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)
    collection = collection.filter(bbox=collection[0].centroid.buffer(6))
    collection.load()
    collection.plot_pixels()


@print_function_name
def test_ndvi():

    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)
    _test_ndvi(collection, np.ma.core.MaskedArray, cloudless=False)

    collection = sg.Sentinel2CloudlessCollection(path_sentinel, level=None, res=10)
    print(collection)
    _test_ndvi(collection, np.ndarray, cloudless=True)


@print_function_name
def _test_ndvi(collection, type_should_be, cloudless: bool):
    """Running ndvi and checking how it's plotted with explore."""
    n = 1000

    for (tile_id,), tile_collection in collection.groupby("tile"):

        for img in tile_collection:
            assert img.tile == tile_id, (img.tile, tile_id)
            img = img[img.ndvi_bands]
            if not cloudless:
                img = img.apply(
                    lambda band: (band.load().values + (band.boa_add_offset or 0))
                    / band.boa_quantification_value
                )
            ndvi = img.ndvi()
            assert isinstance(ndvi.values, type_should_be), type(ndvi.values)
            assert ndvi.values.max() <= 1, ndvi.values.max()
            assert ndvi.values.min() >= -1, ndvi.values.min()

            if type_should_be == np.ma.core.MaskedArray:
                assert np.sum(ndvi.values.mask)

            print("explore ndvi, masking=", img.masking)
            gdf = ndvi.to_geopandas()
            gdf_sample = gdf.sample(min(n, len(gdf)))
            e = sg.explore(ndvi, gdf_sample=gdf_sample, column="value")

            assert e.rasters
            assert (x := list(sorted([x["label"] for x in e.raster_data]))) == [
                "ndvi",
            ], (x, ndvi.values, e.__dict__)

            e = sg.explore(ndvi.get_n_largest(n), ndvi.get_n_smallest(n), img)
            assert e.rasters
            assert e["labels"] == [
                f"largest_{n}",
                f"smallest_{n}",
            ], e["labels"]

            new_img = sg.Image([ndvi], res=10)

            e = sg.explore(new_img)
            assert e.rasters
            assert (x := list(sorted([x["label"] for x in e.raster_data]))) == [
                "NDVIBand(band_id=None, "
            ], x

            e = sg.explore(sg.Image([ndvi], res=10))
            assert e.rasters
            # cannot find an object name since the image is created within the explore constructor
            assert (x := list(sorted([x["label"] for x in e.raster_data]))) == [
                "NDVIBand(band_id=None, "
            ], x


@print_function_name
def test_ndvi_predictions():
    _test_ndvi_predictions(run_lstsq)
    _test_ndvi_predictions(run_random_forest)


@print_function_name
def test_pixelwise():
    array1 = np.ma.array(
        [
            [100, 100, 100],
            [0.4, 0.35, 0.3],
        ],
        mask=[[True, True, True], [False, False, False]],
    )
    array2 = np.ma.array(
        [
            [100, 100, 0.3],
            [0.4, 0.5, 0.6],
        ],
        mask=[[True, True, False], [False, False, False]],
    )
    array3 = np.ma.array(
        [
            [100, 0.1, 0.6],
            [0.9, 0.45, 0.9],
        ],
        mask=[[True, False, False], [False, False, False]],
    )

    # sg.raster.image_collection.pixelwise(
    #     np.array([array1, array2, array3]),
    #     mask_array=np.array([arr.mask for arr in [array1, array2, array3]])
    # )

    collection = sg.ImageCollection(
        [
            sg.Image([sg.Band(arr, bounds=(0, 0, 1, 1), crs=25833)])
            for arr in [array1, array2, array3]
        ],
        res=10,
    )

    def return_self(x):
        return x

    row_indices, col_indices, values = collection.pixelwise(return_self).to_tuple()
    assert list(row_indices) == list([0, 0, 1, 1, 1]), row_indices
    assert list(col_indices) == list([1, 2, 0, 1, 2]), col_indices
    assert isinstance(values, list), type(values)

    dict_results = collection.pixelwise(return_self).to_dict()
    keys_should_be = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert list(dict_results.keys()) == keys_should_be, dict_results
    gdf = collection.pixelwise(return_self).to_geopandas()
    assert list(gdf.index) == keys_should_be, list(gdf.index)
    assert list(gdf.columns) == ["value", "geometry"]
    print(dict_results)
    print(gdf)

    arrays = collection.pixelwise(return_self).to_numpy()
    print(arrays)

    lengths_should_be = np.array([[np.nan, 1, 2], [3, 3, 3]])

    def run_pixelwise(pixel_values, days_since_start, not_alligned_array):
        assert pixel_values.max() < 1, pixel_values
        assert len(pixel_values) >= 1
        assert pixel_values.shape == days_since_start.shape

        assert len(not_alligned_array) == 3

        if len(pixel_values) == 2:
            assert np.array_equal(
                days_since_start, np.array([100, 110])
            ), days_since_start

        if len(pixel_values) == 1:
            assert np.array_equal(days_since_start, np.array([110])), days_since_start

        return len(pixel_values)

    days_since_start = np.array([0, 100, 110])

    lengths = collection.pixelwise(
        run_pixelwise,
        index_aligned_kwargs=dict(days_since_start=days_since_start),
        kwargs=dict(not_alligned_array=days_since_start),
    ).to_numpy()
    assert np.array_equal(
        lengths, lengths_should_be.astype(lengths.dtype), equal_nan=True
    ), lengths

    def run_pixelwise_not_masked(pixel_values, days_since_start):
        assert len(pixel_values) == 3, pixel_values
        assert len(days_since_start) == 3, days_since_start

    _, _, _ = collection.pixelwise(
        run_pixelwise_not_masked,
        index_aligned_kwargs=dict(days_since_start=days_since_start),
        masked=False,
    ).to_tuple()

    predicted_start, predicted_end, n_observations = collection.pixelwise(
        get_predictions_1d,
        index_aligned_kwargs=dict(a=days_since_start),
        kwargs=dict(prediction_func=run_lstsq),
    ).to_numpy()

    # rounding manually since numpy cannot round None
    predicted_start = np.round(predicted_start, 3)
    predicted_end = np.round(predicted_end, 3)
    # predicted_start = np.array(
    #     [
    #         [round(value, 3) if value is not None else None for value in row]
    #         for row in predicted_start
    #     ]
    # )
    # predicted_end = np.array(
    #     [
    #         [round(value, 3) if value is not None else None for value in row]
    #         for row in predicted_end
    #     ]
    # )

    assert np.array_equal(
        n_observations, lengths_should_be, equal_nan=True
    ), n_observations
    assert np.array_equal(
        predicted_start,
        np.array([[np.nan, 0.1, 0.3], [0.377, 0.353, 0.288]]),
        equal_nan=True,
    ), predicted_start

    assert np.array_equal(
        predicted_end,
        np.array([[np.nan, 0.1, 0.6], [0.675, 0.479, 0.778]]),
        equal_nan=True,
    ), predicted_end

    predicted_start, predicted_end, n_observations = collection.pixelwise(
        get_predictions_1d,
        index_aligned_kwargs=dict(a=days_since_start),
        kwargs=dict(prediction_func=run_random_forest),
    ).to_numpy()

    assert np.array_equal(
        n_observations, lengths_should_be, equal_nan=True
    ), n_observations


@print_function_name
def _test_ndvi_predictions(prediction_func):
    nodata = -2
    collection = sg.Sentinel2Collection(
        path_sentinel, level="L2A", res=10, nodata=nodata
    )

    collection = collection.filter(intersects=collection[0].centroid)
    assert len(collection) == 2, len(collection)
    assert list(collection.date) == ["20170826", "20230606"], list(collection.date)

    collection.load()

    mask = collection.union_all().centroid.buffer(100)
    collection.images = [
        collection[0].clip(mask.buffer(100)),
        collection[1].clip(mask),
    ]
    sg.explore(
        x1=collection[0][0].to_geopandas(),
        x2=collection[1][0].to_geopandas(),
        msk=sg.to_gdf(mask),
        # browser=True,
    )
    first_mask = collection[0][0].values.mask
    assert (first_mask == False).sum() == 1245, (first_mask == False).sum()
    second_mask = collection[1][0].values.mask
    assert (second_mask == False).sum() == 305, (second_mask == False).sum()

    for band in collection[0]:
        band.boa_add_offset = -1000

    def normalize(band: sg.Band):
        values = band.values
        values = (values + band.boa_add_offset) / band.boa_quantification_value
        band.values = (values - np.min(values)) / (np.max(values) - np.min(values))
        return band

    collection.images = [img.apply(normalize).ndvi(padding=0.05) for img in collection]

    days_since_start = np.array(
        (pd.to_datetime(collection.date) - pd.Timestamp(min(collection.date))).dt.days
    )

    predicted_start, predicted_end, n_observations = collection.pixelwise(
        func=get_predictions_1d,
        index_aligned_kwargs=dict(a=days_since_start),
        kwargs=dict(
            prediction_func=prediction_func,
        ),
    ).to_numpy()

    assert np.sum(n_observations == 1) == 940, np.sum(n_observations == 1)
    assert np.sum(n_observations == 2) == 305, np.sum(n_observations == 2)
    assert np.sum(n_observations == nodata) == 88156, np.sum(n_observations == nodata)
    assert np.all(np.isin(n_observations, [1, 2, nodata]))

    # assert (x := np.mean(predicted_start, where=predicted_start != nodata)) <= 0.5, x
    # assert (x := np.mean(predicted_end, where=predicted_end != nodata)) <= 0.5, x
    # assert (x := np.mean(predicted_start, where=predicted_start != nodata)) > 0.25, x
    # assert (x := np.mean(predicted_end, where=predicted_end != nodata)) > 0.25, x

    # assert np.max(predicted_start, where=predicted_start != nodata) <= 1
    # assert np.min(predicted_start, where=predicted_start != nodata) >= -1
    # assert np.max(predicted_end, where=predicted_end != nodata) <= 1
    # assert np.min(predicted - _end, where=predicted_end != nodata) >= -1

    predicted_start = sg.Band(
        predicted_start,
        bounds=collection.bounds,
        crs=collection.crs,
    )
    predicted_end = sg.Band(
        predicted_end,
        bounds=collection.bounds,
        crs=collection.crs,
    )
    n_observations = sg.Band(
        n_observations,
        bounds=collection.bounds,
        crs=collection.crs,
    )

    sg.explore(
        collection.to_geopandas(),
        n_observations=n_observations.to_geopandas(),
        # browser=True,
    )

    sg.explore(
        predicted_start=predicted_start.to_geopandas(),
        predicted_end=predicted_end.to_geopandas(),
        n_observations=n_observations.to_geopandas(),
        column="value",
        # browser=True,
    )


def get_predictions_1d(
    pixel_values: np.ndarray,
    a: np.ndarray,
    prediction_func,
) -> tuple[np.float64, np.float64, int]:
    assert np.min(pixel_values) >= -1, pixel_values
    assert np.max(pixel_values) <= 1, pixel_values

    assert len(a) == len(pixel_values), (len(a), len(pixel_values))

    predicted_start, predicted_end = prediction_func(a, pixel_values)

    n_observations = len(pixel_values)

    return predicted_start, predicted_end, n_observations


def run_lstsq(a: np.ndarray, b: np.ndarray) -> tuple[np.float64, np.float64]:
    a_with_ones = np.vstack([a, np.ones(b.shape[0])]).T
    coef, intercept = np.linalg.lstsq(a_with_ones, b, rcond=None)[0]
    predicted_start = intercept + coef * a[0]
    predicted_end = intercept + coef * a[-1]
    return predicted_start, predicted_end


def run_random_forest(
    a: np.ndarray,
    b: np.ndarray,
    regressor: RandomForestRegressor | None = None,
):
    if regressor is None:
        regressor = RandomForestRegressor(min_samples_split=5, n_estimators=5)
    return predict_with_regressor(a, b, regressor=regressor)


def predict_with_regressor(
    a: np.ndarray,
    b: np.ndarray,
    regressor: RandomForestRegressor,
) -> tuple[np.float64, np.float64]:
    a_reshaped = a.reshape(-1, 1)

    regressor.fit(a_reshaped, b)
    predicted = regressor.predict(a_reshaped)

    predicted_start = predicted[0]
    predicted_end = predicted[-1]

    return predicted_start, predicted_end


@print_function_name
def test_bbox():

    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10, processes=1)
    no_imgs = collection.filter(bbox=Point(0, 0))
    assert not len(no_imgs), no_imgs

    centroid = collection[0].centroid
    imgs = collection.filter(bbox=centroid)
    assert len(imgs) == 2, imgs
    for img in imgs:
        for band in img:
            shape = band.load().values.shape
            assert shape == (0,), shape

    imgs = collection.filter(bbox=centroid.buffer(100))

    assert len(imgs) == 2, imgs
    for img in imgs:
        for band in img:
            shape = band.load().values.shape

            assert shape == (20, 20), shape

    no_imgs = collection.filter(intersects=Point(0, 0))  # intersects=Point(0, 0))
    assert not len(no_imgs), no_imgs

    centroid = collection[0].centroid
    imgs = collection.filter(intersects=centroid)  # intersects=centroid)
    assert len(imgs) == 2, imgs

    centroid = collection[1].centroid
    imgs = collection.filter(intersects=centroid)  # intersects=centroid)
    assert len(imgs) == 2, imgs

    centroid = collection[2].centroid
    imgs = collection.filter(intersects=centroid)  # intersects=centroid)
    assert len(imgs) == 1, imgs

    # same with indexing

    no_imgs = collection[[img.intersects(Point(0, 0)) for img in collection]]
    assert not len(no_imgs), no_imgs

    centroid = collection[0].centroid
    imgs = collection[[img.intersects(centroid) for img in collection]]
    assert len(imgs) == 2, imgs

    centroid = collection[1].centroid
    imgs = collection[[img.intersects(centroid) for img in collection]]
    assert len(imgs) == 2, imgs

    centroid = collection[2].centroid
    imgs = collection[[img.intersects(centroid) for img in collection]]
    assert len(imgs) == 1, imgs


@print_function_name
def not_test_sample():

    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)

    size = 200

    for _ in range(5):
        for img in collection.sample(1, size=size):
            for band in img:
                band.load()
                print(band.to_geopandas().pipe(sg.sort_small_first).geometry.area.sum())
        e = sg.explore(collection.sample(1, size=size))

    # low buffer resolution means the area won't be exactly this
    circle_area_should_be = int((size**2) * 3.14159265359)

    e = sg.explore(collection.sample(1, size=size))
    assert (x := e.raster_data[0]["arr"].shape) <= (40, 40, 3), x

    bbox = sg.to_gdf(collection.union_all(), collection.crs)

    e = sg.samplemap(collection, bbox, size=size)
    assert len(e.raster_data) in [1, 2], e.raster_data

    assert (x := e.raster_data[0]["arr"].shape) <= (40, 40, 3), x
    assert (x := int(e._gdfs[0].dissolve().area.sum())) <= circle_area_should_be, (
        x,
        circle_area_should_be,
    )

    print("as gdfs")
    e = sg.explore(
        collection.sample(1, size=size).load().to_geopandas(),
        column="value",
        return_explorer=True,
    )
    if 0:
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


@print_function_name
def test_collection_from_list_of_path():
    paths = [Path(path_sentinel) / name for name in os.listdir(path_sentinel)]
    collection = sg.Sentinel2Collection(
        paths,
        level="L2A",
        res=10,
    )
    len(collection)  # trigger image creation
    collection2 = sg.Sentinel2Collection(
        path_sentinel,
        level="L2A",
        res=10,
    )
    len(collection2)  # trigger image creation

    assert collection.equals(collection2)

    collection3 = sg.Sentinel2Collection(
        [sg.Sentinel2Image(path, res=10) for path in paths if "L2A" in str(path)],
        level="L2A",
        res=10,
    )
    len(collection3)  # trigger image creation

    assert collection.equals(collection3)

    collection4 = sg.Sentinel2Collection(
        [
            sg.Sentinel2Image(
                [
                    sg.Sentinel2Band(Path(path) / name, res=10)
                    for name in os.listdir(path)
                    if ".tif" in name and "SCL" not in name
                ]
            )
            for path in paths
            if "L2A" in str(path)
        ],
        level="L2A",
        res=10,
    )
    len(collection4)  # trigger image creation

    assert collection.equals(collection4)


def test_metadata_attributes():
    _test_metadata_attributes(metadata_from_xml=True)
    _test_metadata_attributes(metadata_from_xml=False)


@print_function_name
def _test_metadata_attributes(metadata_from_xml: bool):
    """Metadata attributes should be accessible through xml files for both Band, Image and Collection."""
    if not metadata_from_xml:
        metadata = get_metadata_df([testdata], 1, band_endswith="m_clipped.tif")
        # metadata = pd.read_parquet(metadata_df_path)
    else:
        metadata = None

    first_img_path = (
        Path(path_sentinel)
        / "S2A_MSIL2A_20230624T104621_N0509_R051_T32VPM_20230624T170454.SAFE"
    )

    img = sg.Sentinel2Image(first_img_path, res=10, metadata=metadata)
    assert [band._metadata_from_xml is metadata_from_xml for band in img]
    assert img.processing_baseline == "05.00", img.processing_baseline
    assert int(img.cloud_cover_percentage) == 25, img.cloud_cover_percentage
    assert img.is_refined is True, img.is_refined

    band = sg.Sentinel2Band(
        first_img_path / "T32VPM_20230624T104621_B02_10m_clipped.tif",
        res=10,
        metadata=metadata,
    )
    assert band.processing_baseline == "05.00", band.processing_baseline
    assert int(band.cloud_cover_percentage) == 25, band.cloud_cover_percentage
    assert band.is_refined is True, band.is_refined
    assert band.boa_add_offset == -1000, band.boa_add_offset
    assert band.boa_quantification_value == 10000, band.boa_quantification_value

    collection = sg.Sentinel2Collection(
        path_sentinel, level="L2A", res=10, metadata=metadata
    )

    offsets = []
    for img in collection:
        assert img.processing_baseline in [
            "02.08",
            "05.09",
            "05.00",
        ], img.processing_baseline
        for band in img:
            assert band.processing_baseline in [
                "02.08",
                "05.09",
                "05.00",
            ], band.processing_baseline

            offsets.append(band.boa_add_offset == -1000)
            assert band.boa_add_offset in [-1000, None], band.boa_add_offset

    assert sum(offsets) == 24, (sum(offsets), offsets)
    assert (x := list(collection.processing_baseline)) == ["02.08", "05.09", "05.00"], x

    assert (x := list(collection.is_refined)) == [False, True, True], x

    assert (x := list(collection.cloud_cover_percentage.fillna(0).astype(int))) == [
        36,
        0,
        25,
    ], x

    only_high_cloud_percentage = collection[collection.cloud_cover_percentage > 30]
    assert len(only_high_cloud_percentage) == 1, len(only_high_cloud_percentage)

    only_correct_processing_baseline = collection[
        collection.processing_baseline >= "05.00"
    ]
    assert len(only_correct_processing_baseline) == 2, len(
        only_correct_processing_baseline
    )

    only_refined = collection[collection.is_refined]
    assert len(only_refined) == 2, len(only_refined)


@print_function_name
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
    assert (x := band.load().value_counts()).sum() == 84571, (x, x.sum())
    assert (x := band.value_counts()).max() == 341, (x, x.sum())

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


@print_function_name
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


@print_function_name
def test_masking():

    n_loads = sg.raster.image_collection._LOAD_COUNTER
    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)
    collection.load()
    assert sg.raster.image_collection._LOAD_COUNTER == 39 + n_loads, (
        sg.raster.image_collection._LOAD_COUNTER + n_loads
    )
    assert (
        sg.raster.image_collection._LOAD_COUNTER
        == len({band for img in collection for band in img})
        + 3
        + n_loads  # three masks
    ), sg.raster.image_collection._LOAD_COUNTER

    for img in collection:
        for band in img:
            assert isinstance(band.values, np.ma.core.MaskedArray)

    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=20)

    img = collection[2]

    band_addresses = tuple(id(band) for band in img)

    for i, band in enumerate(img):
        assert band_addresses[i] == id(band), (band_addresses[i], id(band))

    assert (
        sg.raster.image_collection._LOAD_COUNTER == 39 + n_loads
    ), sg.raster.image_collection._LOAD_COUNTER

    assert img.masking

    for i, band in enumerate(img):
        assert band_addresses[i] == id(band), (band_addresses[i], id(band))
        assert band._values is None
        band.load()

    for i, band in enumerate(img):
        assert band_addresses[i] == id(band), (band_addresses[i], id(band))

    for i, band in enumerate(img):
        assert band_addresses[i] == id(band), (band_addresses[i], id(band))

    collection.load()

    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=20)
    collection.load()

    for img in collection:
        for band in img:
            assert isinstance(band.values, np.ma.core.MaskedArray), type(band.values)

    assert collection.masking
    for img in collection:
        assert img.masking
        assert "SCL" not in img
        for band in img:
            assert band.band_id != "SCL"
            band = band.load()
            print(band)
            print(band.values)
            assert np.sum(band.values)
            assert np.sum(band.values.data)
            assert np.sum(band.values.mask)
            assert isinstance(band.values, np.ma.core.MaskedArray)

    collection = sg.ImageCollection(path_sentinel, level=None, res=20)

    assert collection.masking is None
    assert len(collection)
    for img in collection:
        assert img.masking is None
        for band in img:
            band = band.load()
            assert np.sum(band.values)
            assert np.sum(band.values.data)


@print_function_name
def test_merge():

    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10, nodata=0)

    # merged_by_band = collection.merge_by_band(method="mean", nodata=0)
    # for band in merged_by_band:
    #     band.write(
    #         f"c:/users/ort/git/ssb-sgis/tests/testdata/raster/{band.band_id}.tif"
    #     )
    t = perf_counter()
    collection.load()

    merged_by_band = collection.merge_by_band(method="mean")
    print("merged_by_band mean", perf_counter() - t)
    t = perf_counter()
    # merged_by_band = collection.merge_by_band(method="median", nodata=0)
    # print("merged_by_band median", perf_counter() - t)

    print()
    print(collection.bounds)
    print(merged_by_band.bounds)
    sg.explore(
        collection,
        merged_by_band,
        cc=sg.to_gdf(collection.bounds, collection.crs).assign(area=lambda x: x.area),
        bbb=sg.to_gdf(merged_by_band.bounds, collection.crs).assign(
            area=lambda x: x.area
        ),
    )
    print()
    print()
    assert len(merged_by_band) == 12, len(merged_by_band)

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
        continue

        assert meaned.shape == (299, 299)
        assert medianed.shape == (299, 299)

        # reading all bands as 3d array and taking mean/median manually
        arr = np.array([band.load().values for band in img])
        assert arr.shape in [
            (12, 299, 299),
            (13, 299, 299),
        ], arr.shape
        manually_meaned = np.mean(arr, axis=0).astype(int)
        assert manually_meaned.shape == (299, 299)

        manually_medianed = np.median(arr, axis=0).astype(int)
        assert manually_medianed.shape == (299, 299)

        assert int(np.mean(meaned)) == int(np.mean(manually_meaned))
        assert int(np.mean(medianed)) == int(np.mean(manually_medianed))

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

    return

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
    df = merged_by_band.to_geopandas()
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


@print_function_name
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


@print_function_name
def test_groupby():

    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)

    assert isinstance(collection, Iterable)

    assert len(n := collection.groupby("date")) == 3, (n, len(n))
    assert len(n := collection.groupby("year")) == 2, (n, len(n))
    assert len(n := collection.groupby("month")) == 2, (n, len(n))
    assert len(n := collection.groupby(["year", "month"])) == 2, (n, len(n))
    assert len(n := collection.groupby("tile")) == 2, (n, len(n))
    assert len(n := collection.groupby("band_id")) == 12, (n, len(n))
    assert len(n := collection.groupby(["band_id", "date"])) == 36, (n, len(n))

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
        band_ids = list({band.band_id for img in subcollection for band in img})
        assert len(band_ids) in [12, 13], band_ids
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

    for (
        year,
        month,
    ), subcollection in collection.groupby(["year", "month"]):
        merged = subcollection.load().merge_by_band()
        assert isinstance(month, str), month
        assert month.startswith("0")
        assert isinstance(year, str), year
        assert year.startswith("20")
        assert len(month) == 2, month
        assert len(year) == 4, year
        for img in subcollection:
            assert img.month == month
            for band in img:
                assert band.month == month

        largest_date = ""
        for img in subcollection:
            assert img.date > largest_date, (img.date, largest_date, subcollection, img)
            largest_date = img.date


@print_function_name
def test_regexes():
    """Regex search should work even if some patterns give no matches."""
    default_pat = re.compile(
        sg.raster.image_collection.DEFAULT_FILENAME_REGEX, flags=re.VERBOSE
    )
    assert not re.search(default_pat, "ndvi")
    assert re.search(default_pat, "ndvi.tif")

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


@print_function_name
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


@print_function_name
def test_iteration():

    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)

    assert isinstance(collection, sg.Sentinel2Collection), type(collection)
    assert len(collection.images) == 3, len(collection.images)

    file_paths = list(sorted({band.path for img in collection for band in img}))
    assert len(file_paths) == 36, len(file_paths)
    assert len(collection) == 3, len(collection)

    assert isinstance(collection.union_all(), MultiPolygon), collection.union_all()

    for img in collection:
        with pytest.raises(KeyError):
            img[img.masking["band_id"]]
        assert isinstance(img, sg.Sentinel2Image), type(img)
        assert img.band_ids, img.band_ids
        assert all(x is not None for x in img.band_ids), img.band_ids
        assert img.bands, img.bands
        assert all(isinstance(x, sg.Band) for x in img.bands), img.bands
        assert all(isinstance(x, sg.Band) for x in img), list(img)
        assert img.name, img.name

        assert img.date, img.date
        assert img.tile, img.tile
        assert img.level, img.level

        assert isinstance(img.union_all(), Polygon), img.union_all()

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

        arr = img["B02"].load().values
        assert isinstance(arr, np.ndarray), arr
        assert (arr.shape) == (299, 299), arr.shape

        any_8a = False
        for band in img:
            assert isinstance(band, sg.Band), band
            arr = band.load().values
            assert isinstance(arr, np.ndarray), arr
            assert (arr.shape) == (299, 299), arr.shape
            if "8A" in band.path:
                assert band.band_id == "B8A", band.band_id
                any_8a = True

        assert any_8a

        for band_id, file_path in zip(img.band_ids, img.file_paths, strict=True):
            assert isinstance(band_id, str), band_id
            assert isinstance(file_path, str), type(file_path)
            raster = img[band_id]
            assert raster.band_id is not None, raster.band_id
            raster = img[file_path]
            assert raster.band_id is not None, raster.band_id


@print_function_name
def test_iteration_base_image_collection():

    collection = sg.ImageCollection(path_sentinel, level=None, res=10)
    assert isinstance(collection, sg.ImageCollection), type(collection)

    assert len(collection.images) == 4, len(collection.images)
    assert len(collection) == 4, len(collection)
    file_paths = list(sorted({band.path for img in collection for band in img}))
    assert len(file_paths) == 55, len(file_paths)

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

        for band in img:
            assert isinstance(band, sg.Band), band
            arr = band.load().values
            assert isinstance(arr, np.ndarray), arr
            print(band.__dict__)
            assert (arr.shape) == (299, 299) or (arr.shape) == (200, 200), (
                arr.shape,
                band,
            )


@print_function_name
def test_convertion():

    assert sg.raster.image_collection.Sentinel2Config.masking
    assert sg.Sentinel2Band.masking
    assert sg.Sentinel2Image.masking
    assert sg.Sentinel2Collection.masking

    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=100)
    band = collection[0]["B02"]

    assert band.res == 100, band.res

    arr = band.load().values
    _from_array = sg.Band(arr, crs=band.crs, bounds=band.bounds)
    assert (shape := _from_array.values.shape) == (29, 29), shape

    gdf = band.to_geopandas(column="val")
    from_geopandas = sg.Band.from_geopandas(gdf, res=band.res)

    mask_as_geopandas = band.copy()
    mask_as_geopandas.values = band.values.mask
    mask_as_geopandas = mask_as_geopandas.to_geopandas()[lambda x: x["value"] == True]
    mask_as_geopandas

    assert len(mask_as_geopandas) == 30, len(mask_as_geopandas)
    mask_gdf_intersection = (
        sg.clean_overlay(mask_as_geopandas, gdf).pipe(sg.buff, -0.1).pipe(sg.buff, 0.1)
    )
    assert not len(mask_gdf_intersection), mask_gdf_intersection

    e = sg.explore(
        mask_as_geopandas=mask_as_geopandas,
        from_geopandas=from_geopandas.to_geopandas(),
        band=band.to_geopandas(),
        gdf=gdf,
    )
    assert len(e._gdfs) == 4
    n_rows = [len(gdf) for gdf in e._gdfs]
    assert n_rows == [30, 821, 791, 791], n_rows
    assert (shape := from_geopandas.values.shape) == (29, 29), shape


@print_function_name
def not_test_to_xarray():
    import xarray as xr

    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", res=10)
    collection.load()
    for img in collection:
        xarr = img.to_xarray()
        assert xarr.shape == (12, 299, 299), xarr.shape
        assert xarr.isnull().sum(), img.values.mask.sum()
        assert xarr.isnull().sum() == img.values.mask.sum(), (
            xarr.isnull().sum(),
            img.values.mask.sum(),
        )

        for band in img:
            band.load()
            xarr = band.to_xarray()
            assert xarr.shape == (299, 299), xarr.shape
            assert xarr.isnull().sum(), band.values.mask.sum()
            assert xarr.isnull().sum() == band.values.mask.sum(), (
                xarr.isnull().sum(),
                band.values.mask.sum(),
            )

    xarr = collection.to_xarray()

    print(xarr)
    assert isinstance(xarr, xr.Dataset)
    assert len(xarr) == 2
    assert list(xarr.dims) == ["x", "y", "band_id", "date"], list(xarr.dims)
    assert len(xarr.x) == 598, len(xarr.x)
    assert len(xarr.y) == 598, len(xarr.y)
    assert len(xarr.band_id) == 12, len(xarr.band_id)
    assert len(xarr.date) == 3, len(xarr.date)

    print(xarr.date)

    xarr2 = xarr.where(
        (xarr.date >= "2022") & (xarr.band_id.isin(["B01", "B02", "B04"])),
        drop=True,
    )
    print(xarr2)
    assert isinstance(xarr2, xr.Dataset)
    assert len(xarr2.x) == 598, len(xarr2.x)
    assert len(xarr2.y) == 598, len(xarr2.y)
    assert len(xarr2.band_id) == 3, len(xarr2.band_id)
    assert len(xarr.date) == 2, len(xarr.date)
    assert len(xarr2) == 1

    collection[0].name = "name1"

    xarr = collection.to_xarray(by="name")

    print(xarr)
    assert isinstance(xarr, xr.Dataset)
    assert len(xarr) == 2
    assert list(xarr.dims) == ["x", "y", "band_id", "name"], list(xarr.dims)
    assert len(xarr.x) == 598, len(xarr.x)
    assert len(xarr.y) == 598, len(xarr.y)
    assert len(xarr.band_id) == 12, len(xarr.band_id)
    assert len(xarr2.name) == 2, len(xarr2.name)

    xarr2 = xarr.where(
        (xarr.name == "name1") & (xarr.band_id.isin(["B01", "B02", "B04"])),
        drop=True,
    )
    print(xarr2)
    assert isinstance(xarr2, xr.Dataset)
    assert len(xarr2) == 1
    assert len(xarr2.x) == 598, len(xarr2.x)
    assert len(xarr2.y) == 598, len(xarr2.y)
    assert len(xarr2.band_id) == 3, len(xarr2.band_id)
    assert len(xarr2.name) == 1, len(xarr2.name)


@print_function_name
def test_clip():
    collection = sg.Sentinel2Collection(path_sentinel, level="L2A", nodata=-1, res=10)

    # sg.explore(collection, browser=True)

    collection = collection.filter(bands=["B02"])

    i = 0
    for img in collection:
        img._test_index = i
        i += 1

    paths_here = [img.path for img in collection]

    collection.load()

    collection[0][0].values[:] = 1
    collection[0][0].values[:].mask = False

    assert [img._test_index for img in collection] == [0, 1, 2], [
        img._test_index for img in collection
    ]

    assert [img.path for img in collection] == paths_here, (
        paths_here,
        [img.path for img in collection],
    )

    mask = collection[0].centroid.buffer(50)
    gdfs = {}

    clipped = collection.clip(mask, dropna=False)

    assert [img._test_index for img in clipped] == [0, 1, 2], [
        img._test_index for img in clipped
    ]

    clipped = collection.clip(mask)

    assert [img._test_index for img in clipped] == [0, 1], [
        img._test_index for img in clipped
    ]

    band1 = clipped[0][0]
    shape = band1.values.shape
    assert shape == (299, 299), shape
    values = band1.copy().values
    values[values.mask] = 0
    values[np.isnan(values.data)] = 0
    sum_ = np.sum(values.data)
    print(sum_, band1.date, Path(band1.path).stem)
    print(band1.values.mask.sum())
    print(band1.values.data)
    gdfs[band1.date] = band1.to_geopandas()
    assert sum_ == 73, (sum_, values)

    band2 = clipped[1][0]
    shape = band2.values.shape
    assert shape == (299, 299), shape
    values = band2.copy().values
    values[values.mask] = 0
    values[np.isnan(values.data)] = 0
    sum_ = np.sum(values.data)
    print(sum_, band2.date, Path(band2.path).stem)
    print(values.mask.sum())
    print(values.data)
    print(values.data[130:150, 130:150])
    gdfs[band2.date] = band2.to_geopandas()

    sg.explore(**gdfs, msk=sg.to_gdf(mask, band1.crs).buffer(100))

    assert sum_ == 117963, (sum_, values)


def get_metadata_df(
    root_dirs: list[str],
    processes: int,
    band_endswith: str = ".tif",
) -> pd.DataFrame:
    """Get Sentine2 metadata to use to set attributes on Images and Bands.

    This file should be written to disc, but that won't work on github action.
    """
    all_relevant_file_paths = set()
    for root in root_dirs:
        root = str(root).rstrip("/")
        for file_path in sg.helpers.get_all_files(root):
            if file_path.endswith(band_endswith):
                all_relevant_file_paths.add(file_path)
                parent = str(Path(file_path).parent)
                all_relevant_file_paths.add(parent)

    df: list[dict] = sg.Parallel(processes, backend="threading").map(
        _get_metadata_for_one_path,
        all_relevant_file_paths,
        kwargs={"band_endswith": band_endswith},
    )

    if not df or all(not x for x in df):
        return

    df = pd.DataFrame(df).set_index("file_path")

    df = df.loc[lambda x: x.index.notna()]

    # df.to_parquet(metadata_df_path)
    return df


def _get_metadata_for_one_path(file_path: str, band_endswith: str) -> dict:
    print(file_path)
    try:
        if band_endswith in file_path:
            try:
                obj = sg.Sentinel2Band(file_path, res=None)
            except KeyError as e:
                if "sentinel" in file_path.lower() and "SCL" in file_path:
                    return {}
            except Exception as e:
                if "sentinel" in file_path.lower():
                    raise e
                obj = sg.Band(file_path, res=None)
        else:
            try:
                obj = sg.Sentinel2Image(file_path, res=None)
            except Exception as e:
                if "sentinel" in file_path.lower() and "SCL" not in file_path:
                    raise e
                try:
                    obj = sg.Image(file_path, res=None)
                except Exception as e:
                    print(e)
                    print(file_path)
                    return {}
        try:
            metadata = {
                "file_path": file_path,
                "bounds": obj.bounds,
                "crs": str(pyproj.CRS(obj.crs).to_string()),
            }
        except Exception as e:
            print()
            print(e)
            print(file_path)
            return {}
        for key in obj.metadata_attributes:
            try:
                metadata[key] = getattr(obj, key)
            except AttributeError as e:
                if "SCL" not in file_path:
                    raise e
        return metadata
    except RasterioIOError as e:
        print("RasterioIOError", e, file_path)
        return {}
    except CRSError as e:
        print("pyproj.CRSError", e, file_path)
        return {}


def main():
    test_merge()
    test_ndvi()
    test_explore()
    test_pixelwise()
    test_ndvi_predictions()
    test_clip()
    test_convertion()
    test_metadata_attributes()
    test_bbox()
    test_collection_from_list_of_path()
    test_indexing()
    test_regexes()
    test_date_ranges()
    test_single_banded()
    test_buffer()
    test_iteration()
    test_gradient()
    test_iteration_base_image_collection()
    test_groupby()
    test_cloud()
    test_concat_image_collections()
    test_with_mosaic()
    test_masking()
    test_zonal()
    test_merge()
    test_plot_pixels()
    not_test_to_xarray()
    not_test_sample()
    not_test_sample()
    not_test_sample()
    not_test_sample()
    not_test_sample()
    not_test_sample()


if __name__ == "__main__":
    main()
