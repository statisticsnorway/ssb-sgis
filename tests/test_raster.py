# %%
import sys
from pathlib import Path

import numpy as np
from IPython.display import display
from shapely import box


src = str(Path(__file__).parent.parent) + "/src"
testdata = str(Path(__file__).parent.parent) + "/tests/testdata"

sys.path.insert(0, src)

import sgis as sg


path_singleband = testdata + "/dtm_10.tif"
path_two_bands = testdata + "/dtm_10_two_bands.tif"
sg.Raster.dapla = False


def test_raster():
    test_convertion()

    test_zonal()

    test_transform()

    test_indexes_and_shape()

    test_elevation()

    test_res()

    test_to_crs()


def test_transform():
    gdf = sg.random_points(10)
    transform1 = sg.Raster.get_transform_from_bounds(gdf, shape=(1, 1))

    # should also work with geoseries, shapely geometry and tuple
    transform = sg.Raster.get_transform_from_bounds(gdf.geometry, shape=(1, 1))
    assert transform1 == transform
    transform = sg.Raster.get_transform_from_bounds(gdf.unary_union, shape=(1, 1))
    assert transform1 == transform
    transform = sg.Raster.get_transform_from_bounds(gdf.total_bounds, shape=(1, 1))
    assert transform1 == transform
    transform = sg.Raster.get_transform_from_bounds(
        tuple(gdf.total_bounds), shape=(1, 1)
    )
    assert transform1 == transform


def test_elevation():
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

    # creating a simple Raster with a resolution of 10 (50 / width or height).
    r = sg.ElevationRaster.from_array(arr, crs=None, bounds=(0, 0, 50, 50))
    print(r.res)
    gradient = r.gradient(copy=True)
    print(gradient.array)
    assert np.max(gradient.array) == 1, gradient.array

    degrees = r.degrees(copy=True)
    print(degrees.array)

    assert np.max(degrees.array) == 45, degrees.array

    r = sg.ElevationRaster.from_path(path_two_bands).load()
    assert r.shape == (2, 201, 201)

    max_ = int(np.nanmax(r.array))
    gradient = r.gradient(copy=True)
    assert max_ == int(np.nanmax(r.array))
    assert int(np.nanmax(gradient.array)) == 17
    print(np.nanmax(gradient.array))
    degrees = r.degrees()
    assert int(np.nanmax(degrees.array)) == 86
    print(np.nanmax(degrees.array))
    assert len(degrees.shape) == 3
    print(degrees.shape)
    print(degrees.array.shape)
    display(degrees.to_gdf())
    gdf = degrees.to_gdf()
    sg.explore(gdf[gdf["band"] == 1], "value")
    sg.explore(gdf[gdf["band"] == 2], "value")


def test_zonal():
    r = sg.Raster.from_path(path_singleband, band_indexes=1).load()
    gdf = sg.make_grid(r.bounds, 100, crs=r.crs)

    gdf.index = [np.random.choice([*"abc"]) for _ in range(len(gdf))]

    zonal_stats = r.zonal(gdf, aggfunc=[sum, np.mean, "median"], raster_calc_func=None)
    assert gdf.index.equals(zonal_stats.index)

    if __name__ == "__main__":
        display(zonal_stats)
        display(gdf)
        sg.explore(r.to_gdf(), "value")
        sg.explore(zonal_stats, "mean")


def test_convertion():
    r = sg.Raster.from_path(path_singleband, band_indexes=1).load()

    r = sg.Raster.from_dict(r.meta)
    print(r.__dict__)
    ss

    arr = r.array
    r_from_array = sg.Raster.from_array(arr, meta=r.meta)

    gdf = r.to_gdf(column="val")

    # multiple columns give multiple bands
    gdf["val_x2"] = gdf["val"] * -1
    r_from_gdf = sg.Raster.from_gdf(gdf, column=["val", "val_x2"], res=r.res)
    assert r_from_gdf.shape == (2, 201, 201)

    r_from_gdf = sg.Raster.from_gdf(gdf, column="val", res=r.res)
    assert r_from_gdf.shape == (201, 201)
    assert r_from_gdf.name == "val"

    # putting three 2-dimensional array on top of each other
    r3 = sg.Raster.from_array(
        np.array([r.array, r_from_array.array, r_from_gdf.array]), meta=r.meta
    )
    assert r3.count == 3, r3.count
    assert r3.shape == (3, 201, 201), r3.shape
    assert r3.bounds == r.bounds

    r3_as_gdf = r3.to_gdf()
    assert r3_as_gdf["band"].isin([1, 2, 3]).all()
    assert all(x in list(r3_as_gdf["band"]) for x in [1, 2, 3])


def test_res():
    r = sg.Raster.from_path(path_singleband, band_indexes=1)
    mask_utm33 = sg.to_gdf(box(*r.bounds), crs=r.crs)

    for _ in range(5):
        r = sg.Raster.from_path(path_singleband, band_indexes=1)
        mask_utm33["geometry"] = mask_utm33.sample_points(5).buffer(100)
        r = r.clip(mask_utm33, crop=True)
        assert r.res == (10, 10)


def test_to_crs():
    r = sg.Raster.from_path(path_singleband, band_indexes=1)
    mask_utm33 = sg.to_gdf(box(*r.bounds), crs=r.crs).centroid.buffer(50)
    r = r.clip(mask=mask_utm33)
    r = r.to_crs(25832)

    mask_utm32 = mask_utm33.to_crs(25832)
    assert r.to_gdf().intersects(mask_utm32.unary_union).any()

    r = r.to_crs(25833)
    assert r.to_gdf().intersects(mask_utm33.unary_union).any()

    original = sg.Raster.from_path(path_singleband, band_indexes=1)
    assert original.to_gdf().intersects(r.unary_union).any()


def test_indexes_and_shape():
    # specifying single index is only thing that returns 2dim ndarray
    r = sg.Raster.from_path(path_singleband, band_indexes=1)
    assert len(r.shape) == 2, r.shape
    assert r.shape == (201, 201), r.shape

    r = sg.Raster.from_path(path_singleband, band_indexes=(1,))
    assert len(r.shape) == 3, r.shape
    assert r.shape[0] == 1, r.shape

    r = sg.Raster.from_path(path_singleband)
    assert len(r.shape) == 3, r.shape
    assert r.shape[0] == 1, r.shape

    r2 = sg.Raster.from_path(path_two_bands)
    assert len(r2.shape) == 3, r2.shape
    assert r2.shape[0] == 2, r2.shape
    r2 = r2.load()
    assert len(r2.shape) == 3, r2.shape
    assert r2.shape[0] == 2, r2.shape


def not_test_raster():
    testpath = testdata + "/test.tif"
    r = sg.Raster.from_path(path_singleband, band_indexes=1).load()
    r.write_tif(testpath)


def save_two_band_image():
    r = sg.Raster.from_path(path_singleband, band_indexes=1).load()
    r.array[r.array < 0] = 0

    r2 = r * -1
    r2.array = np.array([r.array, r2.array])
    assert len(r2.shape) == 3, r2.shape
    assert r2.shape[0] == 2, r2.shape
    r2.plot()
    r2.write_tif(path_two_bands)

    r2 = sg.Raster.from_path(path_two_bands)
    assert r2.shape[0] == 2, r2.shape
    r2 = r2.load()
    assert r2.shape[0] == 2, r2.shape
    r2.plot()


if __name__ == "__main__":
    # save_two_band_image()

    test_raster()

    not_test_raster()

    print("ferdig")
