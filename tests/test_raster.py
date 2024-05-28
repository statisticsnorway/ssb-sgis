# %%
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from IPython.display import display
from shapely import box

src = str(Path(__file__).parent.parent) + "/src"
testdata = str(Path(__file__).parent.parent) + "/tests/testdata/raster"

sys.path.insert(0, src)

import sgis as sg

path_singleband = testdata + "/dtm_10.tif"
path_two_bands = testdata + "/dtm_10_two_bands.tif"


def test_sample():
    r = sg.Raster.from_path(path_two_bands)
    sample = r.sample(10)
    sample2 = r.sample(10)
    sg.explore(r.load().to_gdf(), sample.to_gdf(), sample2.to_gdf())


def test_transform():
    gdf = sg.random_points(10)
    transform1 = sg.get_transform_from_bounds(gdf, shape=(1, 1))

    # should also work with geoseries, shapely geometry and tuple
    transform = sg.get_transform_from_bounds(gdf.geometry, shape=(1, 1))
    assert transform1 == transform
    transform = sg.get_transform_from_bounds(gdf.unary_union, shape=(1, 1))
    assert transform1 == transform
    transform = sg.get_transform_from_bounds(gdf.total_bounds, shape=(1, 1))
    assert transform1 == transform
    transform = sg.get_transform_from_bounds(tuple(gdf.total_bounds), shape=(1, 1))
    assert transform1 == transform


@pytest.mark.skip(reason="This test requires GUI")
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

    # creating a simple Raster with a resolution of 10 (50 / width or height).
    r = sg.Raster.from_array(arr, crs=None, bounds=(0, 0, 50, 50))
    gradient = r.gradient(copy=True)
    assert np.max(gradient.values) == 1, gradient.values

    degrees = r.gradient(degrees=True, copy=True)

    assert np.max(degrees.values) == 45, np.max(degrees.values)

    r = sg.Raster.from_path(path_singleband, indexes=1, nodata=0).load()
    # sg.explore(r.to_gdf(), "value")
    assert int(np.min(r.values)) == 0, np.min(r.values)
    degrees = r.gradient(degrees=True, copy=True)
    assert int(np.max(degrees.values)) == 75, np.max(degrees.values)
    gradient = r.gradient(copy=True)
    assert int(np.max(gradient.values)) == 3, np.max(gradient.values)

    r = sg.Raster.from_path(path_two_bands, indexes=1, nodata=0).load()
    # sg.explore(r.to_gdf(), "value")
    assert r.shape == (101, 101), r.shape
    gradient = r.gradient(copy=True)
    assert int(np.max(gradient.values)) == 3, np.max(gradient.values)

    degrees = r.gradient(degrees=True, copy=True)
    assert int(np.max(degrees.values)) == 75, np.max(degrees.values)

    r = sg.Raster.from_path(path_two_bands, indexes=(1, 2))
    assert r.shape == (2, 101, 101), r.shape

    degrees = r.load().gradient(degrees=True)

    assert int(np.nanmax(degrees.values)) == 75, int(np.nanmax(degrees.values))
    print(degrees.shape)
    print(degrees.values.shape)
    assert len(degrees.shape) == 3
    gdf = degrees.to_gdf()
    if __name__ == "__main__":
        sg.explore(gdf[gdf["indexes"] == 1], "value")
        sg.explore(gdf[gdf["indexes"] == 2], "value")

    max_ = int(np.nanmax(r.values))
    gradient = r.gradient(copy=True)
    gradient.plot()
    assert max_ == int(np.nanmax(r.values))
    assert int(np.nanmax(gradient.values)) == 6, int(np.nanmax(gradient.values))
    print(np.nanmax(gradient.values))


def test_zonal():
    r = sg.Raster.from_path(path_singleband, indexes=1).load()
    gdf = sg.make_grid(r.bounds, 100, crs=r.crs)

    gdf.index = [np.random.choice([*"abc"]) for _ in range(len(gdf))]

    zonal_stats = r.zonal(gdf, aggfunc=[sum, np.mean, "median"], array_func=None)
    assert gdf.index.equals(zonal_stats.index)

    if __name__ == "__main__":
        display(zonal_stats)
        display(gdf)


def test_resize():
    r = sg.Raster.from_path(path_singleband, nodata=0).load()
    assert r.min() == 0
    assert r.shape == (201, 201), r.shape
    assert r.res == 10, r.res

    r = sg.Raster.from_path(path_singleband, res=20).load()
    assert int(r.res) == 20, r.res
    assert r.shape == (100, 100), r.shape

    r = sg.Raster.from_path(path_singleband, res=30).load()
    assert r.shape == (67, 67), r.shape
    assert r.res == 30, r.res

    r = sg.Raster.from_path(path_singleband, indexes=1, nodata=0, res=20).load()
    assert r.shape == (100, 100), r.shape
    assert r.min() == 0, r.min()

    r = sg.Raster.from_path(path_singleband, nodata=0, res=10).clip(r.bounds)
    print(type(r.values))
    print(r.values)
    assert r.res == 10, r.res
    assert r.nodata == 0, r.nodata
    assert r.shape == (201, 201), r.shape
    assert r.min() == 0, r.min()

    r = sg.Raster.from_path(path_singleband, res=20).clip(r.bounds)
    assert int(r.res) == 20, r.res
    assert r.shape == (100, 100), r.shape
    assert r.nodata == -32767, r.nodata
    assert r.min() == 0, r.min()

    r = sg.Raster.from_path(path_singleband, res=20).clip(
        sg.to_shapely(r.bounds).buffer(30)
    )
    assert r.nodata == -32767, r.nodata
    assert r.min() == -32767, r.min()

    r = sg.Raster.from_path(path_singleband, res=30).clip(r.bounds)
    # sg.explore(r.to_gdf(), "value")
    # assert r.shape == (1, 67, 67), r.shape
    assert r.res == 30, r.res


def test_clip():
    r = sg.Raster.from_path(path_singleband, nodata=-999)

    out_of_bounds = sg.to_gdf([0, 0, 1, 1], crs=r.crs)
    clipped = r.copy().clip(out_of_bounds)
    assert not np.size(clipped.values), clipped.values

    circle = r.unary_union.centroid.buffer(100)

    clipped_from_path = r.copy().clip(circle)
    assert clipped_from_path.shape == (1, 20, 20), clipped_from_path.shape

    clipped_memoryfile = r.copy().load().clip(circle)
    assert np.array_equal(clipped_memoryfile.values, clipped_from_path.values)
    assert clipped_memoryfile.equals(clipped_from_path)


def test_clip_res():
    r = sg.Raster.from_path(path_singleband, nodata=-999)
    assert r.shape == (201, 201), r.shape

    res = 10
    assert r.res == res

    # whole = r.copy().load()

    # masks outside should return nodata on area outside of mask
    southeast_corner = sg.to_gdf(r.bounds[:2], crs=r.crs)
    square_in_corner = sg.to_gdf(southeast_corner.buffer(400).total_bounds, crs=r.crs)

    assert r.values is None

    def general_assertions(raster, masked: bool):
        if not masked:
            assert raster.values.min() == -999, raster.values.min()
        else:
            assert raster.values.min() == 132, raster.values.min()

        assert int(raster.values.max()) == 190, raster.values.max()

        # has correct res
        assert int(area := raster.to_gdf().area.median()) == res * res, area**0.5

    def masked_and_boundless():
        clipped = r.copy().clip(square_in_corner, masked=True, boundless=True)
        print("masked_and_boundless")
        sg.explore(clipped.to_gdf())
        return
        general_assertions(clipped, masked=True)
        print(clipped.shape)

        clipped_from_memfile = (
            r.copy().load().clip(square_in_corner, masked=True, boundless=True)
        )
        general_assertions(clipped_from_memfile, masked=True)

    def masked_not_boundless():
        clipped = r.copy().clip(square_in_corner, masked=True, boundless=False)
        print("masked_not_boundless")
        sg.explore(clipped.to_gdf())
        return
        general_assertions(clipped, masked=True)
        print(clipped.shape)

        clipped_from_memfile = (
            r.copy().load().clip(square_in_corner, masked=True, boundless=False)
        )
        general_assertions(clipped_from_memfile, masked=True)

    def not_masked_not_boundless():
        clipped = r.copy().clip(square_in_corner, masked=False, boundless=False)
        print("not_masked_not_boundless")
        sg.explore(clipped.to_gdf())
        return
        general_assertions(clipped, masked=False)

        clipped_from_memfile = (
            r.copy().load().clip(square_in_corner, masked=False, boundless=False)
        )
        general_assertions(clipped_from_memfile, masked=False)

    def not_masked_but_boundless():
        clipped = r.copy().clip(square_in_corner, masked=False, boundless=True)
        print("not_masked_but_boundless")
        sg.explore(clipped.to_gdf())
        return
        general_assertions(clipped, masked=False)

        intersected = clipped.to_gdf().clip(square_in_corner)
        same_area = (
            int(clipped.area.sum())
            == int(square_in_corner.area.sum())
            == int(intersected.area.sum())
        )

        assert same_area, (
            int(clipped.area.sum()),
            int(square_in_corner.area.sum()),
            int(intersected.area.sum()),
        )

        clipped_from_memfile = (
            r.copy().load().clip(square_in_corner, masked=False, boundless=True)
        )
        general_assertions(clipped_from_memfile, masked=False)

        intersected = clipped_from_memfile.to_gdf().clip(square_in_corner)
        same_area = (
            int(clipped_from_memfile.area.sum())
            == int(square_in_corner.area.sum())
            == int(intersected.area.sum())
        )
        assert same_area, (
            int(clipped_from_memfile.area.sum()),
            int(square_in_corner.area.sum()),
            int(intersected.area.sum()),
        )

    not_masked_not_boundless()
    masked_not_boundless()
    masked_and_boundless()
    not_masked_but_boundless()


def test_convertion():
    r = sg.Raster.from_path(path_singleband, indexes=1).load()
    assert isinstance(r, sg.Raster)

    arr = r.values
    r_from_array = sg.Raster.from_array(arr, **r.profile)
    assert (shape := r_from_array.shape) == (201, 201), shape

    gdf = r.to_gdf(column="val")
    r_from_gdf = sg.Raster.from_gdf(gdf, columns="val", res=r.res)
    sg.explore(r_from_gdf.to_gdf(), r.to_gdf(), gdf)

    assert (shape := r_from_gdf.shape) == (201, 200), shape
    assert r_from_gdf.name == "val"

    # multiple columns give multiple bands
    gdf["val_x2"] = gdf["val"] * -1
    r_from_gdf = sg.Raster.from_gdf(gdf, columns=["val", "val_x2"], res=r.res)
    assert (shape := r_from_gdf.shape) == (2, 201, 200), shape

    # # putting three 2-dimensional array on top of each other
    # r3 = sg.Raster.from_array(
    #     np.array([r.values, r_from_array.values, r_from_gdf.values]), **r.profile
    # )
    # assert r3.count == 3, r3.count
    # assert r3.shape == (3, 201, 201), r3.shape
    # assert r3.bounds == r.bounds

    # r3_as_gdf = r3.to_gdf()
    # assert r3_as_gdf["indexes"].isin([1, 2, 3]).all()
    # assert all(x in list(r3_as_gdf["indexes"]) for x in [1, 2, 3])


# def test_res():
#     r = sg.Raster.from_path(path_singleband, indexes=1)
#     mask_utm33 = sg.to_gdf(box(*r.bounds), crs=r.crs)

#     for _ in range(3):
#         r = sg.Raster.from_path(path_singleband, indexes=1)
#         mask_utm33["geometry"] = mask_utm33.sample_points(5).buffer(100)
#         r = r.clip(mask_utm33)

#         assert int(round(r.res, 0)) == 10, r.res


def test_to_crs():
    r = sg.Raster.from_path(path_singleband, indexes=1)
    mask_utm33 = sg.to_gdf(box(*r.bounds), crs=r.crs).centroid.buffer(50)
    r = r.clip(mask=mask_utm33)
    r = r.to_crs(25832)

    mask_utm32 = mask_utm33.to_crs(25832)
    assert r.to_gdf().intersects(mask_utm32.unary_union).any()

    r = r.to_crs(25833)
    assert r.to_gdf().intersects(mask_utm33.unary_union).any()

    original = sg.Raster.from_path(path_singleband, indexes=1).load()
    assert original.to_gdf().intersects(r.unary_union).any()


def test_indexes_and_shape():
    # specifying single index is only thing that returns 2dim ndarray
    r = sg.Raster.from_path(path_singleband, indexes=1)
    assert len(r.shape) == 2, r.shape
    assert r.shape == (201, 201), r.shape

    r = sg.Raster.from_path(path_singleband)
    assert len(r.shape) == 3, r.shape
    assert r.shape == (1, 201, 201), r.shape

    r = sg.Raster.from_path(path_singleband, indexes=(1,))
    assert len(r.shape) == 3, r.shape
    assert r.shape == (1, 201, 201), r.shape

    r2 = sg.Raster.from_path(path_two_bands, indexes=(1, 2))
    assert len(r2.shape) == 3, r2.shape
    assert r2.shape[0] == 2, r2.shape
    r2 = r2.load()
    assert len(r2.shape) == 3, r2.shape
    assert r2.shape[0] == 2, r2.shape


def test_xarray():
    r = sg.Raster.from_path(path_two_bands)
    xarr = r.load().to_xarray()
    print(xarr)

    r = sg.Raster.from_path(path_singleband, indexes=1)
    xarr = r.load().to_xarray()
    assert isinstance(xarr, xr.DataArray)

    print(r.bounds)
    print(xarr)
    print(xarr.to_dataset())

    print(xarr * 2)


def save_two_band_image():
    r = sg.Raster.from_path(path_singleband, indexes=1)

    mask = sg.to_gdf(r.unary_union, crs=r.crs).buffer(-500)
    r = r.clip(mask)
    r.values[r.values < 0] = 0

    r2 = r * -1
    r2.values = np.array([r.values, r2.values])
    assert len(r2.shape) == 3, r2.shape
    assert r2.shape[0] == 2, r2.shape
    r2.plot()
    r = sg.Raster.from_path(path_two_bands).load()
    r2.write(path_two_bands)

    r2 = sg.Raster.from_path(path_two_bands)
    assert r2.shape[0] == 2, r2.shape
    r2 = r2.load()
    assert r2.shape[0] == 2, r2.shape
    r2.plot()


def not_test_write():
    r = sg.Raster.from_path(path_singleband, indexes=1)

    mask = sg.to_gdf(r.unary_union, crs=r.crs).buffer(-500)
    r = r.clip(mask)
    r.values[r.values < 0] = 0

    r2 = r * -1
    r2.values = np.array([r.values, r2.values])
    assert len(r2.shape) == 3, r2.shape
    assert r2.shape[0] == 2, r2.shape
    r2.write(f"{testdata}/test.tif")

    r3 = sg.Raster.from_path(f"{testdata}/test.tif", indexes=(1, 2)).load()
    assert r3.shape == r2.shape
    assert int(np.mean(r3.values)) == int(np.mean(r2.values))


if __name__ == "__main__":
    # save_two_band_image()

    if 0:
        path = "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/raster/dtm_10.tif"
        raster = sg.Raster.from_path(path)
        raster.plot()
        raster

    test_clip_res()
    test_resize()
    test_clip()
    test_zonal()
    test_to_crs()
    # test_res()

    test_xarray()
    test_convertion()

    test_transform()
    test_gradient()

    test_indexes_and_shape()

    test_sample()
    not_test_write()

    print("ferdig")

# %%
