# %%
import multiprocessing
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import shapely
import xarray as xr
from IPython.display import display
from shapely import box


src = str(Path(__file__).parent.parent) + "/src"
testdata = str(Path(__file__).parent.parent) + "/tests/testdata/raster"

sys.path.insert(0, src)

import sgis as sg


path_singleband = testdata + "/dtm_10.tif"
path_two_bands = testdata + "/dtm_10_two_bands.tif"
path_sentinel = testdata + "/sentinel2"
sg.GeoDataCube._test = True


def x2(x):
    return x * 2


def test_xdataset():
    cube = (
        sg.GeoDataCube.from_root(testdata, raster_type=sg.Sentinel2)
        .query("date.notna()")
        .load()
    )

    print(cube)
    xdataset = cube.to_xarray()
    assert isinstance(xdataset, xr.Dataset)
    print(xdataset)
    print(xdataset.attrs)
    print(dir(xdataset))


def test_query():
    cube = sg.GeoDataCube.from_root(
        testdata, endswith=".tif", raster_type=sg.ElevationRaster
    )

    for i in cube.date.unique():
        c1 = cube.query(f"date == {i}")
        assert isinstance(c1, sg.GeoDataCube)


def test_shape():
    cube = sg.GeoDataCube.from_root(
        testdata, endswith=".tif", raster_type=sg.ElevationRaster
    ).query("subfolder != 'sentinel2'")
    cube = cube.load(res=10)
    assert (cube.res == (10, 10)).all(), cube.res
    cube = cube.load(res=30)

    c = cube.unary_union.centroid.buffer(100)
    cube = cube.clip(c)
    assert (cube.res == (10, 10)).all(), cube.res
    assert (cube.shape == (20, 20)).all(), cube.shape

    print(cube.arrays)
    print(cube.run_raster_method("min"))
    assert min(cube.run_raster_method("min")) == -999, min(
        cube.run_raster_method("min")
    )

    cube = cube.clip(c, res=20)
    assert (cube.res == (20, 20)).all(), cube.res
    assert (cube.shape == (10, 10)).all(), cube.shape


def test_copy():
    cube = sg.GeoDataCube.from_root(
        testdata, endswith=".tif", raster_type=sg.ElevationRaster
    )

    assert cube.arrays.isna().all()
    cube2 = cube.load().load().load()
    assert cube.arrays.isna().all()
    assert cube2.arrays.notna().all()

    print(cube2.max())
    cube3 = (cube2.pool(2) * 2).execute()
    print(cube2.max())
    print(cube3.max())
    assert int(cube3.max()) == int(cube2.max() * 2), (
        int(cube3.max()),
        int(cube2.max() * 2),
    )


def test_elevation():
    cube = (
        sg.GeoDataCube.from_root(
            testdata,
            endswith=".tif",
            raster_type=sg.ElevationRaster,
            nodata=0,
        )
        .query("subfolder != 'sentinel2'")
        .load()
    )

    degrees = cube.copy().gradient(degrees=True)
    assert int(degrees.max().max()) == 89, degrees.max().max()

    degrees = cube.copy().run_raster_method("gradient", degrees=True)

    gradient = cube.copy().gradient()
    assert int(gradient.max().max()) == 366, gradient.max().max()

    gradient = cube.copy().run_raster_method("gradient")


def test_sentinel():
    cube = sg.GeoDataCube.from_root(
        path_sentinel, endswith=".tif", raster_type=sg.Sentinel2
    ).query("name.notna()")

    ndvi = cube.ndvi(band_name_red="B4", band_name_nir="B8")
    print(ndvi)

    ndvi_pooled = cube.pool(3).ndvi(band_name_red="B4", band_name_nir="B8").execute()
    print(ndvi_pooled)

    print(ndvi.min(), ndvi_pooled.min())
    print(ndvi.max(), ndvi_pooled.max())

    assert ndvi.equals(ndvi_pooled), ndvi.equals(ndvi_pooled, verbose=True)


def not_test_df():
    df_path = testdata + "/cube_df.parquet"
    try:
        os.remove(df_path)
    except FileNotFoundError:
        pass
    try:
        cube = sg.GeoDataCube.from_root(testdata, endswith=".tif").explode()
        df = cube._prepare_df_for_parquet()
        df["my_idx"] = range(len(df))
        df.to_parquet(df_path)
        cube_from_cube_df = sg.GeoDataCube.from_cube_df(df_path).explode()
        display(cube_from_cube_df.df)
        assert "my_idx" in cube_from_cube_df.df
        assert hasattr(cube_from_cube_df, "_from_cube_df")
        assert cube_from_cube_df.boxes.intersects(cube.unary_union).all()

        cube_from_cube_df = sg.GeoDataCube.from_root(
            testdata, endswith=".tif"
        ).explode()
        assert hasattr(cube_from_cube_df, "_from_cube_df")
        assert cube_from_cube_df.boxes.intersects(cube.unary_union).all()
        os.remove(df_path)
    except Exception as e:
        os.remove(df_path)
        raise e


def test_from_root():
    import glob

    files = [file for file in glob.glob(str(Path(testdata)) + "/*") if ".tif" in file]
    cube = sg.GeoDataCube.from_paths(files)

    cube = sg.GeoDataCube.from_root(testdata, endswith=".tif").explode()
    assert len(cube) == 22, cube
    display(cube)

    cube = sg.GeoDataCube.from_root(testdata, regex=r"\.tif$").explode()
    assert len(cube) == 22, cube
    display(cube)


def test_to_gdf():
    query = "name.str.contains('two_bands')"
    cube = (
        sg.GeoDataCube.from_root(testdata, endswith=".tif", crs=25833)
        .explode()
        .query(query)
        .load()
    )
    assert len(cube) == 2, len(cube)

    cube.df["new_col"] = ["a", "b"]
    cube.df["new_col2"] = [1, 3]

    gdf = cube.to_gdf()

    assert "new_col" in gdf and "new_col2" in gdf, gdf.columns

    gdf = cube.pool(2).to_gdf().execute()

    assert "new_col" in gdf and "new_col2" in gdf, gdf.columns


def test_to_crs():
    query = "name.str.contains('two_bands')"
    cube = (
        sg.GeoDataCube.from_root(testdata, endswith=".tif", crs=25833, nodata=0)
        .explode()
        .query(query)
    )
    assert len(cube) == 2, len(cube)

    merged_rio = cube.merge().to_gdf()

    merged_xarr = cube.load().merge().to_gdf()

    point = shapely.affinity.translate(cube.unary_union.centroid, xoff=0, yoff=300)
    buffered = sg.to_gdf(point, crs=25833).buffer(100)

    assert (
        merged_rio.clip(buffered)
        .geom_almost_equals(merged_xarr.clip(buffered), decimal=3)
        .all()
    )
    assert tuple(merged_rio.total_bounds) == tuple(merged_xarr.total_bounds)

    cube_25832 = (
        sg.GeoDataCube.from_root(testdata, endswith=".tif", crs=25832, nodata=0)
        .explode()
        .query(query)
    )

    assert cube_25832.crs == 25832
    wpd_rio = cube_25832.load().to_crs(25833).to_gdf()

    assert tuple(merged_rio.total_bounds) == tuple(wpd_rio.total_bounds)
    equal_geoms = merged_rio.clip(buffered).geom_almost_equals(
        wpd_rio.clip(buffered), decimal=0
    )

    assert equal_geoms.all(), equal_geoms.value_counts()

    merged_xarr_32 = cube.load().to_crs(25832).merge().to_crs(25833).to_gdf()
    assert (
        merged_rio.clip(buffered)
        .geom_almost_equals(merged_xarr_32.clip(buffered), decimal=0)
        .all()
    )

    mergexarr32 = cube.to_crs(25832).merge().to_crs(25833).to_gdf()
    assert (
        merged_rio.clip(buffered)
        .geom_almost_equals(mergexarr32.clip(buffered), decimal=0)
        .all()
    )

    sg.explore(
        wpd_rio,
        merged_rio,
        # merged_xarr,
        mergexarr32,
        merged_xarr_32,
        "value",
    )
    assert cube.arrays.isna().all()


def test_merge():
    cube = sg.GeoDataCube.from_root(testdata, endswith=".tif").explode()
    print(cube.df)
    assert len(cube) == 20, len(cube)
    cube.df["area"] = cube.area

    all_merged = cube.copy().query("subfolder != 'sentinel2'").merge()
    assert len(all_merged) == 1, len(all_merged)

    only_multiband_merged = cube.copy().merge(by=["raster_id"])
    assert len(only_multiband_merged) == 19, len(only_multiband_merged)
    only_multiband_merged.df["area"] = only_multiband_merged.area

    cube2 = cube.copy().load().merge_by_bounds(by="res")
    cube2.df["area"] = cube2.area
    assert len(cube2) == 5, len(cube2)

    xarray_merge_by = cube.copy().load().merge(by="subfolder")
    assert len(xarray_merge_by) == 3, len(xarray_merge_by)
    assert list(sorted(xarray_merge_by.res)) == [(10, 10), (10, 10), (30, 30)], list(
        xarray_merge_by.res.values
    )
    rasterio_merge_by = cube.copy().load().merge(by="subfolder")
    assert len(rasterio_merge_by) == 3, len(rasterio_merge_by)
    assert list(sorted(rasterio_merge_by.res)) == [(10, 10), (10, 10), (30, 30)], list(
        rasterio_merge_by.res.values
    )
    x_mean = int(np.mean([r.array.mean() for r in xarray_merge_by]))
    r_mean = int(np.mean([r.array.mean() for r in rasterio_merge_by]))
    assert x_mean == r_mean, (x_mean, r_mean)


def test_dissolve():
    cube = sg.GeoDataCube.from_root(testdata, endswith=".tif")
    cube = cube.merge_by_bounds()
    list(cube.shape) == [(1, 201, 201), (2, 201, 201)]
    print(cube)
    cube = cube.dissolve_bands("mean")
    list(cube.shape) == [(201, 201), (201, 201)]
    print(cube)


def test_intersection():
    cube = sg.GeoDataCube.from_root(testdata, endswith=".tif").query(
        "subfolder != 'sentinel2'"
    )
    cube._crs = 25833
    grid = sg.make_grid(cube.unary_union, gridsize=1000, crs=cube.crs)
    grid["idx"] = range(len(grid))

    print(len(grid))
    print(len(cube))

    intersected = cube.intersection(grid)
    print(len(intersected))
    print(intersected.df["idx"])
    print(intersected)

    intersected_pooled = cube.pool(3).intersection(grid).execute()
    print(len(intersected))
    print(intersected.df["idx"])
    print(intersected)

    assert intersected_pooled.equals(intersected)


def test_explode():
    cube = sg.GeoDataCube.from_root(testdata, endswith=".tif")
    print(cube.shape)
    cube = cube.explode()
    print(cube.shape)


def test_merge_from_array():
    should_give = np.array(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17]],
    )

    arr1 = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    arr2 = np.array([[9, 10, 11], [12, 13, 14], [15, 16, 17]])

    arr_3d = np.array(
        [
            [
                [10, 10, 10],
                [10, 10, 10],
                [10, 10, 10],
                [10, 10, 10],
                [10, 10, 10],
                [10, 10, 10],
            ]
        ]
    )

    assert arr1.shape == (3, 3)
    assert arr2.shape == (3, 3)
    assert arr_3d.shape == (1, 6, 3), arr_3d.shape

    r2 = sg.Raster.from_array(arr1, bounds=(0, 3, 3, 6), crs=25833)
    r1 = sg.Raster.from_array(arr2, bounds=(0, 0, 3, 3), crs=25833)
    r3 = sg.Raster.from_array(arr_3d, bounds=(0, 0, 3, 6), crs=25833)

    def should_be_same():
        cube = sg.GeoDataCube([r1])
        merged = cube.merge()
        assert cube.total_bounds == merged.total_bounds
        assert cube.res.equals(merged.res)
        assert cube.area.sum() == merged.area.sum()
        assert cube.equals(merged), (cube.arrays.iloc[0], merged.arrays.iloc[0])

        cube = sg.GeoDataCube([r1, r2])
        assert cube.area.sum() == 18, cube.area.sum()

        merged = cube.merge()
        assert np.array_equal(merged.arrays.iloc[0], should_give)
        assert len(merged) == 1
        assert merged.total_bounds == cube.total_bounds, (
            merged.total_bounds,
            cube.total_bounds,
        )
        assert merged.area.sum() == 18, merged.area.sum()

    should_be_same()

    def zeros_should_be_10(should_give):
        should_give = np.where(should_give == 0, 10, should_give)
        should_give = np.array([should_give])
        assert should_give.shape == (1, 6, 3)

        cube = sg.GeoDataCube([r1, r2, r3])
        assert cube.area.sum() == 18 * 2, cube.area.sum()

        merged = cube.merge()
        print(merged.arrays.iloc[0])
        assert np.array_equal(merged.arrays.iloc[0], should_give), merged.arrays.iloc[0]
        sg.explore(merged.to_gdf(), "value")
        assert len(merged) == 1
        assert merged.area.sum() == 18, merged.area.sum()

    zeros_should_be_10(should_give)

    def len_should_be_2_and_3():
        cube = sg.GeoDataCube([r1, r2, r3])
        cube.df["numcol"] = [0, 1, 2]
        cube.df["txtcol"] = [*"aab"]

        merged = cube.merge(by="txtcol")
        assert len(merged) == 2
        assert merged.area.sum() == 18 * 2, merged.area.sum()

        merged = cube.merge(by="numcol")
        assert len(merged) == 3
        assert merged.area.sum() == 18 * 2, merged.area.sum()

    def all_should_be_10():
        merged = sg.GeoDataCube([r3, r2, r1]).merge()
        print(arr_3d.shape)
        print(merged.arrays.iloc[0].shape)
        assert np.array_equal(merged.arrays.iloc[0], arr_3d), merged.arrays.iloc[0]

    all_should_be_10()


def test_from_gdf():
    cube = sg.GeoDataCube.from_root(testdata, endswith=".tif")
    gdf = cube[0].load().to_gdf("val")
    print(gdf)
    cube = sg.GeoDataCube.from_gdf(
        gdf, tilesize=100, processes=1, columns=["val"], res=10
    )
    print(cube.df)

    cube = sg.GeoDataCube.from_gdf(
        gdf, tilesize=100, processes=4, columns=["val"], res=10
    )
    print(cube.df)


def test_sample():
    cube = sg.GeoDataCube.from_root(testdata, endswith=".tif")
    sample = cube.sample(buffer=100)
    assert len(sample) == 1

    samples = cube.sample(10)
    assert len(samples) == 10, len(samples)

    for cube in samples:
        assert isinstance(cube, sg.GeoDataCube)
        for array in cube.arrays:
            assert isinstance(array, np.ndarray)


def test_zonal():
    arr_1d = np.arange(0, 8)
    arr_2d = arr_1d + np.arange(8).reshape(-1, 1)
    print(arr_2d)
    assert arr_2d.shape == (8, 8)

    r = sg.Raster.from_array(arr_2d, crs=25833, bounds=(0, 0, 9, 9), nodata=0)
    r2 = sg.Raster.from_array(arr_2d, crs=25833, bounds=(0, 9, 9, 19), nodata=0)
    cube = sg.GeoDataCube([r, r2])

    grid = sg.make_grid_from_bbox(*cube.total_bounds, gridsize=4, crs=25833)
    assert len(grid) == 24

    touch_or_overlap = grid.overlaps(cube.unary_union) | grid.within(cube.unary_union)
    assert sum(touch_or_overlap) == 15

    zonal = cube.zonal(grid, aggfunc="sum")
    assert len(zonal) == 15, len(zonal)

    print(zonal)
    assert list(sorted(zonal["sum"])) == [
        1,
        3,
        6,
        6,
        9,
        16,
        16,
        24,
        24,
        30,
        30,
        48,
        48,
        48,
        48,
    ], list(sorted(zonal["sum"]))


def test_pool():
    cube = sg.GeoDataCube.from_root(testdata, endswith=".tif", dtype=np.float32).query(
        "subfolder == 'sentinel2'"
    )
    cube._crs = 25833

    center = cube.unary_union.centroid.buffer(200)

    results = (cube.copy().clip(center) * 2).array_map(x2).array_map(
        np.float32
    ).explode() // 2

    results_pooled = (
        (cube.pool(4).clip(center) * 2).array_map(x2).array_map(np.float32).explode()
        // 2
    ).execute()

    assert results.equals(results_pooled)

    grid = sg.make_grid(center, gridsize=100, crs=cube.crs)
    zonal = cube.copy().zonal(grid, aggfunc=["sum", np.mean], by_date=False)
    print(zonal)
    # assert int(zonal["sum"].max()) == 10, int(zonal["sum"].max())
    # assert int(zonal["sum"].mean()) == 10, int(zonal["sum"].mean())

    zonal_pooled = (
        cube.pool(4).zonal(grid, aggfunc=["sum", np.mean], by_date=False).execute()
    )
    assert zonal.equals(zonal_pooled)


def write_sentinel():
    src_path_sentinel = r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\SENTINEL2X_20230415-230437-251_L3A_T32VLL_C_V1-3"

    cube = sg.GeoDataCube.from_root(
        src_path_sentinel, endswith=".tif", raster_type=sg.Sentinel2
    )

    mask = sg.to_gdf(cube.unary_union.centroid.buffer(1000))

    def _save_raster(file, src_path_sentinel):
        if ".tif" not in file:
            return
        path = Path(src_path_sentinel) / file
        r = sg.Raster.from_path(path).clip(mask)
        out_path = str(Path(path_sentinel) / Path(file).stem) + "_clipped.tif"
        print(out_path)
        r.write(out_path)

    for file in os.listdir(src_path_sentinel):
        _save_raster(file, src_path_sentinel)

    for file in os.listdir(src_path_sentinel + "/MASKS"):
        _save_raster(file, src_path_sentinel + "/MASKS")


if __name__ == "__main__":
    import cProfile

    # write_sentinel()

    def test_cube():
        test_to_gdf()
        test_query()
        test_intersection()
        test_explode()
        test_merge_from_array()
        test_elevation()
        test_zonal()
        test_sentinel()
        test_pool()
        test_xdataset()

        test_shape()
        test_dissolve()
        test_copy()
        test_from_gdf()
        not_test_df()
        test_from_root()
        test_merge()
        test_sample()
        test_to_crs()
        # test_retile()

    test_cube()
    # cProfile.run("test_cube()", sort="cumtime")

# %%