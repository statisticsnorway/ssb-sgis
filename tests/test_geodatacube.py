# %%
import multiprocessing
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from IPython.display import display
from shapely import box


src = str(Path(__file__).parent.parent) + "/src"
testdata = str(Path(__file__).parent.parent) + "/tests/testdata/raster"

sys.path.insert(0, src)

import sgis as sg


path_singleband = testdata + "/dtm_10.tif"
path_two_bands = testdata + "/dtm_10_two_bands.tif"


def x2(x):
    return x * 2


def test_to_crs():
    cube = sg.GeoDataCube.from_root(
        testdata, endswith=".tif", raster_type=sg.ElevationRaster
    ).load()
    print(cube.bounds)
    cube = cube.to_crs(25832)
    print(cube.bounds)
    cube = cube.load()
    print(cube.bounds)


def test_shape():
    cube = sg.GeoDataCube.from_root(
        testdata, endswith=".tif", raster_type=sg.ElevationRaster
    )
    cube = cube.load(res=10)
    assert (cube.res == (10, 10)).all(), cube.res
    cube = cube.load(res=30)

    c = cube.unary_union.centroid.buffer(100)
    cube = cube.clip(c, res=10)
    assert (cube.res == (10, 10)).all(), cube.res
    assert (cube.shape.str[1:] == (20, 20)).all(), cube.shape.str[1:]

    cube = cube.clip(c, res=20)
    assert (cube.res == (20, 20)).all(), cube.res
    assert (cube.shape.str[1:] == (10, 10)).all(), cube.shape.str[1:]


def test_copy():
    cube = sg.GeoDataCube.from_root(
        testdata, endswith=".tif", raster_type=sg.ElevationRaster
    )

    assert cube.arrays.isna().all()
    cube2 = cube.load().load().load()
    assert cube.arrays.isna().all()
    assert cube2.arrays.notna().all()

    cube3 = (cube2.chain(processes=2) * 2).execute()
    assert int(cube3.max()) == int(cube2.max()) * 2


def test_elevation():
    cube = sg.GeoDataCube.from_root(
        testdata, endswith=".tif", raster_type=sg.ElevationRaster
    ).load()
    print(cube.raster_attribute("degrees"))

    print(cube.max())

    print(cube.copy().run_raster_method("gradient").max())
    print(cube.copy().run_raster_method("degrees").max())

    print(cube.copy().chain(processes=3).run_raster_method("degrees").execute().max())


def not_test_indices():
    path_sentinel = r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\SENTINEL2X_20230415-230437-251_L3A_T32VLL_C_V1-3"

    cube = sg.GeoDataCube.from_root(path_sentinel)
    print(cube)


def not_test_sentinel():
    path_sentinel = r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\SENTINEL2X_20230415-230437-251_L3A_T32VLL_C_V1-3"

    cube = sg.GeoDataCube.from_root(
        path_sentinel, endswith=".tif", raster_type=sg.Sentinel2
    ).query("band_name.notna()")
    display(cube.df)
    display(cube.meta)

    cube = sg.ndvi_index(cube, band_name_red="B4", band_name_nir="B8")
    display(cube.df)

    ss

    cube = sg.SentinelCube.from_root(path_sentinel, with_masks=False)
    print(cube)

    ndvi = cube.ndvi()
    print(ndvi)
    ss

    ndvi = cube.chain(processes=3).load().ndvi().execute()
    print(ndvi)

    ss
    cube = sg.GeoDataCube.from_root(
        path_sentinel, endswith=".tif", raster_type=sg.Sentinel2
    )
    cube._df = cube.df[cube.raster_attribute("is_mask")]
    cube = cube.load()

    for r in cube:
        print(r.path)
        r.plot()
    ss

    for col in cube.df:
        print(cube.df[col])
    display(cube.df)
    print(cube.raster_attribute("band_name"))
    print(cube.raster_attribute("date"))
    print(cube.copy().merge(by="band_name"))

    rbg = cube.copy().df[cube.raster_attribute("is_rbg")]
    assert len(rbg) == 3
    display(rbg)

    rbg = cube.df[
        lambda x: x.band_name.isin(cube.raster_attribute("rgb_bands").iloc[0])
    ]
    assert len(rbg) == 3
    display(rbg)


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
    assert len(cube) == 4, cube
    display(cube)

    cube = sg.GeoDataCube.from_root(testdata, regex=r"\.tif$").explode()
    assert len(cube) == 4, cube
    display(cube)


def test_retile():
    cube = sg.GeoDataCube.from_root(testdata, endswith=".tif")
    assert cube.area.max() == 4040100
    retiled = cube.retile(tilesize=100, res=10, band_id="name")
    retiled.df["area"] = retiled.area
    assert retiled.area.max() == 10000
    print(retiled.df)

    retiled = (
        cube.chain(processes=4).retile(tilesize=100, res=10, band_id="name").execute()
    )
    retiled.df["area"] = retiled.area
    assert retiled.area.max() == 10000
    print(retiled.df)


def test_merge():
    cube = sg.GeoDataCube.from_root(testdata, endswith=".tif").explode()
    assert len(cube) == 4, len(cube)
    cube.df["area"] = cube.area

    cube2 = cube.copy().load().merge_by_bounds(by="res")
    cube2.df["area"] = cube2.area
    display(cube2.df)
    print(cube2.res)
    assert len(cube2) == 3, len(cube2)

    cube2 = cube.copy().merge()
    assert len(cube2) == 1, len(cube2)
    assert list(cube2.res.values) == [(10, 10)], list(cube2.res.values)
    cube2.df["area"] = cube2.area
    display(cube2.df)
    print(cube.res)

    cube2 = cube.copy().merge(by=["band_index", "path"])
    assert list(cube2.res.values) == [(10, 10), (10, 10), (10, 10), (30, 30)], list(
        cube2.res.values
    )
    assert len(cube2) == 4, len(cube2)
    cube2.df["area"] = cube2.area
    display(cube2.df)
    print(cube.res)

    cube2 = cube.copy().merge(by="band_index")
    assert len(cube2) == 2, len(cube2)
    assert (cube2.shape.str.len() == 2).all(), "should return 2d array"
    assert list(cube2.res.values) == [(10, 10), (10, 10)], list(cube2.res.values)
    cube2.df["area"] = cube2.area
    display(cube2.df)
    print(cube.res)

    cube2 = cube.copy().merge_by_bounds()
    display(cube2.df)
    display(cube2.df["band_index"])
    assert len(cube2) == 2, len(cube2)
    assert (cube2.shape.str.len() == 3).all(), "should return 3d array"
    assert list(cube2.res.values) == [(10, 10), (10, 10)], list(cube2.res.values)
    cube2.df["area"] = cube2.area
    print(cube.res)


def test_dissolve():
    cube = sg.GeoDataCube.from_root(testdata, endswith=".tif")
    cube = cube.merge_by_bounds()
    list(cube.shape) == [(1, 201, 201), (2, 201, 201)]
    print(cube)
    cube = cube.dissolve_bands("mean")
    list(cube.shape) == [(201, 201), (201, 201)]
    print(cube)


def not_test_merge():
    arr012 = np.array([[0, 1, 2], [0, 1, 2]])
    arr345 = np.array([[3, 4, 5], [3, 4, 5]])
    arr_3d = np.array([arr012, arr345])
    assert arr012.shape == (2, 3)
    assert arr345.shape == (2, 3)
    assert arr_3d.shape == (2, 2, 3)

    rasters = [
        sg.Raster.from_array(arr, bounds=(0, 0, 2, 2), crs=None)
        for arr in (arr012, arr345, arr_3d)
    ]

    cube = sg.GeoDataCube(rasters)
    cube.df["numcol"] = [0, 1, 2]
    cube.df["txtcol"] = [*"aab"]
    print(cube)
    assert cube.area.sum() == 12, cube.area.sum()


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
    samples = cube.sample(10, buffer=100)
    sg.explore(sample.to_gdf(), samples.to_gdf())
    ss

    for cube in samples:
        for array in cube.arrays:
            pass


def test_chaining():
    cube = sg.GeoDataCube.from_root(testdata, endswith=".tif")

    cube = (
        (cube.chain(processes=6).load().map(x2).map(np.float16) * 2).explode() / 2
    ).execute()


def _test_add_meta():
    cubesti_sentinel = r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\SENTINEL2X_20230415-230437-251_L3A_T32VLL_C_V1-3"

    cube = sg.GeoDataCube.from_root(cubesti_sentinel, endswith=".tif")

    assert all(hasattr(r, "_meta_added") for r in cube), [
        hasattr(r, "_meta_added") for r in cube
    ]
    display(cube.df)
    cube.update_df()
    display(cube.df)

    cube2 = sg.GeoDataCube.from_cube_df(cube.df.drop(["raster"], axis=1))
    assert not any(hasattr(r, "_meta_added") for r in cube2), [
        hasattr(r, "_meta_added") for r in cube2
    ]


if __name__ == "__main__":
    import cProfile

    # TODO: band_name til band

    def test_cube():
        # not_test_sentinel()
        test_merge()
        test_shape()
        test_copy()
        test_chaining()
        test_elevation()
        test_from_gdf()
        not_test_df()
        test_from_root()
        test_dissolve()
        test_merge()
        test_retile()
        # test_sample()

    test_cube()
    # cProfile.run("test_cube()", sort="cumtime")
