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
sg.Raster.dapla = False
sg.GeoDataCube.dapla = False


def x2(x):
    return x * 2


def not_test_df():
    df_path = testdata + "/cube_df.parquet"
    try:
        os.remove(df_path)
    except FileNotFoundError:
        pass
    try:
        cube = sg.GeoDataCube.from_root(testdata, endswith=".tif").explode()
        df = cube._prepare_df_for_parquet()
        df.to_parquet(df_path)
        display(df)
        cube_from_df = sg.GeoDataCube.from_df(df_path).explode()
        assert hasattr(cube_from_df, "_from_df")
        assert cube_from_df.boxes.intersects(cube.unary_union).all()

        cube_from_df = sg.GeoDataCube.from_root(testdata, endswith=".tif").explode()
        assert hasattr(cube_from_df, "_from_df")
        assert cube_from_df.boxes.intersects(cube.unary_union).all()
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


def test_merge():
    cube = sg.GeoDataCube.from_root(testdata, endswith=".tif").explode()
    assert len(cube) == 4, len(cube)
    cube.df["area"] = cube.area
    display(cube.df)
    print(cube.res)

    cube2 = cube.copy().merge()
    assert len(cube2) == 1, len(cube2)
    assert list(cube2.res.values) == [(10, 10)], list(cube2.res.values)
    cube2.df["area"] = cube2.area
    display(cube2.df)
    print(cube.res)

    cube2 = cube.copy().merge(by=["band_indexes", "path"])
    assert list(cube2.res.values) == [(10, 10), (10, 10), (10, 10), (30, 30)], list(
        cube2.res.values
    )
    assert len(cube2) == 4, len(cube2)
    cube2.df["area"] = cube2.area
    display(cube2.df)
    print(cube.res)

    cube2 = cube.copy().merge_by_band()
    assert len(cube2) == 2, len(cube2)
    assert (cube2.shape.str.len() == 2).all(), "should return 2d array"
    assert list(cube2.res.values) == [(10, 10), (10, 10)], list(cube2.res.values)
    cube2.df["area"] = cube2.area
    display(cube2.df)
    print(cube.res)

    cube2 = cube.copy().merge_by_bounds()
    display(cube2.df)
    display(cube2.df["band_indexes"])
    assert len(cube2) == 2, len(cube2)
    assert (cube2.shape.str.len() == 3).all(), "should return 3d array"
    assert list(cube2.res.values) == [(10, 10), (10, 10)], list(cube2.res.values)
    cube2.df["area"] = cube2.area
    print(cube.res)


def test_dissolve():
    cube = sg.GeoDataCube.from_root(testdata, endswith=".tif")
    cube.merge_by_bounds()
    cube.shape == pd.Series([(1, 201, 201), (2, 201, 201)])
    print(cube)
    cube.dissolve_bands("mean")
    cube.shape == pd.Series([(201, 201), (201, 201)])
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


def test_chaining():
    cube = sg.GeoDataCube.from_root(testdata, endswith=".tif")
    cube = ((cube.chain(processes=6).load().map(x2) * 2).explode() * 2).execute()


def _test_add_meta():
    cubesti_sentinel = r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\SENTINEL2X_20230415-230437-251_L3A_T32VLL_C_V1-3"

    cube = sg.GeoDataCube.from_root(cubesti_sentinel, endswith=".tif", dapla=False)

    assert all(hasattr(r, "_meta_added") for r in cube), [
        hasattr(r, "_meta_added") for r in cube
    ]
    display(cube.df)
    cube.update_df()
    display(cube.df)

    cube2 = sg.GeoDataCube.from_df(cube.df.drop(["raster"], axis=1))
    assert not any(hasattr(r, "_meta_added") for r in cube2), [
        hasattr(r, "_meta_added") for r in cube2
    ]


def test_cube(processes=None):
    cubesti_sentinel = r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\SENTINEL2X_20230415-230437-251_L3A_T32VLL_C_V1-3"

    cube = sg.GeoDataCube.from_root(
        cubesti_sentinel,
        endswith=".tif",
        processes=processes,
    )
    cube.bounds
    cube.height
    cube.width
    cube2 = sg.GeoDataCube.from_root(
        cubesti_sentinel, endswith=".tif", dapla=False
    ).add_meta()

    mask = cube2.unary_union.centroid.buffer(10000)
    # mask = sg.to_gdf([0, 0], crs=cube2.crs)

    cube2 = cube2.clip(mask, crop=True)

    cube = (cube.clip(mask, crop=True) * 2).explode()
    return

    print(cube.bounds)
    print(cube.total_bounds)
    print(cube.band_indexes)
    print(cube.res)
    print(cube.tiles)

    r = sg.Raster.from_path(path_singleband)
    r2 = sg.ElevationRaster.from_path(path_two_bands)

    cube = sg.GeoDataCube(
        [r2],
        use_multiprocessing=use_multiprocessing,
        processes=processes,
    )

    print(cube.gradient())
    print(cube.degrees())

    cube = sg.GeoDataCube.from_root(
        Path(path_singleband).parent, endswith=".tif", dapla=False
    )

    concatted = sg.concat_cubes([cube, cube])
    assert len(concatted) == len(cube) * 2, concatted

    r.load()
    r2.load()
    cube = sg.GeoDataCube(
        [r, r2, r, r, r, r, r],
        use_multiprocessing=use_multiprocessing,
        processes=processes,
    )

    assert len(cube) == 7, cube
    assert all(len(r.shape) == 3 for r in cube.df["raster"]), cube

    exploded = cube.copy().explode()
    assert all(len(r.shape) == 3 for r in cube.df["raster"]), cube
    assert exploded._rasters_have_changed is False

    assert len(exploded) == 8, exploded
    assert all(len(r.shape) == 2 for r in exploded.df["raster"]), exploded

    # check that it's copying
    assert len(exploded) == len({id(x) for x in exploded.df["raster"]})

    cube = cube.load().explode()
    assert len(cube) == 8, cube
    assert all(len(r.shape) == 2 for r in cube.df["raster"]), cube
    assert len(cube) == len({id(x) for x in cube.df["raster"]})

    print("intersects")
    print(cube.intersects(r.unary_union))

    assert exploded._rasters_have_changed is False
    cube = cube.clip(mask=cube.unary_union)
    assert len(cube) == 8, cube
    assert exploded._rasters_have_changed is True

    print(cube)
    print(cube.band_indexes)
    print(cube.band_indexes)
    print(cube.shape)
    print(cube.res)
    print(cube.crs)

    # should not be possible with multiple clips without saving to files
    i = 0
    try:
        cube = cube.clip(mask=cube.unary_union)
    except ValueError:
        i += 1
    if i == 0:
        raise ValueError("Did not raise...")

    print(cube.mean())
    print(cube.max())

    cube = cube.map(x2)
    cube = cube.map(abs)
    print(cube.mean())
    print(cube.max())

    cube = cube * 2
    print(cube.mean())
    print(cube.max())

    cube.df = cube.df.reset_index(drop=True)
    print(cube)
    assert isinstance(cube, sg.GeoDataCube), type(cube)
    assert isinstance(cube.df, pd.DataFrame), type(cube.df)
    assert isinstance(cube.df["raster"], pd.Series), type(cube.df["raster"])
    assert isinstance(cube.df.loc[0, "raster"], sg.Raster), type(
        cube.df.loc[0, "raster"]
    )
    assert isinstance(cube.df.loc[1, "raster"], sg.ElevationRaster), type(
        cube.df.loc[1, "raster"]
    )
    assert isinstance(cube.df.loc[2, "raster"], sg.ElevationRaster), type(
        cube.df.loc[2, "raster"]
    )


if __name__ == "__main__":
    test_from_root()
    test_dissolve()
    not_test_df()
    test_merge()
    test_chaining()

    for processes in [1, 6]:
        time = perf_counter()
        test_cube(processes=processes)

        if processes == 6:
            print(f"with multiprocessing:", perf_counter() - time)
        else:
            print(f"without multiprocessing:", perf_counter() - time)
