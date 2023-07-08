# %%
import multiprocessing
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from IPython.display import display
from shapely import box


src = str(Path(__file__).parent.parent) + "/src"
testdata = str(Path(__file__).parent.parent) + "/tests/testdata"

sys.path.insert(0, src)

import sgis as sg


path_singleband = testdata + "/dtm_10.tif"
path_two_bands = testdata + "/dtm_10_two_bands.tif"
sg.Raster.dapla = False

"""
støvsuge
rydde

maten
briller

"""


def x2(x):
    return x * 2


def test_cube(use_multiprocessing=False, processes=None):
    cubesti_sentinel = r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\SENTINEL2X_20230415-230437-251_L3A_T32VLL_C_V1-3"

    cube = sg.GeoDataCube.from_root(cubesti_sentinel, contains=".tif", dapla=False)

    mask = cube.unary_union.centroid.buffer(100)
    cube = (cube.clip(mask) * 2).explode()
    print(cube)
    return

    print(cube.bounds)
    print(cube.total_bounds)
    print(cube.band_indexes)
    print(cube.res)
    print(cube.tiles)

    r = sg.Raster.from_path(path_singleband)
    r2 = sg.ElevationRaster.from_path(path_two_bands)

    cube = sg.GeoDataCube.from_root(
        Path(path_singleband).parent, contains=".tif", dapla=False
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
    time = perf_counter()

    test_cube(use_multiprocessing=False)
    print("use_multiprocessing=False", perf_counter() - time)

    test_cube(use_multiprocessing=True, processes=multiprocessing.cpu_count() - 2)
    print("use_multiprocessing=True", perf_counter() - time)
