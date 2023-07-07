# %%
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


def x2(x):
    return x * 2


def test_rasterseries(use_multiprocessing=False):
    cubesti_sentinel = r"C:\Users\ort\OneDrive - Statistisk sentralbyrå\data\SENTINEL2X_20230415-230437-251_L3A_T32VLL_C_V1-3"

    s = sg.GeoDataCube.from_root(cubesti_sentinel, contains=".tif", dapla=False)
    print(s.bounds)
    print(s.total_bounds)
    print(s.band_indexes)
    print(s.res)
    print(s.tiles)
    ss

    r = sg.Raster.from_path(path_singleband)
    r2 = sg.ElevationRaster.from_path(path_two_bands)

    mask_utm33 = sg.to_gdf(box(*r.bounds), crs=r.crs).centroid.buffer(250)

    s = sg.GeoDataCube.from_root(
        Path(path_singleband).parent, contains=".tif", dapla=False
    )
    print(s)

    s = sg.GeoDataCube([r, r2, r, r, r, r, r], use_multiprocessing=use_multiprocessing)

    s = s.clip(mask=mask_utm33, crop=True)
    print(s)
    print(s.indexes)
    print(s.indexes)
    print(s.shape)
    print(s.res)
    print(s.crs)
    sss

    print(s.mean())
    print(s.max())

    s = s.map(x2)
    s = s.map(abs)
    print(s.mean())
    print(s.max())

    s = s * 2
    print(s.mean())
    print(s.max())

    assert isinstance(s, sg.GeoDataCube), type(s)
    assert isinstance(s.df, pd.DataFrame), type(s.df)
    assert isinstance(s.df["raster"], pd.Series), type(s)
    assert isinstance(s.df.loc[0, "raster"], sg.Raster), type(s)
    assert isinstance(s.df.loc[1, "raster"], sg.ElevationRaster)


if __name__ == "__main__":
    time = perf_counter()
    test_rasterseries(use_multiprocessing=True)
    print("use_multiprocessing=True", perf_counter() - time)
    test_rasterseries(use_multiprocessing=False)
    print("use_multiprocessing=False", perf_counter() - time)
