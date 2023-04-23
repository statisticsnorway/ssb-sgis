# %%
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point


src = str(Path(__file__).parent.parent) + "/src"

sys.path.insert(0, src)

import sgis as sg


def test_to_gdf():
    _dflike_geom_col()

    _single_geom_in_dict()

    _preserves_index()

    _df_mixed_types()

    _incorrect_geom_col()

    _dflike_single_col()

    _recursive()

    _xyz()

    _iterators()

    _wkt_wkb()

    _coords()

    _linestring()

    _df()

    _geoseries()


def _dflike_single_col():
    single_key = {"geom_col_name": [(10, 60), (11, 59)]}
    print(single_key)
    gdf = sg.to_gdf(single_key)
    print(gdf)
    print("")
    assert gdf.shape == (2, 1), gdf.shape
    assert gdf.index.to_list() == [0, 1]
    assert not gdf.geometry.isna().any(), gdf
    assert gdf.columns.to_list() == ["geometry"]

    df = pd.DataFrame(single_key)
    print(df)
    gdf2 = sg.to_gdf(df)
    assert gdf2.equals(gdf)

    index = [1, 3]
    gdf = sg.to_gdf(single_key, index=index)
    assert gdf.index.to_list() == index
    # should give a None
    assert gdf.geometry.isna().any(), gdf

    gdf2 = sg.to_gdf(df, index=index)
    assert gdf2.equals(gdf)


def _dflike_geom_col():
    dict_ = {"col": [1, 2], "geometry": [(10, 60), (11, 59)]}
    gdf = sg.to_gdf(dict_, crs=4326)

    assert gdf.equals(
        gpd.GeoDataFrame(
            {"col": [1, 2], "geometry": [Point(10, 60), Point(11, 59)]},
            geometry="geometry",
            crs=4326,
        )
    ), gdf

    df = pd.DataFrame(dict_)
    gdf2 = sg.to_gdf(df, geometry="geometry", crs=4326)
    assert gdf2.equals(gdf), gdf2

    dict_ = {"col": [1, 2, 3], "geometry": [(10, 60), (11, 59)]}

    with pytest.raises(ValueError):
        sg.to_gdf(dict_)


def _preserves_index():
    dict_ = {"col": [1, 2], "geometry": [(10, 60), (11, 59)]}
    index = [1, 3]
    df = pd.DataFrame(dict_, index=index)

    gdf = sg.to_gdf(dict_, index=index)
    assert gdf.index.to_list() == index

    gdf = sg.to_gdf(df, index=index)
    assert gdf.index.to_list() == index


def _single_geom_in_dict():
    should_equal = gpd.GeoDataFrame(
        {"geometry": [Point(263206.1, 6651199.5)]}, geometry="geometry"
    )

    geom = {0: [263206.1, 6651199.5]}
    gdf = sg.to_gdf(geom)
    assert gdf.equals(should_equal), gdf

    print("\n\n")
    geom = {"geometry": [263206.1, 6651199.5]}
    gdf = sg.to_gdf(geom)
    assert gdf.equals(should_equal), gdf


def _incorrect_geom_col():
    dict_ = {"col": [1, 2], "geom": [(10, 60), (11, 59)]}
    gdf = sg.to_gdf(dict_, geometry=["geom"])

    df = pd.DataFrame(dict_)
    gdf = sg.to_gdf(df, geometry="geom")

    with pytest.raises(ValueError):
        sg.to_gdf(dict_, geometry="geometry")
    with pytest.raises(ValueError):
        sg.to_gdf(df, geometry="geometry")


def _df_mixed_types():
    df = pd.DataFrame(
        {
            "what": ["coords", "coordslist", "array", "listcomp", "wkt", "wkb"],
            "geometry": [
                (60, 10),
                [(60, 10), (59, 11)],
                np.random.randint(10, size=(5, 3)),
                [(x, y, z) for x, y, z in np.random.randint(10, size=(5, 3))],
                "point (60 10)",
                b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00N@\x00\x00\x00\x00\x00\x00$@",
            ],
        }
    )
    gdf = sg.to_gdf(df, crs=4326)
    print(df)
    print(gdf)
    print("")
    assert not gdf.geometry.isna().sum()


def _recursive():
    coords = [[[([([[([([10, 60])])]])], [(59, 10)])]], (5, 10)]
    gdf = sg.to_gdf(coords, crs=4326)
    assert not gdf.geometry.isna().sum()
    print(coords)
    print(gdf)
    print("")


def _xyz():
    dict_ = {"col": [1, 2], "x": [10, 11], "y": [60, 59], "z": [10, 20]}
    gdf = sg.to_gdf(dict_, geometry=["x", "y", "z"])
    print(dict_)
    print(gdf)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.shape == (2, 5)
    assert not gdf.geometry.isna().sum()
    assert gdf.geometry.has_z.all()

    gdf2 = sg.to_gdf(dict_, geometry="xyz")
    print(dict_)
    print(gdf2)
    assert gdf.equals(gdf2)

    df = pd.DataFrame(dict_)
    gdf = sg.to_gdf(df, geometry=["x", "y", "z"])
    print(dict_)
    print(gdf)
    print("")
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.shape == (2, 5)
    assert not gdf.geometry.isna().sum()
    assert gdf.geometry.has_z.all()

    zipped = zip(df.x, df.y, df.z)
    gdf = sg.to_gdf(zipped)
    assert not gdf.geometry.isna().sum()
    zipped = zip(dict_["x"], dict_["y"], dict_["z"])
    gdf = sg.to_gdf(zipped)
    assert not gdf.geometry.isna().sum()
    print(dict_)
    print(gdf)
    print("")


def _iterators():
    set_ = {(10, 60), (59, 10), (5, 10)}
    gdf = sg.to_gdf(set_, crs=4326)
    assert not gdf.geometry.isna().sum()

    dict_ = {1: (10, 60), 2: (11, 59)}
    gdf = sg.to_gdf(dict_, crs=4326)
    assert list(gdf.index) == [1, 2]
    assert not gdf.geometry.isna().sum()
    print(dict_)
    print(gdf)
    print("")

    geom_array_ = gdf.geometry.values
    gdf = sg.to_gdf(geom_array_)
    assert len(gdf) == 2
    assert not gdf.geometry.isna().sum()
    assert gdf.index.to_list() == [0, 1]

    np_array = np.array(gdf.geometry.values)
    gdf = sg.to_gdf(np_array)
    assert not gdf.geometry.isna().sum()
    assert len(gdf) == 2

    list_ = [x for x in gdf.geometry.values]
    gdf = sg.to_gdf(list_)
    assert not gdf.geometry.isna().sum()
    assert len(gdf) == 2

    generator_ = (x for x in gdf.geometry.values)
    gdf = sg.to_gdf(generator_)
    assert len(gdf) == 2, gdf
    assert not gdf.geometry.isna().sum()

    n = 10
    generator2_ = ((x, y) for x, y in np.random.randint(10, size=[n, 2]))
    gdf = sg.to_gdf(generator2_)
    assert not gdf.geometry.isna().sum()
    assert len(gdf) == n


def _wkt_wkb():
    wkt = "POINT (60 10)"
    gdf = sg.to_gdf(wkt, crs=4326)
    assert len(gdf) == 1
    assert not gdf.geometry.isna().sum()

    wkb = b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00N@\x00\x00\x00\x00\x00\x00$@"
    gdf = sg.to_gdf(wkb, crs=4326)
    assert len(gdf) == 1
    assert not gdf.geometry.isna().sum()


def _coords():
    coords = 10, 60
    gdf = sg.to_gdf(coords, crs=4326)
    assert len(gdf) == 1
    assert not gdf.geometry.isna().sum()
    coords = (10, 60, 1)
    gdf = sg.to_gdf(coords, crs=4326)
    assert len(gdf) == 1
    assert not gdf.geometry.isna().sum()

    coords = ((10, 60), (1, 1))
    gdf = sg.to_gdf(coords, crs=4326)
    assert len(gdf) == 2
    assert not gdf.geometry.isna().sum()

    coords = [[10, 60], [1, 1]]
    gdf = sg.to_gdf(coords, crs=4326)
    assert len(gdf) == 2
    assert not gdf.geometry.isna().sum()

    coords = [(10, 60), (1, 1)]
    gdf = sg.to_gdf(coords, crs=4326)
    assert len(gdf) == 2
    assert not gdf.geometry.isna().sum()

    coordsarray = sg.coordinate_array(gdf)
    gdf = sg.to_gdf(coordsarray, crs=4326)
    assert len(gdf) == 2
    assert not gdf.geometry.isna().sum()


def _geoseries():
    gdf = sg.to_gdf([(10, 60), (1, 1)])
    geoseries = gdf.geometry
    geoseries_with_crs = geoseries.set_crs(4326)
    assert geoseries.crs is None
    assert geoseries_with_crs.crs == 4326

    gdf = pd.concat(
        sg.to_gdf(geom) for geom in [geoseries_with_crs, geoseries, geoseries]
    )
    assert gdf.crs == 4326
    assert len(gdf) == 6
    assert not gdf.geometry.isna().sum()

    failed = False
    try:
        gdf = sg.to_gdf([geoseries_with_crs, geoseries, geoseries])
    except TypeError:
        failed = True
    assert failed

    gdf = sg.to_gdf([(10, 60), (1, 1)], crs=4326)
    gdf = sg.to_gdf(gdf.geometry, crs=25833)
    assert gdf.crs == 25833

    assert sg.to_gdf(geoseries).index.to_list() == [0, 1]
    geoseries.index = [1, 2]
    assert sg.to_gdf(geoseries).index.to_list() == [1, 2]
    assert not sg.to_gdf(geoseries).geometry.isna().sum()


def _df():
    gdf = sg.to_gdf([(10, 60), (1, 1)])
    df = pd.DataFrame(gdf)
    buffered = sg.to_gdf(df).buffer(10)
    assert len(buffered) == 2
    assert not buffered.geometry.isna().sum()


def _linestring():
    coords = [(10, 60), (1, 1)]
    gdf = sg.to_gdf(coords)
    assert not gdf.length.sum()

    gdf = sg.to_gdf(LineString(coords))
    assert gdf.length.sum()


if __name__ == "__main__":
    test_to_gdf()
