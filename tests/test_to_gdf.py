# %%
import sys
from json import loads
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
    _json()

    _incorrect_geom_col()

    _series_like()

    _dflike_geom_col()

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

    _bbox()


def _bbox():
    gdf = sg.to_gdf(
        [600000.0, 7490220.0, 709800.0, 7600020.0],
        crs=25833,
    )
    assert sg.get_geom_type(gdf) == "polygon", sg.get_geom_type(gdf)


def _json():
    gdf = pd.concat([sg.random_points(10), sg.random_points(10).pipe(sg.buff, 1)])
    gdf.crs = 25833
    gdf2 = sg.to_gdf(gdf.__geo_interface__, crs=25833)
    print(gdf2)
    assert sg.get_geom_type(gdf2) == "mixed", sg.get_geom_type(gdf2)
    assert gdf.shape == gdf2.shape
    assert gdf.crs.equals(gdf2.crs)

    gdf2 = sg.to_gdf(loads(gdf.to_json()))
    assert gdf.crs.equals(gdf2.crs)
    assert sg.get_geom_type(gdf) == "mixed", sg.get_geom_type(gdf)

    gdf2 = sg.to_gdf(loads(gdf.to_json())["features"])
    assert sg.get_geom_type(gdf) == "mixed", sg.get_geom_type(gdf)

    gdf2 = sg.to_gdf([x["geometry"] for x in loads(gdf.to_json())["features"]])
    gdf2 = sg.to_gdf(loads(gdf.to_json())["features"][0]["geometry"])
    assert sg.get_geom_type(gdf2) == "point", sg.get_geom_type(gdf2)
    gdf2 = sg.to_gdf(loads(gdf.to_json())["features"][0])
    assert sg.get_geom_type(gdf2) == "point", sg.get_geom_type(gdf2)


def _series_like():
    should_equal = gpd.GeoDataFrame(
        {"geometry": [Point(10, 60), Point(11, 59)]}, geometry="geometry", index=[1, 3]
    )
    dict_ = {1: (10, 60), 3: (11, 59)}
    gdf = sg.to_gdf(dict_, crs=4326)
    assert gdf.equals(should_equal)

    series = pd.Series([(10, 60), (11, 59)], index=[1, 3])
    gdf = sg.to_gdf(series, crs=4326)
    assert gdf.equals(should_equal)


def _dflike_single_col():
    single_key = {"geom_col_name": [(10, 60), (11, 59)]}
    print(single_key)
    gdf = sg.to_gdf(single_key)
    print(gdf)
    assert gdf.shape == (2, 1), gdf.shape
    assert gdf.index.to_list() == [0, 1]
    assert not gdf.geometry.isna().any(), gdf
    assert gdf.columns.to_list() == ["geom_col_name"]

    df = pd.DataFrame(single_key)
    print(df)
    gdf2 = sg.to_gdf(df)
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

    # Invalid df becuse unequal length
    dict_ = {"col": [1, 2, 3], "geometry": [(10, 60), (11, 59)]}
    with pytest.raises(ValueError):
        sg.to_gdf(dict_)


def _preserves_index():
    should_equal = gpd.GeoDataFrame(
        {"col": [1, 2], "geometry": [Point(10, 60), Point(11, 59)]}, geometry="geometry"
    )
    assert not should_equal.col.isna().sum(), should_equal
    assert not should_equal.geometry.isna().sum(), should_equal

    dict_ = {"col": [1, 2], "geometry": [(10, 60), (11, 59)]}
    gdf = sg.to_gdf(dict_)
    assert gdf.equals(should_equal), gdf

    df = pd.DataFrame(dict_)
    gdf = sg.to_gdf(df)
    assert gdf.equals(should_equal), gdf

    index = [1, 3]
    should_equal.index = index
    assert list(should_equal.index) == index
    assert not should_equal.col.isna().sum(), should_equal
    assert not should_equal.geometry.isna().sum(), should_equal

    # setting index in DataFrame
    df = pd.DataFrame(dict_, index=index)
    gdf = sg.to_gdf(df)
    assert gdf.equals(should_equal), gdf

    # setting index in to_gdf
    gdf = sg.to_gdf(dict_, index=index)
    assert gdf.equals(should_equal), gdf

    # setting index in to_gdf when df has different index should give NA
    df = pd.DataFrame(dict_)
    gdf = sg.to_gdf(df, index=[1, 3])
    assert gdf.col.isna().sum() == 1, gdf
    assert gdf.geometry.isna().sum() == 1, gdf

    # the above should be same as calling DataFrame twice with different index
    df = pd.DataFrame(sg.to_gdf(dict_, geometry="geometry"), index=index)
    gdf2 = gpd.GeoDataFrame(df, geometry="geometry")
    assert gdf2.col.isna().sum() == 1, gdf2
    assert gdf2.geometry.isna().sum() == 1, gdf2

    # using a non-pandas type and spefifying index should work
    gdf = sg.to_gdf([(0, 0), (1, 1)], index=pd.Index([1, 3]))
    assert list(gdf.index) == [1, 3]


def _incorrect_geom_col():
    dict_ = {"col": [1, 2], "geom": [(10, 60), (11, 59)]}
    gdf = sg.to_gdf(dict_, geometry=["geom"])
    assert gdf.shape == (2, 2)

    df = pd.DataFrame(dict_)
    gdf = sg.to_gdf(df, geometry="geom")
    assert gdf.shape == (2, 2)

    # this should work with dict and series, but not DataFrame because of wrong geometry column
    print(sg.to_gdf(dict_))
    print(sg.to_gdf(pd.Series(dict_)))

    with pytest.raises(ValueError):
        print(sg.to_gdf(df))
    with pytest.raises(ValueError):
        print(sg.to_gdf(dict_, geometry="geometry"))
    with pytest.raises(ValueError):
        print(sg.to_gdf(df, geometry="geometry"))


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

    # should also work with string
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

    xs = np.random.rand(100)
    ys = np.random.rand(100)
    points_df = pd.DataFrame({"x": xs, "y": ys})
    points = sg.to_gdf(points_df, geometry=["x", "y"])


def _iterators():
    set_ = {(1, 60), (5, 10), (7, 10)}
    gdf = sg.to_gdf(set_, crs=4326)
    assert not gdf.geometry.isna().sum()
    assert gdf.shape == (3, 1)

    geom_array_ = gdf.geometry.values
    gdf2 = sg.to_gdf(geom_array_)
    assert gdf2.equals(gdf)

    np_array = np.array(gdf.geometry.values)
    gdf3 = sg.to_gdf(np_array)
    assert gdf.equals(gdf3)

    list_ = [x for x in gdf.geometry.values]
    gdf4 = sg.to_gdf(list_)
    assert gdf.equals(gdf4)

    generator_ = (x for x in gdf.geometry.values)
    gdf5 = sg.to_gdf(generator_)
    assert gdf.equals(gdf5)

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

    gdf = sg.to_gdf(gdf.geometry, geometry="geom", crs=25833)
    assert list(gdf.columns) == ["geom"]


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
