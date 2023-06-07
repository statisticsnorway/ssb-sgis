# %%

import sys
import timeit
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest


src = str(Path(__file__).parent.parent) + "/src"

sys.path.insert(0, src)

import sgis as sg


def test_buffdissexp(gdf_fixture):
    sg.buffdissexp(gdf_fixture, 10, ignore_index=True)

    for distance in [1, 10, 100, 1000, 10000]:
        copy = gdf_fixture.copy()

        # with geopandas
        copy["geometry"] = copy.buffer(distance, resolution=50).make_valid()
        copy = copy.dissolve(by="txtcol")
        copy["geometry"] = copy.make_valid()
        copy = copy.explode(index_parts=False)

        copy2 = sg.buffdissexp(gdf_fixture, distance, by="txtcol")
        assert copy.equals(copy2)


def test_buffdiss(gdf_fixture):
    sg.buffdiss(gdf_fixture, 10)

    for distance in [1, 10, 100, 1000, 10000]:
        copy = gdf_fixture.copy()

        # with geopandas
        copy["geometry"] = copy.buffer(distance, resolution=50).make_valid()
        copy = copy.dissolve(by="txtcol")
        copy["geometry"] = copy.make_valid()

        copy2 = sg.buffdiss(gdf_fixture, distance, by="txtcol")

        assert copy.equals(copy2)


def test_dissexp(gdf_fixture):
    sg.dissexp(gdf_fixture, ignore_index=True)

    copy = gdf_fixture.copy()

    # with geopandas
    copy = copy.dissolve(by="txtcol")
    copy["geometry"] = copy.make_valid()
    copy = copy.explode(index_parts=False)

    copy2 = sg.dissexp(gdf_fixture, by="txtcol")

    assert copy.equals(copy2), (copy, copy2)


def test_buffdissexp_index():
    gdf = sg.to_gdf([(0, 0), (1, 1), (2, 2)]).assign(cat=[*"aab"])

    # when not dissolving by column, index should be 0, 1, 2...
    singlepart = sg.buffdissexp(gdf, 0.1)
    assert list(singlepart.index) == [0, 1, 2], singlepart
    assert list(singlepart.columns) == ["cat", "geometry"], singlepart

    # when by and not as_index, index should be 0, 1, 2...
    singlepart = sg.buffdissexp(gdf, 0.1, by="cat", as_index=False)
    assert list(singlepart.index) == [0, 1, 2], singlepart
    assert list(singlepart.cat) == [*"aab"], singlepart
    assert list(singlepart.columns) == ["cat", "geometry"], singlepart

    # to respect geopandas, only specifying 'by' should give 'by' column as index
    singlepart = sg.buffdissexp(gdf, 0.1, by="cat")
    assert list(singlepart.index) == [*"aab"], singlepart
    assert list(singlepart.columns) == ["geometry"], singlepart

    singlepart = sg.buffdissexp(gdf[["geometry"]], 0.1)
    assert list(singlepart.index) == [0, 1, 2], singlepart
    assert list(singlepart.columns) == ["geometry"], singlepart

    singlepart = sg.buffdissexp(gdf, 1)
    assert list(singlepart.index) == [0], singlepart
    assert (list(singlepart.columns)) == (["cat", "geometry"]), singlepart.columns

    # if the index is set, it will be gone after dissolve (same in geopandas)
    singlepart = sg.buffdissexp(gdf.set_index("cat"), 0.1)
    assert list(singlepart.index) == [0, 1, 2], singlepart
    assert list(singlepart.columns) == ["geometry"], singlepart

    singlepart = sg.buffdissexp(gdf, 0.1, by="cat", ignore_index=True)
    assert list(singlepart.index) == [0, 1, 2], singlepart


def test_dissexp_index():
    gdf = sg.to_gdf([(0, 0), (1, 1), (2, 2)]).assign(cat=[*"aab"])

    # this is what is returned with the geopandas defaults
    singlepart = sg.dissexp(gdf, ignore_index=False)
    assert list(singlepart.index) == [0, 0, 0], singlepart
    assert (list(singlepart.columns)) == (["cat", "geometry"]), singlepart

    # if dissolve by and default explode, the by column is removed completely
    singlepart = sg.dissexp(gdf, by="cat", ignore_index=True)
    assert list(singlepart.index) == [0, 1, 2], singlepart
    assert list(singlepart.columns) == ["geometry"], singlepart

    # when not dissolving by column, index should be 0, 1, 2...
    singlepart = sg.dissexp(gdf)
    assert list(singlepart.index) == [0, 1, 2], singlepart
    assert list(singlepart.columns) == ["cat", "geometry"], singlepart

    # when by and not as_index, index should be 0, 1, 2...
    singlepart = sg.dissexp(gdf, by="cat", as_index=False)
    assert list(singlepart.index) == [0, 1, 2], singlepart
    assert list(singlepart.cat) == [*"aab"], singlepart
    assert list(singlepart.columns) == ["cat", "geometry"], singlepart

    # to respect geopandas, only specifying 'by' should give 'by' column as index
    singlepart = sg.dissexp(gdf, by="cat")
    assert list(singlepart.index) == [*"aab"], singlepart
    assert list(singlepart.columns) == ["geometry"], singlepart

    singlepart = sg.dissexp(gdf[["geometry"]])
    assert list(singlepart.index) == [0, 1, 2], singlepart
    assert list(singlepart.columns) == ["geometry"], singlepart

    singlepart = sg.dissexp(gdf.set_index("cat"))
    assert list(singlepart.index) == [0, 1, 2], singlepart
    assert list(singlepart.columns) == ["geometry"], singlepart

    singlepart = sg.dissexp(gdf, ignore_index=False)
    assert list(singlepart.index) == [0, 0, 0], singlepart
    assert list(singlepart.columns) == ["cat", "geometry"], singlepart

    singlepart = sg.dissexp(gdf, by="cat", ignore_index=True)
    assert list(singlepart.index) == [0, 1, 2], singlepart


if __name__ == "__main__":
    import cProfile

    test_dissexp_index()
    test_buffdissexp_index()
