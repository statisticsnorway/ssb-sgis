# %%

import sys
from pathlib import Path

import pytest


src = str(Path(__file__).parent.parent) + "/src"

sys.path.insert(0, src)

import sgis as sg


def test_buffdissexp(gdf_fixture):
    with pytest.raises(ValueError):
        sg.buffdissexp(gdf_fixture, 10, by="txtcol", ignore_index=True)

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
    with pytest.raises(ValueError):
        sg.buffdiss(gdf_fixture, 10, by="txtcol", ignore_index=True)

    sg.buffdiss(gdf_fixture, 10, ignore_index=True)

    for distance in [1, 10, 100, 1000, 10000]:
        copy = gdf_fixture.copy()

        # with geopandas
        copy["geometry"] = copy.buffer(distance, resolution=50).make_valid()
        copy = copy.dissolve(by="txtcol")
        copy["geometry"] = copy.make_valid()

        copy2 = sg.buffdiss(gdf_fixture, distance, by="txtcol")

        assert copy.equals(copy2)


def test_dissexp(gdf_fixture):
    with pytest.raises(ValueError):
        sg.dissexp(gdf_fixture, by="txtcol", ignore_index=True)

    sg.dissexp(gdf_fixture, ignore_index=True)

    copy = gdf_fixture.copy()

    # with geopandas
    copy = copy.dissolve(by="txtcol")
    copy["geometry"] = copy.make_valid()
    copy = copy.explode(index_parts=False)

    copy2 = sg.dissexp(gdf_fixture, by="txtcol")

    assert copy.equals(copy2), (copy, copy2)
