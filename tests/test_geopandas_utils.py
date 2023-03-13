# %%

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.wkt import loads


src = str(Path(__file__).parent.parent) + "/src"

sys.path.insert(0, src)

import sgis as gs


def test_snap_to():
    points = gs.to_gdf([(0, 0), (1, 1)])
    to = gs.to_gdf([(2, 2), (3, 3)])
    to["snap_to_idx"] = to.index

    # snap all points
    snapped = gs.snap_to(points, to)
    assert len(snapped) == 2
    assert all([list(geom.coords)[0] == (2, 2) for geom in snapped.geometry])

    # with id col
    snapped = gs.snap_to(points, to, id_col="snap_to_idx")
    assert len(snapped) == 2
    assert all(snapped.snap_to_idx == 0)

    # with max_dist
    snapped = gs.snap_to(points, to, id_col="snap_to_idx", max_dist=1.5)
    assert len(snapped) == 2
    assert not all([list(geom.coords)[0] == (2, 2) for geom in snapped.geometry])
    assert any([list(geom.coords)[0] == (2, 2) for geom in snapped.geometry])

    # with identical distances
    point = gs.to_gdf([0, 0])
    to = gs.to_gdf([(0, 1), (1, 0)])
    to["snap_to_idx"] = to.index
    snapped = gs.snap_to(point, to, id_col="snap_to_idx")
    assert len(snapped) == 1
    assert all(snapped.snap_to_idx == 0)
    assert all([list(geom.coords)[0] == (0, 1) for geom in snapped.geometry])

    # opposite order of to coords
    to = gs.to_gdf([(1, 0), (0, 1)])
    to["snap_to_idx"] = to.index
    snapped = gs.snap_to(point, to, id_col="snap_to_idx")
    assert len(snapped) == 1
    assert all(snapped.snap_to_idx == 1)
    assert all([list(geom.coords)[0] == (0, 1) for geom in snapped.geometry])

    # duplicate geometries in 'to'
    point = gs.to_gdf([0, 0])
    to = gs.to_gdf([(0, 1), (0, 1)])
    to["snap_to_idx"] = to.index
    snapped = gs.snap_to(point, to, id_col="snap_to_idx")
    assert len(snapped) == 2
    assert any(snapped.snap_to_idx == 0)
    assert any(snapped.snap_to_idx == 1)
    assert all([list(geom.coords)[0] == (0, 1) for geom in snapped.geometry])


def test_buffdissexp(gdf_fixture):
    for distance in [1, 10, 100, 1000, 10000]:
        copy = gdf_fixture.copy()[["geometry"]]
        copy = gs.buff(copy, distance)
        copy = gs.diss(copy)
        copy = copy.explode(ignore_index=True)

        areal1 = copy.area.sum()
        lengde1 = copy.length.sum()

        copy = gs.buffdissexp(gdf_fixture, distance, copy=True)

        assert (
            areal1 == copy.area.sum() and lengde1 == copy.length.sum()
        ), "ulik lengde/areal"

    gs.buffdissexp(gdf_fixture, 100)
    gs.buffdissexp(gdf_fixture.geometry, 100)
    gs.buffdissexp(gdf_fixture.unary_union, 100)


def test_geos(gdf_fixture):
    copy = gs.buffdissexp(gdf_fixture, 25, copy=True)
    assert len(copy) == 4, "feil antall rader. Noe galt/nytt med GEOS' GIS-algoritmer?"
    assert (
        round(copy.area.sum(), 5) == 1035381.10389
    ), "feil areal. Noe galt/nytt med GEOS' GIS-algoritmer?"
    assert (
        round(copy.length.sum(), 5) == 16689.46148
    ), "feil lengde. Noe galt/nytt med GEOS' GIS-algoritmer?"


def test_aggfuncs(gdf_fixture):
    copy = gs.dissexp(gdf_fixture, by="txtcol", aggfunc="sum")
    assert (
        len(copy) == 11
    ), "dissexp by txtcol skal gi 11 rader, tre stykk linestrings..."

    copy = gs.buffdiss(gdf_fixture, 100, by="txtcol", aggfunc="sum", copy=True)
    assert (
        copy.numcol.sum()
        == gdf_fixture.numcol.sum()
        == sum([1, 2, 3, 4, 5, 6, 7, 8, 9])
    )

    copy = gs.buffdissexp(
        gdf_fixture, 100, by="txtcol", aggfunc=["sum", "mean"], copy=True
    )
    assert (
        "numcol_sum" in copy.columns and "numcol_mean" in copy.columns
    ), "kolonnene følger ikke mønstret 'kolonnenavn_aggfunc'"
    assert len(copy) == 6, "feil lengde"

    copy = gs.buffdissexp(
        gdf_fixture, 1000, by="txtcol", aggfunc=["sum", "mean"], copy=True
    )
    assert len(copy) == 4, "feil lengde"

    copy = gs.buffdissexp(gdf_fixture, 100, by="numcol", copy=True)
    assert len(copy) == 9, "feil lengde"

    copy = gs.buffdissexp(gdf_fixture, 100, by=["numcol", "txtcol"], copy=True)
    assert (
        "numcol" in copy.columns and "txtcol" in copy.columns
    ), "kolonnene mangler. Er de index?"
    assert len(copy) == 9, "feil lengde"


def test_close_holes(gdf_fixture):
    p = gdf_fixture.loc[gdf_fixture.geom_type == "Point"]
    buff1 = gs.buff(p, 100)
    buff2 = gs.buff(p, 200)
    rings_with_holes = gs.overlay(buff2, buff1, how="difference")
    holes_closed = gs.close_holes(rings_with_holes)
    assert sum(holes_closed.area) > sum(rings_with_holes.area)
    holes_closed = gs.close_holes(rings_with_holes, 10000)
    assert sum(holes_closed.area) > sum(rings_with_holes.area)
    holes_not_closed = gs.close_holes(rings_with_holes, 0.000001)
    assert sum(holes_not_closed.area) == sum(rings_with_holes.area)


def test_clean(gdf_fixture):
    missing = gpd.GeoDataFrame(
        {"geometry": [None, np.nan]}, geometry="geometry", crs=25833
    )
    empty = gpd.GeoDataFrame(
        {"geometry": gpd.GeoSeries(loads("POINT (0 0)")).buffer(0)},
        geometry="geometry",
        crs=25833,
    )
    gdf = gs.gdf_concat([gdf_fixture, missing, empty])
    assert len(gdf) == 12
    gdf2 = gs.clean_geoms(gdf_fixture)
    ser = gs.clean_geoms(gdf_fixture.geometry)
    assert len(gdf2) == 9
    assert len(ser) == 9


def sjoin_overlay(gdf_fixture):
    gdf1 = gs.buff(gdf_fixture, 25, copy=True)
    gdf2 = gs.buff(gdf_fixture, 100, copy=True)
    gdf2["nykoll"] = 1
    gdf = gs.sjoin(gdf1, gdf2)
    assert all(col in ["geometry", "numcol", "txtcol", "nykoll"] for col in gdf.columns)
    assert not any(
        col not in list(gdf.columns)
        for col in ["geometry", "numcol", "txtcol", "nykoll"]
    )
    assert len(gdf) == 25
    gdf = gs.overlay(gdf1, gdf2)
    assert all(col in ["geometry", "numcol", "txtcol", "nykoll"] for col in gdf.columns)
    assert not any(
        col not in list(gdf.columns)
        for col in ["geometry", "numcol", "txtcol", "nykoll"]
    )
    assert len(gdf) == 25

    gdf = gs.overlay_update(gdf2, gdf1)
    assert list(gdf.columns) == ["geometry", "numcol", "txtcol", "nykoll"]
    assert len(gdf) == 18


def test_copy(gdf_fixture):
    """
    Sjekk at copy-parametret i buff funker. Og sjekk pandas' copy-regler samtidig.
    """

    copy = gdf_fixture[gdf_fixture.area == 0]
    assert gdf_fixture.area.sum() != 0

    copy = gdf_fixture.loc[gdf_fixture.area == 0]
    assert gdf_fixture.area.sum() != 0
    assert copy.area.sum() == 0

    bufret = gs.buff(copy, 10, copy=True)
    assert copy.area.sum() == 0

    bufret = gs.buff(copy, 10, copy=False)
    assert copy.area.sum() != 0


def test_neighbors(gdf_fixture):
    naboer = gs.find_neighbors(
        gdf_fixture.iloc[[0]],
        possible_neighbors=gdf_fixture,
        id_col="numcol",
        max_dist=100,
    )
    naboer.sort()
    assert naboer == [1, 2], "feil i find_neighbors"
    naboer = gs.find_neighbors(
        gdf_fixture.iloc[[8]],
        possible_neighbors=gdf_fixture,
        id_col="numcol",
        max_dist=100,
    )
    naboer.sort()
    assert naboer == [4, 5, 7, 8, 9], "feil i find_neighbors"

    points = gs.to_gdf([(0, 0), (0.5, 0.5), (2, 2)])
    points["idx"] = points.index
    p1 = points.iloc[[0]]
    assert gs.find_neighbours(p1, points, id_col="idx") == [0]
    assert gs.find_neighbours(p1, points, id_col="idx", max_dist=1) == [0, 1]
    assert gs.find_neighbours(p1, points, id_col="idx", max_dist=3) == [0, 1, 2]


def test_to_gdf():
    geom = 10, 60
    print(gs.to_gdf(geom, crs=4326))
    geom = (10, 60, 1)
    print(gs.to_gdf(geom, crs=4326))

    geom = ((10, 60), (1, 1))
    print(gs.to_gdf(geom, crs=4326))

    geom = [[10, 60], [1, 1]]
    print(gs.to_gdf(geom, crs=4326))

    geom = [(10, 60), (1, 1)]
    gdf = gs.to_gdf(geom, crs=4326)

    print(gs.to_gdf([gdf.geometry, gdf.geometry, gdf.geometry]))
    coordsarray = gs.coordinate_array(gdf)
    print(gs.to_gdf(coordsarray, crs=4326))

    assert gs.to_gdf(gdf.geometry).crs == 4326
    assert gs.to_gdf(gdf.geometry, crs=4326).crs == 4326

    print(gs.to_gdf(LineString(geom), crs=4326))
    print(gs.to_gdf(gs.to_gdf(geom, crs=4326).geometry))

    geom = "POINT (60 10)"
    print(gs.to_gdf(geom, crs=4326))
    geom = b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00N@\x00\x00\x00\x00\x00\x00$@"
    print(gs.to_gdf(geom, crs=4326))

    geom = {1: (10, 60), 2: (11, 59)}
    gdf = gs.to_gdf(geom, crs=4326)
    print(gdf)
    assert list(gdf.index) == [1, 2]

    geom = {1: (10, 60), 2: (11, 59)}
    gdf = gs.to_gdf(geom, crs=4326)
    print(gdf)
    assert gdf.index.to_list() == [1, 2]


def test_snap(gdf_fixture):
    punkter = gdf_fixture[gdf_fixture.length == 0]
    annet = gdf_fixture[gdf_fixture.length != 0]
    snapped = gs.snap_to(punkter, annet, max_dist=None, copy=True)
    assert all(snapped.intersects(annet.buffer(1).unary_union))
    snapped = gs.snap_to(punkter, annet, max_dist=200, copy=True)
    assert sum(snapped.intersects(annet.buffer(1).unary_union)) == 3
    snapped = gs.snap_to(punkter, annet, max_dist=20, copy=True)
    assert sum(snapped.intersects(annet.buffer(1).unary_union)) == 1

    snapped = gs.snap_to(punkter, annet, max_dist=None, to_node=True, copy=True)

    assert all(
        geom in list(gs.to_multipoint(annet).explode().geometry)
        for geom in snapped.geometry
    )


def test_to_multipoint(gdf_fixture):
    mp = gs.to_multipoint(gdf_fixture)
    assert mp.length.sum() == 0
    mp = gs.to_multipoint(gdf_fixture.geometry)
    assert mp.length.sum() == 0
    mp = gs.to_multipoint(gdf_fixture.unary_union)
    assert mp.length == 0


def main():
    info = """
    The test was created 08.01.2023 with the following package versions.
    From C++: GEOS 3.11.1, PROJ 9.1.0, GDAL 3.6.1.
    From Python: geopandas 0.12.2, shapely 2.0.0, pyproj 3.4.1, pandas 1.5.2, numpy 1.24.
    """
    print(info)

    print("Versjoner nå:")
    from shapely.geos import geos_version

    geos_versjon = ".".join([str(x) for x in geos_version])
    print(f"{gpd.__version__ = }")
    print(f"{geos_versjon    = }")
    print(f"{pd.__version__  = }")
    print(f"{np.__version__  = }")

    src = str(Path(__file__).parent).strip("tests") + "src"

    sys.path.append(src)

    from conftest import make_gdf

    test_snap(make_gdf())

    print("Success")


if __name__ == "__main__":
    main()
