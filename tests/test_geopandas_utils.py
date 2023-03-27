# %%

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Polygon
from shapely.wkt import loads


src = str(Path(__file__).parent.parent) + "/src"

sys.path.insert(0, src)

import sgis as sg


def test_snap_and_get_ids():
    points = sg.to_gdf([(0, 0), (1, 1)])
    to = sg.to_gdf([(2, 2), (3, 3)])
    to["snap_idx"] = to.index

    snapped = sg.snap_and_get_ids(points, to, id_col="snap_idx")
    assert len(snapped) == 2
    assert all(snapped.snap_idx == 0)

    # with max_dist
    snapped = sg.snap_and_get_ids(points, to, id_col="snap_idx", max_dist=1.5)
    assert len(snapped) == 2
    assert not all([list(geom.coords)[0] == (2, 2) for geom in snapped.geometry])
    assert any([list(geom.coords)[0] == (2, 2) for geom in snapped.geometry])

    # with identical distances
    point = sg.to_gdf([0, 0])
    to = sg.to_gdf([(0, 1), (1, 0)])
    to["snap_idx"] = to.index
    snapped = sg.snap_and_get_ids(point, to, id_col="snap_idx")
    assert len(snapped) == 1
    assert all(snapped.snap_idx == 0)
    assert all([list(geom.coords)[0] == (0, 1) for geom in snapped.geometry])

    # opposite order of to coords
    to = sg.to_gdf([(1, 0), (0, 1)])
    to["snap_idx"] = to.index
    snapped = sg.snap_and_get_ids(point, to, id_col="snap_idx")
    assert len(snapped) == 1
    assert all(snapped.snap_idx == 1)
    assert all([list(geom.coords)[0] == (0, 1) for geom in snapped.geometry])

    # duplicate geometries in 'to'
    point = sg.to_gdf([0, 0])
    to = sg.to_gdf([(0, 1), (0, 1)])
    to["snap_idx"] = to.index
    snapped = sg.snap_and_get_ids(point, to, id_col="snap_idx")
    assert len(snapped) == 2
    assert any(snapped.snap_idx == 0)
    assert any(snapped.snap_idx == 1)
    assert all([list(geom.coords)[0] == (0, 1) for geom in snapped.geometry])


def test_buffdissexp(gdf_fixture):
    for distance in [1, 10, 100, 1000, 10000]:
        copy = gdf_fixture.copy()[["geometry"]]
        copy = sg.buff(copy, distance)
        copy = copy.dissolve()
        copy = copy.explode(ignore_index=True)

        areal1 = copy.area.sum()
        lengde1 = copy.length.sum()

        copy = sg.buffdissexp(gdf_fixture, distance)

        assert (
            areal1 == copy.area.sum() and lengde1 == copy.length.sum()
        ), "ulik lengde/areal"

    sg.buffdissexp(gdf_fixture, 100)


def test_geos(gdf_fixture):
    copy = sg.buffdissexp(gdf_fixture, 25)
    assert len(copy) == 4, "feil antall rader. Noe galt/nytt med GEOS' GIS-algoritmer?"
    assert (
        round(copy.area.sum(), 5) == 1035381.10389
    ), "feil areal. Noe galt/nytt med GEOS' GIS-algoritmer?"
    assert (
        round(copy.length.sum(), 5) == 16689.46148
    ), "feil lengde. Noe galt/nytt med GEOS' GIS-algoritmer?"


def test_close_all_holes(gdf_fixture):
    p = gdf_fixture.loc[gdf_fixture.geom_type == "Point"]
    buff1 = sg.buff(p, 100)
    buff2 = sg.buff(p, 200)
    rings_with_holes = sg.overlay(buff2, buff1, how="difference")

    holes_closed = sg.close_all_holes(rings_with_holes)
    assert sum(holes_closed.area) > sum(rings_with_holes.area)

    holes_closed = sg.close_small_holes(rings_with_holes, max_area=10000 * 1_000_000)
    assert sum(holes_closed.area) > sum(rings_with_holes.area)

    holes_not_closed = sg.close_small_holes(rings_with_holes, max_area=1)
    assert sum(holes_not_closed.area) == sum(rings_with_holes.area)


def test_clean_clip():
    p = sg.random_points(100)
    buff1 = sg.buff(p, 100)
    buff2 = sg.buff(p, 200)

    clipped = buff1.clip(buff2)
    clipped2 = sg.clean_clip(buff1, buff2)
    assert clipped.equals(clipped2)


def test_clean():
    invalid_geometry = sg.to_gdf(
        Polygon([(0, 1), (0, 0), (0, 1), (1, 1), (1, 2), (0, 2), (0, 0)])
    )

    empty_geometry = sg.to_gdf("POINT (0 0)").pipe(sg.buff, 0)

    missing_geometry = gpd.GeoDataFrame(
        {"geometry": [None]}, geometry="geometry", crs=25833
    )

    problematic_geometries = pd.concat(
        [invalid_geometry, missing_geometry, empty_geometry], ignore_index=True
    )

    assert len(problematic_geometries) == 3
    gdf = sg.clean_geoms(problematic_geometries)
    gdf = sg.to_single_geom_type(gdf, "polygon")
    print(gdf)
    assert len(gdf) == 1
    assert sg.get_geom_type(gdf) == "polygon"
    ser = sg.clean_geoms(problematic_geometries.geometry)
    ser = sg.to_single_geom_type(ser, "polygon")
    assert len(ser) == 1
    assert sg.get_geom_type(ser) == "polygon"

    valid_geometry = sg.to_gdf([0, 1])
    problematic_geometries = pd.concat(
        [problematic_geometries, valid_geometry], ignore_index=True
    )
    gdf = sg.clean_geoms(problematic_geometries)
    assert len(gdf) == 2
    gdf = sg.to_single_geom_type(gdf, "polygon")
    print(gdf)
    assert sg.get_geom_type(gdf) == "polygon"
    assert len(gdf) == 1


def sjoin_overlay(gdf_fixture):
    gdf1 = sg.buff(gdf_fixture, 25)
    gdf2 = sg.buff(gdf_fixture, 100)
    gdf2["nykoll"] = 1
    gdf = sg.sjoin(gdf1, gdf2)
    assert all(col in ["geometry", "numcol", "txtcol", "nykoll"] for col in gdf.columns)
    assert not any(
        col not in list(gdf.columns)
        for col in ["geometry", "numcol", "txtcol", "nykoll"]
    )
    assert len(gdf) == 25
    gdf = sg.overlay(gdf1, gdf2)
    assert all(col in ["geometry", "numcol", "txtcol", "nykoll"] for col in gdf.columns)
    assert not any(
        col not in list(gdf.columns)
        for col in ["geometry", "numcol", "txtcol", "nykoll"]
    )
    assert len(gdf) == 25

    gdf = sg.overlay_update(gdf2, gdf1)
    assert list(gdf.columns) == ["geometry", "numcol", "txtcol", "nykoll"]
    assert len(gdf) == 18


def test_get_neighbor_indices():
    points = sg.to_gdf([(0, 0), (0.5, 0.5), (2, 2)])
    p1 = points.iloc[[0]]

    assert sg.get_neighbor_indices(p1, points) == [0]
    assert sg.get_neighbor_indices(p1, points, max_dist=1) == [0, 1]
    assert sg.get_neighbor_indices(p1, points, max_dist=3) == [0, 1, 2]


def test_get_neighbor_ids():
    points = sg.to_gdf([(0, 0), (0.5, 0.5), (2, 2)])
    p1 = points.iloc[[0]]
    points["id_col"] = [*"abc"]

    assert sg.get_neighbor_ids(p1, points, id_col="id_col") == ["a"]
    assert sg.get_neighbor_ids(p1, points, max_dist=1, id_col="id_col") == ["a", "b"]
    assert sg.get_neighbor_ids(p1, points, max_dist=3, id_col="id_col") == [
        "a",
        "b",
        "c",
    ]


def test_snap():
    point = sg.to_gdf([0, 0])
    points = sg.to_gdf([(0, 0), (1, 0), (2, 0), (3, 0)])
    points["idx"] = points.index

    snapped = sg.snap_all(points, point)
    print(snapped)
    print([geom.x == 0 and geom.y == 0 for geom in snapped.geometry])
    assert all(geom.x == 0 and geom.y == 0 for geom in snapped.geometry)

    snapped = sg.snap_within_distance(points, point, 10, distance_col="snap_distance")
    print(snapped)
    assert [geom.x == 0 and geom.y == 0 for geom in snapped.geometry]
    assert all(geom.x == 0 and geom.y == 0 for geom in snapped.geometry)
    assert snapped.snap_distance.notna().sum() == 3

    snapped = sg.snap_within_distance(
        points, point, 1.001, distance_col="snap_distance"
    )
    print(snapped)
    assert sum([geom.x == 0 and geom.y == 0 for geom in snapped.geometry]) == 2
    assert snapped.snap_distance.notna().sum() == 1


def test_to_multipoint(gdf_fixture):
    mp = sg.to_multipoint(gdf_fixture)
    assert mp.length.sum() == 0
    mp = sg.to_multipoint(gdf_fixture.geometry)
    assert mp.length.sum() == 0
    mp = sg.to_multipoint(gdf_fixture.unary_union)
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
