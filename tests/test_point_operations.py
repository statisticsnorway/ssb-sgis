# %%

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

src = str(Path(__file__).parent.parent) + "/src"

sys.path.insert(0, src)

import sgis as sg


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

    poly = sg.to_gdf(Polygon([(10, 10), (110, 11), (11, 11), (11, 110)]))
    snapped = sg.snap_all(points, to=pd.concat([point, poly]))
    print([geom.x == 0 and geom.y == 0 for geom in snapped.geometry])
    assert all(geom.x == 0 and geom.y == 0 for geom in snapped.geometry)


def test_snap_geoseries():
    point = sg.to_gdf([0, 0]).geometry
    points = sg.to_gdf([(0, 0), (1, 0), (2, 0), (3, 0)]).geometry

    snapped = sg.snap_all(points, point)
    assert isinstance(snapped, gpd.GeoSeries), snapped
    assert all(geom.x == 0 and geom.y == 0 for geom in snapped.geometry)

    snapped = sg.snap_within_distance(points, point, 10)
    assert isinstance(snapped, gpd.GeoSeries), snapped

    # should return GeoDataFrame when distance_col is specified
    snapped = sg.snap_within_distance(points, point, 10, distance_col="snap_distance")
    assert isinstance(snapped, gpd.GeoDataFrame)
    assert [geom.x == 0 and geom.y == 0 for geom in snapped.geometry]
    assert all(geom.x == 0 and geom.y == 0 for geom in snapped.geometry)
    assert snapped.snap_distance.notna().sum() == 3

    snapped = sg.snap_within_distance(
        points, point, 1.001, distance_col="snap_distance"
    )
    assert sum([geom.x == 0 and geom.y == 0 for geom in snapped.geometry]) == 2
    assert snapped.snap_distance.notna().sum() == 1

    poly = sg.to_gdf(Polygon([(10, 10), (110, 11), (11, 11), (11, 110)])).geometry
    snapped = sg.snap_all(points, to=pd.concat([point, poly]))
    print([geom.x == 0 and geom.y == 0 for geom in snapped.geometry])
    assert all(geom.x == 0 and geom.y == 0 for geom in snapped.geometry), print(snapped)


if __name__ == "__main__":
    test_snap_geoseries()
    test_snap()
