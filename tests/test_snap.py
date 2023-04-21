# %%

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
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


def test_snap_series():
    point = sg.to_gdf([0, 0])
    points = sg.to_gdf([(0, 0), (1, 0), (2, 0), (3, 0)])
    points["idx"] = points.index

    snapped = sg.snap_all(points.geometry, point.geometry)
    print(snapped)
    print([geom.x == 0 and geom.y == 0 for geom in snapped.geometry])
    assert all(geom.x == 0 and geom.y == 0 for geom in snapped.geometry)

    snapped = sg.snap_within_distance(
        points.geometry, point.geometry, 10, distance_col="snap_distance"
    )
    print(snapped)
    assert [geom.x == 0 and geom.y == 0 for geom in snapped.geometry]
    assert all(geom.x == 0 and geom.y == 0 for geom in snapped.geometry)
    assert snapped.snap_distance.notna().sum() == 3
    snapped = sg.snap_within_distance(
        points.geometry, point.geometry, 1.001, distance_col="snap_distance"
    )
    print(snapped)
    assert sum([geom.x == 0 and geom.y == 0 for geom in snapped.geometry]) == 2
    assert snapped.snap_distance.notna().sum() == 1

    poly = sg.to_gdf(Polygon([(10, 10), (110, 11), (11, 11), (11, 110)]))
    snapped = sg.snap_all(points.geometry, to=pd.concat([point, poly]).geometry)
    print([geom.x == 0 and geom.y == 0 for geom in snapped.geometry])
    assert all(geom.x == 0 and geom.y == 0 for geom in snapped.geometry)


if __name__ == "__main__":
    test_snap_series()
    test_snap()
