# %%
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString


src = str(Path(__file__).parent.parent) + "/src"

sys.path.insert(0, src)

import sgis as sg


def create_all_geometry_types():
    from shapely.geometry import LinearRing, LineString

    point = sg.to_gdf([(0, 0)])
    multipoint = sg.to_gdf([(10, 10), (11, 11)]).dissolve()
    line = sg.to_gdf(LineString([(20, 20), (21, 21)]))
    multiline = sg.to_gdf(
        [LineString([(30, 30), (31, 31)]), LineString([(32, 32), (33, 33)])]
    ).dissolve()
    polygon = sg.buff(sg.to_gdf([(40, 40)]), 0.25)
    multipolygon = sg.to_gdf([(50, 50), (51, 51)]).dissolve().pipe(sg.buff, 0.25)
    ring = sg.to_gdf(LinearRing([(60, 60), (60, 61), (61, 61), (61, 60), (60, 60)]))
    gdf = pd.concat([point, multipoint, ring, line, multiline, polygon, multipolygon])
    collection = gdf.dissolve()
    return pd.concat([gdf, collection], ignore_index=True)


def test_all_geom_types():
    gdf = create_all_geometry_types()
    assert sg.get_geom_type(gdf) == "mixed"
    assert not sg.is_single_geom_type(gdf)
    assert len(gdf) == 8, len(gdf)

    points = sg.to_single_geom_type(gdf, "point")
    assert sg.get_geom_type(points) == "point"
    assert sg.is_single_geom_type(points)
    # 5 because the geometrycollection is exploded into three points
    assert len(points) == 5, points
    assert len(points.index.unique()) == 3

    polygons = sg.to_single_geom_type(gdf, "polygon")
    # same assertions for polygons
    assert sg.get_geom_type(polygons) == "polygon"
    assert sg.is_single_geom_type(polygons)
    assert len(polygons) == 5, polygons
    assert len(polygons.index.unique()) == 3, polygons.index.unique()

    lines = sg.to_single_geom_type(gdf, "line")
    # one more geom_type for lines (linearring)
    assert sg.get_geom_type(lines) == "line"
    assert sg.is_single_geom_type(lines)
    assert len(lines) == 7, lines
    assert len(lines.index.unique()) == 4, lines.index.unique()

    index = sorted(set(list(points.index) + list(lines.index) + list(polygons.index)))
    assert index == sorted(gdf.index), print(index, sorted(gdf.index))


def test_geom_types():
    point = sg.to_gdf([0, 0])
    assert sg.get_geom_type(point) == "point"
    assert sg.is_single_geom_type(point)

    line = sg.to_gdf(LineString([(2, 2), (3, 3)]))
    assert sg.get_geom_type(line) == "line"
    assert sg.is_single_geom_type(line)

    polygon = sg.buff(point, 1)
    assert sg.get_geom_type(polygon) == "polygon"
    assert sg.is_single_geom_type(polygon)

    gdf = pd.concat([point, line, polygon])
    assert sg.get_geom_type(gdf) == "mixed"
    assert not sg.is_single_geom_type(gdf)

    as_point = sg.to_single_geom_type(gdf, "point")
    assert len(as_point) == 1
    assert sg.get_geom_type(as_point) == "point"
    assert not sg.is_single_geom_type(gdf)
    assert as_point.length.sum() == 0
    assert as_point.area.sum() == 0

    as_line = sg.to_single_geom_type(gdf, "line")
    assert len(as_line) == 1
    assert sg.get_geom_type(as_line) == "line"
    assert not sg.is_single_geom_type(gdf)
    assert as_line.length.sum() != 0
    assert as_line.area.sum() == 0

    as_polygon = sg.to_single_geom_type(gdf, "polygon")
    assert len(as_polygon) == 1
    assert sg.get_geom_type(as_polygon) == "polygon"
    assert not sg.is_single_geom_type(gdf)
    assert as_polygon.length.sum() != 0
    assert as_polygon.area.sum() != 0


if __name__ == "__main__":
    test_geom_types()
