# %%
import sys
from pathlib import Path

import pandas as pd
import shapely
from helpers import create_all_geometry_types
from shapely.geometry import LineString, MultiPoint, Point


src = str(Path(__file__).parent.parent) + "/src"

sys.path.insert(0, src)

import sgis as sg


def test_all_geom_types():
    gdf = create_all_geometry_types()
    assert sg.get_geom_type(gdf) == "mixed"
    assert not sg.is_single_geom_type(gdf)
    assert len(gdf) == 8, len(gdf)

    singlepart = sg.make_all_singlepart(gdf)
    assert len(singlepart) == 20, len(singlepart)

    singlepart = sg.make_all_singlepart(gdf, ignore_index=True)
    assert len(singlepart) == 20, len(singlepart)

    singlepart = sg.make_all_singlepart(gdf, ignore_index=True, index_parts=True)
    assert len(singlepart) == 20, len(singlepart)

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

    assert (res := sg.to_single_geom_type(gdf.geometry.values, "point")).tolist() == [
        Point([0, 0]),
        MultiPoint([(10, 10), (11, 11)]),
        MultiPoint([(0, 0), (10, 10), (11, 11)]),
    ], res
    assert (res := sg.to_single_geom_type(gdf.unary_union, "point")) == MultiPoint(
        ([0, 0], (10, 10), (11, 11))
    ), res

    assert (
        res := [x.wkt for x in sg.to_single_geom_type(gdf.geometry.values, "line")]
    ) == [
        "LINESTRING (60 60, 60 61, 61 61, 61 60, 60 60)",
        "LINESTRING (20 20, 21 21)",
        "MULTILINESTRING ((30 30, 31 31), (32 32, 33 33))",
        "MULTILINESTRING ((60 60, 60 61, 61 61, 61 60, 60 60), (20 20, 21 21), (30 30, 31 31), (32 32, 33 33))",
    ], res
    assert (
        res := sg.to_single_geom_type(gdf.unary_union, "line")
    ) == shapely.wkt.loads(
        "MULTILINESTRING ((60 60, 60 61), (60 61, 61 61), (61 61, 61 60), (61 60, 60 60), (20 20, 21 21), (30 30, 31 31), (32 32, 33 33))"
    ), res

    assert all(
        "POLYGON" in x.wkt
        for x in sg.to_single_geom_type(gdf.geometry.values, "polygon")
    )

    assert "POLYGON" in sg.to_single_geom_type(gdf.unary_union, "polygon").wkt


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
    test_all_geom_types()
    test_geom_types()
