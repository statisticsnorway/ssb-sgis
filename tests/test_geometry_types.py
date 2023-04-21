import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString


src = str(Path(__file__).parent.parent) + "/src"

sys.path.insert(0, src)

import sgis as sg


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
