# %%

import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point


src = str(Path(__file__).parent).strip("tests") + "src"

import sys


sys.path.insert(0, src)


from geopandas import GeoDataFrame

import sgis as sg


def test_get_centerline():
    from oslo import points_oslo, roads_oslo

    circle = sg.to_gdf([0, 0]).buffer(1)
    centerline = sg.get_rough_centerlines(circle, 5)
    sg.qtm(centerline, circle)

    circle_with_hole = circle.difference(sg.to_gdf([0, 0]).buffer(0.5).unary_union)
    centerline = sg.get_rough_centerlines(circle, 5)
    sg.qtm(centerline, circle_with_hole)

    cross = sg.to_gdf(
        LineString(
            [
                (0, 0),
                (0, 2),
                (0, 1),
                (0, 0),
                (-1, 0),
                (0, 0),
                (1, 0),
                (0, 0),
                (0, -1),
            ]
        )
    ).pipe(sg.buff, 0.1, resolution=10)

    centerline = sg.get_rough_centerlines(cross, 3)

    sg.qtm(centerline, cross)

    assert (geom_type := sg.get_geom_type(centerline)) == "line", geom_type
    assert centerline.unary_union.intersects(
        Point(0, 0).buffer(0.1)
    ), centerline.unary_union

    roads = roads_oslo()
    p = points_oslo()
    roads = sg.clean_clip(roads, p.geometry.iloc[0].buffer(100))
    roads = sg.buffdissexp(roads, 2, resolution=1)

    centerlines = sg.get_rough_centerlines(roads, 10)
    sg.qtm(roads, centerlines)

    df = gpd.read_parquet(Path(__file__).parent / "testdata" / "gaps.parquet")
    for i in [50, 25, 10, 5]:
        centerlines = sg.get_rough_centerlines(df, i)
        sg.qtm(df, centerlines)


if __name__ == "__main__":
    import cProfile

    test_get_centerline()
    # cProfile.run("test_get_centerline()", sort="cumtime")
