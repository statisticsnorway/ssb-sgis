# %%

import sys
from pathlib import Path

import geopandas as gpd
from shapely import union_all
from shapely.geometry import LineString

src = str(Path(__file__).parent).replace("tests", "") + "src"


sys.path.insert(0, src)


import sgis as sg


def test_get_centerline():
    from oslo import points_oslo
    from oslo import roads_oslo

    circle = sg.to_gdf([0, 0]).buffer(1)
    centerline = sg.get_rough_centerlines(circle, 5)
    sg.qtm(centerline, circle)

    circle_with_hole = circle.difference(
        union_all(sg.to_gdf([0, 0]).buffer(0.5).geometry.values)
    )
    centerline = sg.get_rough_centerlines(circle, 5)
    sg.qtm(centerline, circle_with_hole)

    cross = sg.to_gdf(
        LineString(
            [
                (0, 0),
                (0, 20),
                (0, 10),
                (0, 0),
                (-10, 0),
                (0, 0),
                (10, 0),
                (0, 0),
                (0, -10),
            ]
        )
    ).pipe(sg.buff, 0.1, resolution=10)

    centerline = sg.get_rough_centerlines(cross, 10)

    sg.qtm(centerline, cross)

    assert (geom_type := sg.get_geom_type(centerline)) == "line", geom_type

    # TODO add this assert
    """assert centerline.union_all().intersects(
        Point(0, 0).buffer(0.1)
    ), centerline.union_all()"""

    roads = roads_oslo()
    p = points_oslo()
    roads = sg.clean_clip(roads, p.geometry.iloc[0].buffer(100))
    roads = sg.buffdissexp(roads, 2, resolution=1)

    centerlines = sg.get_rough_centerlines(roads, 10)
    sg.qtm(roads, centerlines)

    df = gpd.read_parquet(Path(__file__).parent / "testdata" / "gaps.parquet")
    for i in [50, 20, 5]:
        centerlines = sg.get_rough_centerlines(df, i)
        sg.qtm(df, centerlines)


if __name__ == "__main__":
    test_get_centerline()
