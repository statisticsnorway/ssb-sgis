import sys
from pathlib import Path

import pandas as pd


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
