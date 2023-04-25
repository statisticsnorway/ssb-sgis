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


def test_close_holes():
    p = sg.to_gdf([0, 0])

    buff1 = sg.buffdissexp(p, 100)

    no_holes_closed = sg.close_all_holes(buff1)
    assert round(sum(no_holes_closed.area), 3) == round(sum(buff1.area), 3)
    no_holes_closed = sg.close_small_holes(buff1, 10000 * 1_000_000)
    assert round(sum(no_holes_closed.area), 3) == round(sum(buff1.area), 3)

    buff2 = sg.buffdissexp(p, 200)
    rings_with_holes = sg.clean_overlay(buff2, buff1, how="difference")

    # run this for different geometry input types
    def _close_the_holes(rings_with_holes):
        holes_closed = sg.close_all_holes(rings_with_holes)
        if hasattr(holes_closed, "area"):
            assert sum(holes_closed.area) > sum(rings_with_holes.area)

        holes_closed2 = sg.close_small_holes(
            rings_with_holes, max_area=10000 * 1_000_000
        )
        if hasattr(holes_closed, "area"):
            assert round(sum(holes_closed2.area), 3) == round(sum(holes_closed.area), 3)
            assert sum(holes_closed2.area) > sum(rings_with_holes.area)

        holes_not_closed = sg.close_small_holes(rings_with_holes, max_area=1)
        if hasattr(holes_closed, "area"):
            assert sum(holes_not_closed.area) == sum(rings_with_holes.area)
        else:
            assert sum(gpd.GeoSeries(holes_not_closed).area) == sum(
                rings_with_holes.area
            )

    _close_the_holes(rings_with_holes)
    _close_the_holes(rings_with_holes.geometry)


if __name__ == "__main__":
    test_close_holes()
