# %%
from pathlib import Path

import geopandas as gpd
import numpy as np


src = str(Path(__file__).parent).strip("tests") + "src"

import sys


sys.path.insert(0, src)

import sgis as sg


def test_qtm(points_oslo):
    points = points_oslo

    points = points.clip(points.iloc[[0]].buffer(500))

    points.geometry = points.buffer(np.arange(1, len(points) + 1) * 10)
    points["m2"] = points.area
    sg.qtm(points, "m2")

    # making a list of gdfs makes it hard/impossible to get the names as labels
    pointlist = ()
    for i in points.index:
        pointlist = pointlist + (points.loc[[i]],)
    sg.qtm(*pointlist, "m2")
    sg.qtm(*pointlist)

    p1 = points.iloc[[0]]
    p2 = points.iloc[[1]]
    p3 = points.iloc[[2]]
    sg.qtm(p1, p2, p3)
    p4 = points.iloc[[3]]
    p5 = points.iloc[[4]]
    p6 = points.iloc[[5]]
    p7 = points.iloc[[6]]
    sg.qtm(p1, p2, p3, p4, p5, p6, p7)
    sg.qtm(p1, p2, p3, p4, p5, p6, p7, "m2")


def main():
    from oslo import points_oslo

    test_qtm(points_oslo())


if __name__ == "__main__":
    main()
