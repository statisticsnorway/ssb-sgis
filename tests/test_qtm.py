# %%
from pathlib import Path

import geopandas as gpd
import numpy as np


src = str(Path(__file__).parent).strip("tests") + "src"

import sys


sys.path.insert(0, src)

import sgis as sg


def not_test_qtm(points_oslo):
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

    points = sg.read_parquet_url(
        "https://media.githubusercontent.com/media/statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet"
    )

    points100 = sg.buff(points.sample(5), 100).assign(area_m=lambda df: df.area)
    points300 = sg.buff(points.sample(5), 300).assign(area_m=lambda df: df.area)
    points500 = sg.buff(points.sample(5), 500).assign(area_m=lambda df: df.area)
    points500_2 = sg.buff(points.sample(5), 500).assign(area_m=lambda df: df.area)
    points500_3 = sg.buff(points.sample(5), 500).assign(area_m=lambda df: df.area)

    sg.qtm(
        points500,
        points300,
        points100,
        column="area_m",
        title="should be three colors with five circles each",
        legend_title="square meters",
    )

    sg.qtm(
        points500.assign(area_cat=lambda df: df.area.astype(int).astype(str)),
        points300.assign(area_cat=lambda df: df.area.astype(int).astype(str)),
        points100.assign(area_cat=lambda df: df.area.astype(int).astype(str)),
        column="area_cat",
        title="should be three colors with five circles each",
        legend_title="square meters",
    )

    sg.qtm(
        points500_3,
        points500_2,
        points500,
        points300,
        points100,
        column="area_m",
        title="should be three colors, 15 yellow, five green, five purple",
        legend_title="square meters",
    )

    if __name__ == "__main__":
        sg.explore(
            points500_3,
            points500_2,
            points500,
            points300,
            points100,
            column="area_m",
        )

    points750 = sg.buff(points.sample(5), 750).assign(area_m=lambda df: df.area)
    points1000 = sg.buff(points.sample(5), 1000).assign(area_m=lambda df: df.area)
    sg.qtm(
        points1000,
        points1000,
        points1000,
        points1000,
        points1000,
        points750,
        points750,
        points750,
        points750,
        points500,
        points500,
        points500,
        points300,
        points100,
        column="area_m",
        title=(
            "Should be five colors, five unique points \neach "
            "But some colors have multiple \noverlapping points"
        ),
        legend_title="square meters",
    )


def main():
    from oslo import points_oslo

    not_test_qtm(points_oslo())


if __name__ == "__main__":
    main()
