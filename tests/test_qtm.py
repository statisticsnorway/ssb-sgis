# %%
from pathlib import Path

import geopandas as gpd
import numpy as np


src = str(Path(__file__).parent).strip("tests") + "src"

import sys


sys.path.insert(0, src)

import sgis as sg


def not_test_thematicmap(points_oslo):
    points = points_oslo

    points = points.clip(points.iloc[[0]].buffer(500))

    p = ()
    for buffdist in np.arange(1, 40)[::-1]:
        p = p + (sg.buff(points.iloc[[0]], buffdist),)
    sg.qtm(*p, column="area", k=3)
    sg.qtm(*p, column="area", k=5)
    sg.qtm(*p, column="area", k=9)
    sg.qtm(*p, column="area", bins=(1000, 1500, 2500, 3000, 4000))
    sg.qtm(*p, column="area", bins=(0, 1000, 1500, 2500, 3000, 4000, 6000))

    points.geometry = points.buffer(np.arange(1, len(points) + 1) * 10)
    points["m2"] = points.area

    m = sg.ThematicMap(points, points, points, "meters")
    m.plot()

    m = sg.ThematicMap(points, points, points, "area")
    m.add_continous_legend()
    m.plot()

    m = sg.ThematicMap(points, points, points, "area")
    m.add_continous_legend(bin_precicion=0.001)
    m.change_legend_title("bin_precicion=0.001")
    m.plot()

    m = sg.ThematicMap(points, points, points, "area")
    m.add_continous_legend(bin_precicion=1000)
    m.plot()

    m.change_legend_title("New title")
    m.plot()

    m.change_legend_size(fontsize=5, title_fontsize=30, markersize=20)
    m.plot()

    m = sg.ThematicMap(points, points, points, "m2")
    m.add_continous_legend(bin_precicion=1000)
    m.add_title("bin_precicion=1000")
    m.plot()

    m = sg.ThematicMap(points, points, points, "m2", cmap="plasma")
    m.add_background(sg.buff(points_oslo, 100))
    m.add_continous_legend(
        title="Legend title",
        label_suffix="m2",
        label_sep="to",
        markersize=20,
        fontsize=20,
    )
    m.move_legend(1, 0.4)
    m.add_title("Title", size=30)
    m.plot()

    sg.qtm(points, points, points, "m2", black=True)
    sg.qtm(points, points, points, "m2", black=False)


def not_test_qtm(points_oslo):
    points = points_oslo

    points = points.clip(points.iloc[[0]].buffer(500))

    points.geometry = points.buffer(np.arange(1, len(points) + 1) * 10)
    points["m2"] = points.area
    sg.qtm(points, points, points, "m2")

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
    sg.qtm(p1, p2, p3, p4, p5, p6, p7, "m2", title="Should be five colors, gradient")

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
    )

    sg.qtm(
        points500_3,
        points500_2,
        points500,
        points300,
        points100,
        column="area_m",
        title="should be three colors, 15 yellow, five green, five purple",
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
    )


def main():
    from oslo import points_oslo

    not_test_thematicmap(points_oslo())
    not_test_qtm(points_oslo())


if __name__ == "__main__":
    main()
