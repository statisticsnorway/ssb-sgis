# %%
import inspect
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


src = str(Path(__file__).parent).strip("tests") + "src"

import sys


sys.path.insert(0, src)

import sgis as sg


def test_thematicmap(points_oslo):
    points = points_oslo

    points = points.clip(points.iloc[[0]].buffer(500))

    points.geometry = points.buffer(np.arange(1, len(points) + 1) * 10)

    def legend_kwargs(points):
        m = sg.ThematicMap(points, points, points, "meters")
        m.title = inspect.stack()[0][3]
        m.legend.kwargs["labelcolor"] = "red"
        m.plot()

    legend_kwargs(points)

    def size_20(points):
        m = sg.ThematicMap(points, points, points, "meters")
        m.size = 20
        m.title = inspect.stack()[0][3]
        m.plot()
        assert m.title_fontsize == 40, m.title_fontsize
        assert m.legend.title_fontsize == 20 * 1.2
        assert m.legend.fontsize == 20
        assert m.legend.markersize == 20

    size_20(points)

    def check_colors_bins(points):
        print(points.length)
        m = sg.ThematicMap(points, points, points, "meters")
        m.title = "added title"
        m.plot()
        assert m._k == 5
        assert m.bins == [63, 63, 188, 251, 314, 440], m.bins
        assert m.column == "length"
        assert m.legend.title == "length"
        assert m.colorlist == [
            "#fddfdc",
            "#faa8b8",
            "#ea4d9c",
            "#a0017c",
            "#49006a",
        ], m.colorlist
        assert m.cmap_start == 33
        assert m.legend._categories == [
            "63 ",
            "126  - 188 ",
            "251 ",
            "314 ",
            "377  - 440 ",
        ], m.legend._categories
        assert (m.facecolor, m.title_color, m.bg_gdf_color) == (
            "#fefefe",
            "#0f0f0f",
            "#ebebeb",
        )

    check_colors_bins(points)

    def manual_labels_and_bins(points):
        labels_should_be = [
            "63 m to 100 m ",
            "100 m to 200 m ",
            "200 m to 300 m ",
            "300 m to 440 m ",
        ]

        m = sg.ThematicMap(points, points, points, "meters")
        m.bins = [(min(points.length)), 100, 200, 300, (max(points.length))]
        m.legend.labels = [
            f"{int(round(min(points.length),0))} m to 100 m ",
            "100 m to 200 m ",
            "200 m to 300 m ",
            f"300 m to {int(round(max(points.length),0))} m ",
        ]
        m.title = inspect.stack()[0][3]
        m.plot()
        assert m.bins == [63, 100, 200, 300, 440], m.bins
        assert m.legend._categories == labels_should_be, m.legend._categories

        m = sg.ThematicMap(points, points, points, "meters")
        m.bins = [100, 200, 300]
        m.legend.labels = [
            f"{int(round(min(points.length),0))} m to 100 m ",
            "100 m to 200 m ",
            "200 m to 300 m ",
            f"300 m to {int(round(max(points.length),0))} m ",
        ]
        m.title = inspect.stack()[0][3]
        m.plot()
        assert m.bins == [63, 100, 200, 300, 440], m.bins
        assert m.legend._categories == labels_should_be, m.legend._categories

    manual_labels_and_bins(points)

    def k_is_7_equal_to_n_unique(points):
        m = sg.ThematicMap(points, points, points, "meters")
        m.k = 7
        m.title = "k=7 (all)"
        m.plot()
        assert m.title == "k=7 (all)"
        assert m.labels == ["points"] * 3
        assert m.cmap == "RdPu"
        assert m.colorlist == [
            "#fddfdc",
            "#fcbebe",
            "#f98bae",
            "#ea4d9c",
            "#bd1186",
            "#820178",
            "#49006a",
        ], m.colorlist
        assert m._k == 7, m._k
        assert m.bins == [63, 126, 188, 251, 314, 377, 440], m.bins
        assert m.column == "length"
        assert m.legend.title == "length"
        assert m.legend._position_has_been_set is False
        assert m.legend.rounding == 0
        assert m.legend._categories == [
            "63 ",
            "126 ",
            "188 ",
            "251 ",
            "314 ",
            "377 ",
            "440 ",
        ], m.legend._categories

    k_is_7_equal_to_n_unique(points)

    def k_is_3(points):
        m = sg.ThematicMap(points, points, points, "meters")
        m.cmap = "viridis"
        m.black = True
        m.k = 3
        m.title = "black, viridis, k=3"
        m.plot()
        assert m.cmap == "viridis"
        assert m.cmap_start == 0
        assert m.column == "length"
        assert m.legend.title == "length"
        assert m._k == 3
        assert m.bins == [63, 188, 314, 440], m.bins
        assert m.legend._categories == [
            "63  - 188 ",
            "251  - 314 ",
            "377  - 440 ",
        ], m.legend._categories
        assert m.colorlist == ["#440154", "#21918c", "#fde725"], m.colorlist

    k_is_3(points)

    def manual_bins_and_legend_suffix_sep(points):
        m = sg.ThematicMap(points, points, points, "meters")
        m.bins = (100, 200, 300)
        m.legend.label_sep = "to"
        m.legend.label_suffix = "m"
        m.title = inspect.stack()[0][3]
        m.plot()
        assert m.colorlist == ["#fddfdc", "#f98bae", "#bd1186", "#49006a"], m.colorlist
        assert m.legend._categories == [
            "63 m",
            "126 m to 188 m",
            "251 m",
            "314 m to 440 m",
        ], m.legend._categories

    manual_bins_and_legend_suffix_sep(points)

    def fontsize(points):
        m = sg.ThematicMap(points, points, points, "meters")
        m.cmap = "plasma"
        m.title_fontsize = 20
        m.legend.fontsize = 30
        m.legend.title_fontsize = 10
        m.legend.title = "small legend_title_fontsize, large legend_fontsize"
        m.title = inspect.stack()[0][3]
        m.plot()

    fontsize(points)

    def rounding_1(points):
        m = sg.ThematicMap(points, points, points, "meters")
        m.legend.rounding = 1
        m.title = inspect.stack()[0][3]
        m.plot()
        assert m.colorlist == [
            "#fddfdc",
            "#faa8b8",
            "#ea4d9c",
            "#a0017c",
            "#49006a",
        ], m.colorlist
        assert m.legend._categories == [
            "62.8 ",
            "125.6  - 188.4 ",
            "251.2 ",
            "314.0 ",
            "376.8  - 439.6 ",
        ], m.legend._categories

    rounding_1(points)

    def with_nans(points):
        with_nan = points.assign(col_with_nan=lambda x: x.area)
        with_nan.loc[
            (with_nan.col_with_nan > 2000) & (with_nan.col_with_nan < 6000),
            "col_with_nan",
        ] = pd.NA

        m = sg.ThematicMap(with_nan, "col_with_nan")
        m.title = "Middle values missing"
        m.k = 3
        m.plot()
        assert m.colorlist == ["#fddfdc", "#ea4d9c", "#49006a", "#c2c2c2"], m.colorlist
        assert m.legend._categories == [
            "314  - 1255 ",
            "7841  - 11292 ",
            "15369 ",
            "Missing",
        ], m.legend._categories

        def _to_int(value):
            try:
                return int(value)
            except Exception:
                return pd.NA

        with_nan["col_with_nan_cat"] = with_nan["col_with_nan"].map(_to_int)

        m = sg.ThematicMap(with_nan, "col_with_nan_cat")
        m.title = "Middle values missing, categorical"
        m.plot()
        assert m.legend._categories == [
            313,
            1254,
            7841,
            11291,
            15369,
            "Missing",
        ], m.legend._categories
        assert m._categories_colors_dict == {
            313: "#4576ff",
            1254: "#ff455e",
            7841: "#59d45f",
            11291: "#b51d8b",
            15369: "#ffa514",
            "Missing": "#c2c2c2",
        }

    with_nans(points)

    def cmap_start_and_stop(points):
        m = sg.ThematicMap(points, points, points, "meters")
        m.cmap = "viridis"
        m.k = 4
        m.cmap_start = 100
        m.cmap_stop = 150
        m.title = "viridis, cmap_start=50, cmap_stop=150"
        m.plot()
        viridis_50_to_150 = ["#2a768e", "#25858e", "#1f958b", "#21a585"]
        assert m.colorlist == viridis_50_to_150, m.colorlist

    cmap_start_and_stop(points)

    def large_numbers(points):
        for k in np.arange(1, 8):
            print(k)
            m = sg.ThematicMap(
                points.iloc[[0]].assign(large_number=100000000.323),
                points.iloc[[1]].assign(large_number=100002000.0323),
                points.iloc[[2]].assign(large_number=100004000.3244),
                points.iloc[[3]].assign(large_number=100006000.3245),
                points.iloc[[4]].assign(large_number=100008000.4321),
                points.iloc[[5]].assign(large_number=100010000.3232),
                points.iloc[[6]].assign(large_number=100012000.2323),
                "large_number",
            )
            m.k = k
            m.plot()

        p1 = points.iloc[:2].assign(large_number=100000000)
        p2 = points.iloc[2:4].assign(large_number=100000001)
        p3 = points.iloc[4:].assign(large_number=100000002)

        m = sg.ThematicMap(p1, p2, p3, "large_number")
        m.title = "n_unique=3"
        m.plot()
        assert len(m._unique_values) == 3, m._unique_values
        assert m.legend._categories == [
            "100000000 ",
            "100000001 ",
            "100000002 ",
        ], m.legend._categories
        m = sg.ThematicMap(
            points.iloc[[0]].assign(large_number=100000000.323),
            points.iloc[[1]].assign(large_number=100002000.0323),
            points.iloc[[2]].assign(large_number=100004000.3244),
            points.iloc[[3]].assign(large_number=100006000.3245),
            points.iloc[[4]].assign(large_number=100008000.4321),
            points.iloc[[5]].assign(large_number=100010000.3232),
            points.iloc[[6]].assign(large_number=100012000.2323),
            "large_number",
        )
        m.plot()
        assert len(m._unique_values) == 7, m._unique_values
        assert m.colorlist == [
            "#fddfdc",
            "#faa8b8",
            "#ea4d9c",
            "#a0017c",
            "#49006a",
        ], m.colorlist
        assert m.legend._categories == [
            "100000000  - 100002000 ",
            "100004000 ",
            "100006000 ",
            "100008000  - 100010000 ",
            "100012000 ",
        ], m.legend._categories

    large_numbers(points)

    def small_numbers(points):
        for k in np.arange(1, 8):
            print(k)
            m = sg.ThematicMap(
                points.iloc[[0]].assign(small_number=0.00000),
                points.iloc[[1]].assign(small_number=0.00001),
                points.iloc[[2]].assign(small_number=0.00002),
                points.iloc[[3]].assign(small_number=0.00003),
                points.iloc[[4]].assign(small_number=0.00004),
                points.iloc[[5]].assign(small_number=0.00005),
                points.iloc[[6]].assign(small_number=0.00006),
                "small_number",
            )
            m.k = k
            m.plot()

        buffered1 = points.iloc[:2].assign(small_number=0.000000)
        buffered2 = points.iloc[2:4].assign(small_number=0.00001)
        buffered3 = points.iloc[4:].assign(small_number=0.00002)
        m = sg.ThematicMap(buffered1, buffered2, buffered3, "small_number")
        m.title = "n_unique=3"
        m.plot()
        assert len(m._unique_values) == 3, m._unique_values
        assert m.legend.rounding == 6, m.legend.rounding
        assert m.colorlist == ["#fddfdc", "#ea4d9c", "#49006a"], m.colorlist
        assert m.legend._categories == [
            "0.0 ",
            "1e-05 ",
            "2e-05 ",
        ], m.legend._categories

        m = sg.ThematicMap(
            points.iloc[[0]].assign(small_number=0.00000),
            points.iloc[[1]].assign(small_number=0.00001),
            points.iloc[[2]].assign(small_number=0.00002),
            points.iloc[[3]].assign(small_number=0.00003),
            points.iloc[[4]].assign(small_number=0.00004),
            points.iloc[[5]].assign(small_number=0.00005),
            points.iloc[[6]].assign(small_number=0.00006),
            "small_number",
        )
        m.k = 7
        m.title = "k=7"
        m.plot()
        assert m.legend.rounding == 5, m.legend.rounding
        assert len(m._unique_values) == 7, m._unique_values
        assert m.legend._categories == [
            "0.0 ",
            "1e-05 ",
            "2e-05 ",
            "3e-05 ",
            "4e-05 ",
            "5e-05 ",
            "6e-05 ",
        ], m.legend._categories

        m = sg.ThematicMap(
            points.iloc[[0]].assign(quite_small_number=0.01),
            points.iloc[[1]].assign(quite_small_number=0.02),
            points.iloc[[2]].assign(quite_small_number=0.4),
            points.iloc[[3]].assign(quite_small_number=1.03),
            points.iloc[[4]].assign(quite_small_number=1.4),
            points.iloc[[5]].assign(quite_small_number=1.8),
            points.iloc[[6]].assign(quite_small_number=2.6),
            "quite_small_number",
        )
        m.title = "k=5"
        m.plot()
        assert m.legend.rounding == 2, m.legend.rounding
        assert len(m._unique_values) == 6, m._unique_values
        assert m.legend._categories == [
            "0.01 ",
            "0.02  - 0.4 ",
            "1.03  - 1.4 ",
            "1.8 ",
            "2.6 ",
        ], m.legend._categories

    small_numbers(points)

    def negative_numbers(points):
        for k in np.arange(1, 8):
            print(k)
            m = sg.ThematicMap(
                points.iloc[[0]].assign(negative_number=-100_000),
                points.iloc[[1]].assign(negative_number=-100_001),
                points.iloc[[2]].assign(negative_number=-100_002),
                points.iloc[[3]].assign(negative_number=-100_003),
                points.iloc[[4]].assign(negative_number=-100_004),
                points.iloc[[5]].assign(negative_number=-100_005),
                points.iloc[[6]].assign(negative_number=-100_006),
                "negative_number",
            )
            m.k = k
            m.title = f"k={k}"
            m.plot()

        buffered1 = points.iloc[:2].assign(negative_number=-100_000)
        buffered2 = points.iloc[2:4].assign(negative_number=-100_001)
        buffered3 = points.iloc[4:].assign(negative_number=-100_002)
        m = sg.ThematicMap(buffered1, buffered2, buffered3, "negative_number")
        m.plot()
        assert len(m._unique_values) == 3, m._unique_values
        assert m.colorlist == ["#fddfdc", "#ea4d9c", "#49006a"], m.colorlist
        assert m.legend._categories == [
            "-100002 ",
            "-100001 ",
            "-100000 ",
        ], m.legend._categories
        m = sg.ThematicMap(
            points.iloc[[0]].assign(negative_number=-100_000),
            points.iloc[[1]].assign(negative_number=-100_001),
            points.iloc[[2]].assign(negative_number=-100_002),
            points.iloc[[3]].assign(negative_number=-100_003),
            points.iloc[[4]].assign(negative_number=-100_004),
            points.iloc[[5]].assign(negative_number=-100_005),
            points.iloc[[6]].assign(negative_number=-100_006),
            "negative_number",
        )
        m.k = 7
        m.title = "k=7"
        m.plot()
        assert len(m._unique_values) == 7, m._unique_values
        assert m.legend._categories == [
            "-100006 ",
            "-100005 ",
            "-100004 ",
            "-100003 ",
            "-100002 ",
            "-100001 ",
            "-100000 ",
        ], m.legend._categories

        m = sg.ThematicMap(
            points.iloc[[0]].assign(negative_number=-100_000),
            points.iloc[[1]].assign(negative_number=-100_001),
            points.iloc[[2]].assign(negative_number=-100_002),
            points.iloc[[3]].assign(negative_number=-100_003),
            points.iloc[[4]].assign(negative_number=-100_004),
            points.iloc[[5]].assign(negative_number=-100_005),
            points.iloc[[6]].assign(negative_number=-100_006),
            "negative_number",
        )
        m.title = "k=5"
        m.plot()
        assert len(m._unique_values) == 7, m._unique_values
        assert m.legend._categories == [
            "-100006  - -100005 ",
            "-100004 ",
            "-100003 ",
            "-100002  - -100001 ",
            "-100000 ",
        ], m.legend._categories

    negative_numbers(points)


def main():
    from oslo import points_oslo

    test_thematicmap(points_oslo())


if __name__ == "__main__":
    main()
