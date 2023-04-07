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


# add_minmax legger til ny bin når precicion runder ned. Så runde opp?
# samme med ny min når runder opp. Så runde opp/ned?


def test_thematicmap(points_oslo):
    points = points_oslo

    points = points.clip(points.iloc[[0]].buffer(500))

    points.geometry = points.buffer(np.arange(1, len(points) + 1) * 10)

    def with_legend_and_title(points):
        m = sg.ThematicMap(points, points, points, "meters")
        m.add_continous_legend()
        m.title = "added legend and title"
        m.plot()
        assert m._k == 5
        assert m.bins == [62, 125, 188, 251, 314, 439], m.bins
        assert m.column == "length"
        assert m.legend.title == "length"
        assert m.colorlist == ["#fde5e2", "#fbafba", "#ed549d", "#a3017d", "#49006a"]
        assert m.cmap_start == 25
        assert m.legend._categories == [
            "62  - 125 ",
            "188 ",
            "251 ",
            "314 ",
            "376  - 439 ",
        ], m.legend._categories
        assert (m.facecolor, m.title_color, m.bg_gdf_color) == (
            "#fefefe",
            "#0f0f0f",
            "#d1d1cd",
        )

    with_legend_and_title(points)

    def manual_labels(points):
        m = sg.ThematicMap(points, points, points, "meters")
        m.add_continous_legend()
        m.k = 3
        m.legend.labels = ["something", "something more", "last color"]
        m.title = inspect.stack()[0][3]
        m.plot()
        assert m._k == 3
        assert m.bins == [62, 188, 314, 439], m.bins
        assert m.legend._categories == [
            "something",
            "something more",
            "last color",
        ], m.legend._categories

    manual_labels(points)

    def k_is_7_equal_to_n_unique(points):
        m = sg.ThematicMap(points, points, points, "meters", title="k=7 (all)")
        m.k = 7
        m.add_continous_legend()
        m.plot()
        assert m.title == "k=7 (all)"
        assert m.labels == ["points"] * 3
        assert m.cmap == "RdPu"
        assert m.colorlist == [
            "#fde5e2",
            "#fcc6c1",
            "#f994b1",
            "#ed549d",
            "#c01588",
            "#840178",
            "#49006a",
        ], m.colorlist
        assert (m.facecolor, m.title_color, m.bg_gdf_color) == (
            "#fefefe",
            "#0f0f0f",
            "#d1d1cd",
        )
        assert m._k == 7, m._k
        assert m.bins == [62, 125, 188, 251, 314, 376, 439], m.bins
        assert m.column == "length"
        assert m.legend.title == "length"
        assert m.legend._position_has_been_set is False
        assert m.legend.rounding == 0
        assert m.legend._categories == [
            "62 ",
            "125 ",
            "188 ",
            "251 ",
            "314 ",
            "376 ",
            "439 ",
        ], m.legend._categories

    k_is_7_equal_to_n_unique(points)

    def k_is_3(points):
        m = sg.ThematicMap(points, points, points, "meters")
        m.add_continous_legend()
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
        assert m.bins == [62, 188, 314, 439], m.bins
        assert m.legend._categories == [
            "62  - 188 ",
            "251  - 314 ",
            "376  - 439 ",
        ], m.legend._categories
        assert m.colorlist == ["#440154", "#21918c", "#fde725"], m.colorlist
        assert (m.facecolor, m.title_color, m.bg_gdf_color) == (
            "#0f0f0f",
            "#fefefe",
            "#383834",
        )

    k_is_3(points)

    def manual_bins_and_legend_suffix_sep(points):
        m = sg.ThematicMap(points, points, points, "meters")
        m.add_continous_legend()
        m.bins = (100, 200, 300)
        m.legend.label_sep = "to"
        m.legend.label_suffix = "m"
        m.plot()
        assert m.colorlist == ["#fde5e2", "#f994b1", "#c01588", "#49006a"], m.colorlist
        assert m.legend._categories == [
            "62 m",
            "125 m to 188 m",
            "251 m",
            "314 m to 439 m",
        ], m.legend._categories

    manual_bins_and_legend_suffix_sep(points)

    def fontsize(points):
        m = sg.ThematicMap(points, points, points, "meters")
        m.add_continous_legend()
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
        m.add_continous_legend()
        m.legend.rounding = 1
        m.title = inspect.stack()[0][3]
        m.plot()
        assert m.colorlist == [
            "#fde5e2",
            "#fbafba",
            "#ed549d",
            "#a3017d",
            "#49006a",
        ], m.colorlist
        assert m.legend._categories == [
            "62.8  - 125.6 ",
            "188.4 ",
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
        m.add_continous_legend()
        m.title = "Middle values missing"
        m.k = 3
        m.plot()
        assert m.colorlist == ["#fde5e2", "#ed549d", "#49006a", "#969696"], m.colorlist
        assert m.legend._categories == [
            "313  - 1254 ",
            "7841  - 11291 ",
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
        m.add_categorical_legend()
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
            "Missing": "#969696",
        }

    with_nans(points)

    def cmap_start_and_stop(points):
        m = sg.ThematicMap(points, points, points, "meters")
        m.add_continous_legend()
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
            m.add_continous_legend()
            m.k = k
            m.plot()

        p1 = points.iloc[:2].assign(large_number=100000000)
        p2 = points.iloc[2:4].assign(large_number=100000001)
        p3 = points.iloc[4:].assign(large_number=100000002)

        m = sg.ThematicMap(p1, p2, p3, "large_number")
        m.add_continous_legend()
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
        m.add_continous_legend()
        m.plot()
        assert len(m._unique_values) == 7, m._unique_values
        assert m.colorlist == [
            "#fde5e2",
            "#fbafba",
            "#ed549d",
            "#a3017d",
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
            m.add_continous_legend()
            m.k = k
            m.plot()

        buffered1 = points.iloc[:2].assign(small_number=0.000000)
        buffered2 = points.iloc[2:4].assign(small_number=0.00001)
        buffered3 = points.iloc[4:].assign(small_number=0.00002)
        m = sg.ThematicMap(buffered1, buffered2, buffered3, "small_number")
        m.add_continous_legend()
        m.title = "n_unique=3"
        m.plot()
        assert len(m._unique_values) == 3, m._unique_values
        assert m.legend.rounding == 6, m.legend.rounding
        assert m.colorlist == ["#fde5e2", "#ed549d", "#49006a"], m.colorlist
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
        m.add_continous_legend()
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
        m.add_continous_legend()
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
            m.add_continous_legend()
            m.k = k
            m.title = f"k={k}"
            m.plot()

        buffered1 = points.iloc[:2].assign(negative_number=-100_000)
        buffered2 = points.iloc[2:4].assign(negative_number=-100_001)
        buffered3 = points.iloc[4:].assign(negative_number=-100_002)
        m = sg.ThematicMap(buffered1, buffered2, buffered3, "negative_number")
        m.add_continous_legend()
        m.plot()
        assert len(m._unique_values) == 3, m._unique_values
        assert m.colorlist == ["#fde5e2", "#ed549d", "#49006a"], m.colorlist
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
        m.add_continous_legend()
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
        m.add_continous_legend()
        m.title = "k=5"
        m.plot()
        assert len(m._unique_values) == 7, m._unique_values
        assert m.legend._categories == [
            "-100006  - -100004 ",
            "-100003 ",
            "-100002 ",
            "-100001 ",
            "-100000 ",
        ], m.legend._categories

    negative_numbers(points)


def main():
    from oslo import points_oslo

    test_thematicmap(points_oslo())


if __name__ == "__main__":
    main()
