# %%
import inspect
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

src = str(Path(__file__).parent).replace("tests", "") + "src"
testdata = str(Path(__file__).parent.parent) + "/tests/testdata"

import sys

sys.path.insert(0, src)

import sgis as sg

# set to True to not actually create the plots
# because pytest breaks with all these plots on github
__test = 1


def test_hatches():
    gdf = sg.to_gdf([(0.5, 0.5), (0, 0), (1, 1), (1, 0), (0, 1), (0.5, 0.5)]).assign(
        geometry=lambda x: x.geometry.buffer([1.4, 1, 0.8, 0.6, 0.4, 0.2]),
        value=pd.Series([pd.NA, 1, 2, 3, 4, 5]).astype("UInt8"),
    )

    m = sg.ThematicMap(
        dataframe=gdf.iloc[3:].assign(
            value=lambda df: [str(x) if pd.notna(x) else pd.NA for x in df["value"]]
        ),
        column="value",
        title="hatch on extra data",
    )
    m.add_data(no_data=gdf.iloc[3:].buffer(0.2), hatch="/")
    m.plot()
    assert list(m._more_data) == ["no_data"], list(m._more_data)
    assert (x := m._categories_colors_dict) == {
        "3": "#3b93ff",
        "4": "#ff3370",
        "5": "#f7cf19",
    }, x
    assert len(m.legend._patches) == 4

    m = sg.ThematicMap(
        dataframe=gdf,
        column="value",
        nan_label="No data",
        nan_hatch=r"/",
        title="hatch on nan",
    )
    m.plot()
    assert list(m._more_data) == ["No data"]
    assert (x := list(m._unique_colors)) == [
        "#fee6e3",
        "#fbb0ba",
        "#ee559d",
        "#a5017d",
        "#49006a",
    ], x
    assert len(m.legend._patches) == 6

    m = sg.ThematicMap(
        dataframe=gdf,
        column="value",
        nan_label="No Data",
        title="hatch on buffered background, but gray nan",
    )
    m.add_background(gdf.buffer(0.2), hatch="/")
    m.plot()
    assert [x.strip() for x in m.legend._categories] == [
        "1",
        "2",
        "3",
        "4",
        "5",
        "No Data",
    ], m.legend._categories
    assert (x := list(m._unique_colors)) == [
        "#fee6e3",
        "#fbb0ba",
        "#ee559d",
        "#a5017d",
        "#49006a",
    ], x
    assert len(m.legend._patches) == 6
    assert [x.strip() for x in m.legend._categories] == [
        "1",
        "2",
        "3",
        "4",
        "5",
        "No Data",
    ], m.legend._categories

    m = sg.ThematicMap(
        dataframe=gdf,
        column="value",
        title="Numeric, hatch nan",
        alpha=1,
        nan_hatch=r"\"",
        nan_label="No data",
        legend_kwargs=dict(
            title="Sirkel numeric",
        ),
    )
    m.plot()

    m = sg.ThematicMap(
        dataframe=gdf.assign(
            value=lambda df: [str(x) if pd.notna(x) else pd.NA for x in df["value"]]
        ),
        column="value",
        alpha=1,
        nan_color="Green",
        nan_label="No data",
        title="Green 'No data'",
        legend_kwargs=dict(
            title="Sirkel categoric",
        ),
    )
    m.plot()
    assert len(m._categories_colors_dict) == 5, m._categories_colors_dict
    assert len(m._more_data) == 1, m._more_data
    assert m.nan_color == "Green", m.nan_color

    m = sg.ThematicMap(
        dataframe=gdf.assign(
            value=lambda df: [str(x) if pd.notna(x) else pd.NA for x in df["value"]]
        ),
        column="value",
        alpha=1,
        nan_hatch=r"\"",
        nan_label="No data",
        title="hatch nan on 'No data'",
        legend_kwargs=dict(
            title="Sirkel categoric",
        ),
    )
    m.plot()
    assert len(m._categories_colors_dict) == 5, m._categories_colors_dict
    assert len(m._more_data) == 1, m._more_data


# @pytest.mark.skip(
#     reason="This test takes forever on torchgeo bbox, need to investigate"
# )
def test_thematicmap2():
    municipalities = gpd.read_parquet(f"{testdata}/municipalities_2017.parquet")
    m = sg.ThematicMap(
        municipalities,
        column="area_km2",
        title="Municipalities of Norway \nby Area",
        legend_kwargs=dict(
            rounding=-2,
            thousand_sep=" ",
            position=(0.82, 0.25),
        ),
        cmap="Greens",
        cmap_start=45,
        title_position=(0.035, 0.9),
    )
    m.plot()
    assert m.cmap == "Greens"
    assert m.cmap_start == 45
    assert m.title_kwargs == {"loc": "left", "x": 0.035, "y": 0.9}
    assert list(m.bins) == [0, 600, 1400, 2700, 5400, 9700], list(m.bins)
    assert m.legend._categories == [
        "2  - 599 ",
        "600  - 1 399 ",
        "1 400  - 2 699 ",
        "2 700  - 5 399 ",
        "5 400  - 9 730 ",
    ], m.legend._categories
    bins_and_category_lengths = {k: len(v) for k, v in m._bins_unique_values.items()}
    assert bins_and_category_lengths == {
        0: 248,
        1: 121,
        2: 43,
        3: 12,
        4: 2,
    }, bins_and_category_lengths
    assert m.legend.title == "Area Km2", m.legend.title


@pytest.mark.skip(reason="This test requires GUI")
def test_thematicmap(points_oslo):
    points = points_oslo

    points = points.clip(points.iloc[[0]].buffer(500))

    points.geometry = points.buffer(np.arange(1, len(points) + 1) * 10)

    def incorrect_attributes(points):
        m = sg.ThematicMap(pd.concat([points, points, points]), column="meters")

        with pytest.raises(AttributeError):
            m.legend_title = "this should not work"

        with pytest.raises(AttributeError):
            m.legend_title_fontsize = "this should not work"

        m.legend.title_fontsize = "this should work"

    incorrect_attributes(points)

    def pretty_labels(points):
        m = sg.ThematicMap(pd.concat([points, points, points]), column="meters")
        m.title = inspect.stack()[0][3]
        m.legend.title = "not pretty_labels, no bins"
        m.legend.pretty_labels = False
        m.plot(__test=__test)
        assert m.legend._categories == [
            "63 ",
            "126  - 188 ",
            "251 ",
            "314 ",
            "377  - 440 ",
        ], m.legend._categories
        m = sg.ThematicMap(pd.concat([points, points, points]), column="meters")
        m.title = inspect.stack()[0][3]
        m.legend.title = "pretty_labels, no bins"
        m.legend.pretty_labels = True
        m.plot(__test=__test)
        assert m.legend._categories == [
            "63  - 63 ",
            "64  - 188 ",
            "189  - 251 ",
            "252  - 314 ",
            "315  - 440 ",
        ], m.legend._categories

        # column with the exact bin values
        points["col"] = [63, 100, 150, 200, 250, 300, 440]

        m = sg.ThematicMap(pd.concat([points, points, points]), "col")
        m.bins = (99.99, 199.999, 299.99)
        m.title = "not pretty_labels, bins: " + ", ".join(
            [str(bin_) for bin_ in m.bins]
        )
        m.legend.title = "not pretty_labels, bins"
        m.legend.pretty_labels = False
        m.plot(__test=__test)
        assert m.legend._categories == [
            "63 ",
            "100  - 150 ",
            "200  - 250 ",
            "300  - 440 ",
        ], m.legend._categories

        m = sg.ThematicMap(pd.concat([points, points, points]), "col")
        m.bins = (100, 200, 300)
        m.title = "not pretty_labels, bins: " + ", ".join(
            [str(bin_) for bin_ in m.bins]
        )
        m.legend.title = "not pretty_labels, bins"
        m.legend.pretty_labels = False
        m.plot(__test=__test)
        print(m.bins)
        assert m.legend._categories == [
            "63  - 100 ",
            "150  - 200 ",
            "250  - 300 ",
            "440 ",
        ], m.legend._categories

        m = sg.ThematicMap(pd.concat([points, points, points]), "col")
        m.bins = (99.99, 199.999, 299.99)
        m.title = "pretty_labels, bins: " + ", ".join([str(bin_) for bin_ in m.bins])
        m.legend.title = "pretty_labels, bins"
        m.legend.pretty_labels = True
        m.legend.label_sep = "to"
        m.plot(__test=__test)
        assert m.legend._categories == [
            "63  to 100 ",
            "101  to 200 ",
            "201  to 300 ",
            "301  to 440 ",
        ], m.legend._categories

        m = sg.ThematicMap(pd.concat([points, points, points]), "col")
        m.bins = (100, 200, 300)
        m.title = "pretty_labels, bins: " + ", ".join([str(bin_) for bin_ in m.bins])
        m.legend.title = "pretty_labels, bins"
        m.legend.pretty_labels = True
        m.plot(__test=__test)
        assert m.legend._categories == [
            "63  - 100 ",
            "101  - 200 ",
            "201  - 300 ",
            "301  - 440 ",
        ], m.legend._categories

    # pretty_labels(points)

    def thousand_sep_decimal_mark(points):
        m = sg.ThematicMap(points, column="meters")
        for thousand_sep in [None, ".", ",", " "]:
            for decimal_mark in [None, ",", "."]:
                number = 10000000.1234
                m.thousand_sep = thousand_sep
                m.decimal_mark = decimal_mark
                print(thousand_sep, decimal_mark, number)

    thousand_sep_decimal_mark(points)

    def size_10(points):
        m = sg.ThematicMap(pd.concat([points, points, points]), column="meters")
        m.size = 10
        m.title = inspect.stack()[0][3]
        m.plot()
        assert m.title_fontsize == 20, m.title_fontsize
        assert m.legend.title_fontsize == 10 * 1.2
        assert m.legend.fontsize == 10
        assert m.legend.markersize == 10

    size_10(points)

    def check_colors_bins(points):
        m = sg.ThematicMap(
            pd.concat([points, points, points]), title="added title", column="meters"
        )
        m.plot()
        assert m._k == 5
        assert m.column == "length"
        assert m.legend.title == "Length"
        assert list(m._unique_colors) == [
            "#fee6e3",
            "#fbb0ba",
            "#ee559d",
            "#a5017d",
            "#49006a",
        ], m._unique_colors
        assert m.cmap_start == 23
        assert (m.facecolor, m.title_color, m.bg_gdf_color) == (
            "#fefefe",
            "#0f0f0f",
            "#e8e6e6",
        ), (m.facecolor, m.title_color, m.bg_gdf_color)

    check_colors_bins(points)

    def manual_labels_and_bins(points):
        labels_should_be = [
            "63 m to 100 m ",
            "100 m to 200 m ",
            "200 m to 300 m ",
            "300 m to 440 m ",
        ]

        m = sg.ThematicMap(pd.concat([points, points, points]), column="meters")
        assert not m._is_categorical
        m.bins = [(min(points.length)), 100, 200, 300, (max(points.length))]
        m.legend.labels = [
            f"{int(round(min(points.length),0))} m to 100 m ",
            "100 m to 200 m ",
            "200 m to 300 m ",
            f"300 m to {int(round(max(points.length),0))} m ",
        ]
        m.title = inspect.stack()[0][3]
        m.plot()
        assert [int(round(x, 0)) for x in m.bins] == [63, 100, 200, 300, 440], m.bins
        assert m.legend._categories == labels_should_be, m.legend._categories

        m = sg.ThematicMap(pd.concat([points, points, points]), column="meters")
        m.bins = [100, 200, 300]
        m.legend.labels = [
            f"{int(round(min(points.length),0))} m to 100 m ",
            "100 m to 200 m ",
            "200 m to 300 m ",
            f"300 m to {int(round(max(points.length),0))} m ",
        ]
        m.title = inspect.stack()[0][3]
        m.plot()
        assert [int(round(x, 0)) for x in m.bins] == [63, 100, 200, 300, 440], m.bins
        assert m.legend._categories == labels_should_be, m.legend._categories

    manual_labels_and_bins(points)

    def k_is_7_equal_to_n_unique(points):
        m = sg.ThematicMap(pd.concat([points, points, points]), column="meters")
        m.k = 7
        m.title = "k=7 (all)"
        m.plot()
        assert m.title == "k=7 (all)"
        # assert list(m._gdfs) == ["points"] * 3, m._gdfs
        assert m.cmap == "RdPu"
        assert list(m._unique_colors) == [
            "#fee6e3",
            "#fcc7c3",
            "#fa97b2",
            "#ee559d",
            "#c21688",
            "#840178",
            "#49006a",
        ], m._unique_colors
        assert m._k == 7, m._k
        assert m.column == "length"
        assert m.legend.title == "Length", m.legend.title
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
        m = sg.ThematicMap(
            pd.concat([points, points, points]),
            column="meters",
            cmap="viridis",
            dark=True,
            k=3,
            title="black, viridis, k=3",
        )
        m.plot()
        assert m.cmap == "viridis"
        assert m.cmap_start == 0, m.cmap_start
        assert m.column == "length"
        assert m.legend.title == "Length"
        assert m._k == 3
        assert list(m._unique_colors) == [
            "#440154",
            "#21918c",
            "#fde725",
        ], m._unique_colors

    k_is_3(points)

    def manual_bins_and_legend_suffix_sep(points):
        m = sg.ThematicMap(pd.concat([points, points, points]), column="meters")
        m.bins = (100, 200, 300)
        m.legend.label_sep = "to"
        m.legend.label_suffix = "m"
        m.title = inspect.stack()[0][3]
        m.plot()
        assert list(m._unique_colors) == [
            "#fee6e3",
            "#fa97b2",
            "#c21688",
            "#49006a",
        ], m._unique_colors

    manual_bins_and_legend_suffix_sep(points)

    def rounding_1(points):
        m = sg.ThematicMap(
            pd.concat([points, points, points]),
            column="meters",
            legend_kwargs=dict(rounding=1),
        )
        m.title = inspect.stack()[0][3]
        m.plot()
        assert list(m._unique_colors) == [
            "#fee6e3",
            "#fbb0ba",
            "#ee559d",
            "#a5017d",
            "#49006a",
        ], m._unique_colors

    # rounding_1(points)

    def with_nans(points):
        with_nan = points.assign(col_with_nan=lambda x: x.area)
        with_nan.loc[
            (with_nan.col_with_nan > 2000) & (with_nan.col_with_nan < 6000),
            "col_with_nan",
        ] = pd.NA

        m = sg.ThematicMap(with_nan, column="col_with_nan")
        m.title = "Middle values missing"
        m.k = 3
        m.plot()
        assert list(m._unique_colors) == [
            "#fee6e3",
            "#ee559d",
            "#49006a",
        ], m._unique_colors
        assert list(m._more_data) == ["Missing"]

        def _to_int(value):
            try:
                return int(value)
            except Exception:
                return "nan"

        with_nan["col_with_nan_cat"] = with_nan["col_with_nan"].map(_to_int).astype(str)
        with_nan.loc[with_nan.col_with_nan_cat == "nan", "col_with_nan_cat"] = pd.NA

        m = sg.ThematicMap(with_nan, column="col_with_nan_cat")
        m.title = "Middle values missing, categorical"
        m.plot()

    with_nans(points)

    def cmap_start_and_stop(points):
        m = sg.ThematicMap(pd.concat([points, points, points]), column="meters")
        m.cmap = "viridis"
        m.k = 4
        m.cmap_start = 100
        m.cmap_stop = 150
        m.title = "viridis, cmap_start=50, cmap_stop=150"
        m.plot()
        viridis_50_to_150 = ["#2a768e", "#25858e", "#1f958b", "#21a585"]
        assert list(m._unique_colors) == viridis_50_to_150, m._unique_colors

    cmap_start_and_stop(points)

    def large_numbers(points):
        for k in [2, 4, 6, 7]:
            print(k)
            m = sg.ThematicMap(
                points.iloc[[0]].assign(large_number=100000000.323),
                points.iloc[[1]].assign(large_number=100002000.0000323),
                points.iloc[[2]].assign(large_number=100004000.3244),
                points.iloc[[3]].assign(large_number=100006000.3245),
                points.iloc[[4]].assign(large_number=100008000.4321),
                points.iloc[[5]].assign(large_number=100010000.3232),
                points.iloc[[6]].assign(large_number=100012000.2323),
                column="large_number",
            )
            m.k = k
            m.plot()

        p1 = points.iloc[:2].assign(large_number=100000000)
        p2 = points.iloc[2:4].assign(large_number=100000001)
        p3 = points.iloc[4:].assign(large_number=100000002)

        m = sg.ThematicMap(p1, p2, p3, column="large_number")
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
            points.iloc[[1]].assign(large_number=100002000.0000323),
            points.iloc[[2]].assign(large_number=100004000.3244),
            points.iloc[[3]].assign(large_number=100006000.3245),
            points.iloc[[4]].assign(large_number=100008000.4321),
            points.iloc[[5]].assign(large_number=100010000.3232),
            points.iloc[[6]].assign(large_number=100012000.2323),
            column="large_number",
        )
        m.plot()
        assert len(m._unique_values) == 7, m._unique_values
        assert list(m._unique_colors) == [
            "#fee6e3",
            "#fbb0ba",
            "#ee559d",
            "#a5017d",
            "#49006a",
        ], m._unique_colors

    # large_numbers(points)

    def small_numbers(points):
        for k in [2, 4, 6]:
            print(k)
            m = sg.ThematicMap(
                points.iloc[[0]].assign(small_number=0.00000),
                points.iloc[[1]].assign(small_number=0.00001),
                points.iloc[[2]].assign(small_number=0.00002),
                points.iloc[[3]].assign(small_number=0.00003),
                points.iloc[[4]].assign(small_number=0.00004),
                points.iloc[[5]].assign(small_number=0.00005),
                points.iloc[[6]].assign(small_number=0.00006),
                column="small_number",
            )
            m.k = k
            m.plot()

        buffered1 = points.iloc[:2].assign(small_number=0.000000)
        buffered2 = points.iloc[2:4].assign(small_number=0.00001)
        buffered3 = points.iloc[4:].assign(small_number=0.00002)
        m = sg.ThematicMap(buffered1, buffered2, buffered3, column="small_number")
        m.title = "n_unique=3"
        m.plot()
        assert len(m._unique_values) == 3, m._unique_values
        assert m.legend.rounding == 6, m.legend.rounding
        assert list(m._unique_colors) == [
            "#fee6e3",
            "#ee559d",
            "#49006a",
        ], m._unique_colors
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
            column="small_number",
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
            points.iloc[[0]].assign(quite_small_number=0.00001),
            points.iloc[[1]].assign(quite_small_number=0.00002),
            points.iloc[[2]].assign(quite_small_number=0.4),
            points.iloc[[3]].assign(quite_small_number=1.03),
            points.iloc[[4]].assign(quite_small_number=1.4),
            points.iloc[[5]].assign(quite_small_number=1.8),
            points.iloc[[6]].assign(quite_small_number=2.6),
            column="quite_small_number",
        )
        m.title = "k=5"
        m.plot()
        print(m._unique_values)
        assert m.legend.rounding == 6, m.legend.rounding
        assert len(m._unique_values) == 7, m._unique_values

    small_numbers(points)

    def negative_numbers(points):
        for k in [2, 4, 6, 7]:
            print(k)
            m = sg.ThematicMap(
                points.iloc[[0]].assign(negative_number=-100_000),
                points.iloc[[1]].assign(negative_number=-100_001),
                points.iloc[[2]].assign(negative_number=-100_002),
                points.iloc[[3]].assign(negative_number=-100_003),
                points.iloc[[4]].assign(negative_number=-100_004),
                points.iloc[[5]].assign(negative_number=-100_005),
                points.iloc[[6]].assign(negative_number=-100_006),
                column="negative_number",
            )
            m.k = k
            m.title = f"k={k}"
            m.plot()

        buffered1 = points.iloc[:2].assign(negative_number=-100_000)
        buffered2 = points.iloc[2:4].assign(negative_number=-100_001)
        buffered3 = points.iloc[4:].assign(negative_number=-100_002)
        m = sg.ThematicMap(buffered1, buffered2, buffered3, column="negative_number")
        m.plot()
        assert len(m._unique_values) == 3, m._unique_values
        assert list(m._unique_colors) == [
            "#fee6e3",
            "#ee559d",
            "#49006a",
        ], m._unique_colors
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
            column="negative_number",
        )
        m.title = "large, negative numbers"
        m.plot()
        assert len(m._unique_values) == 7, m._unique_values

        m = sg.ThematicMap(
            points.iloc[[0]].assign(negative_number=-0.02001),
            points.iloc[[1]].assign(negative_number=-0.0203),
            points.iloc[[2]].assign(negative_number=-0.20003),
            points.iloc[[3]].assign(negative_number=-0.02203),
            points.iloc[[4]].assign(negative_number=-0.00232),
            points.iloc[[5]].assign(negative_number=-0.0223),
            points.iloc[[6]].assign(negative_number=-0.3232),
            column="negative_number",
        )
        m.title = "small, negative numbers"
        m.plot()
        assert len(m._unique_values) == 7, m._unique_values

    negative_numbers(points)

    def scheme_is_none(points):
        m = sg.ThematicMap(pd.concat([points, points, points]), column="meters")
        m.title = inspect.stack()[0][3]
        m.scheme = None
        m.plot()

    scheme_is_none(points)

    def assert_manually(points):
        m = sg.ThematicMap(pd.concat([points, points, points]), column="meters")
        m.cmap = "plasma"
        m.title_fontsize = 20
        m.legend.fontsize = 30
        m.legend.title_fontsize = 10
        m.title = "small legend_title_fontsize, large legend_fontsize, red color"
        m.legend.kwargs["labelcolor"] = "red"
        m.plot()

    assert_manually(points)

    def categorical_column(points):
        points["category"] = [*"abcddea"]
        m = sg.ThematicMap(
            points, column="category", legend_kwargs=dict(pretty_labels=False)
        )
        m.plot()
        assert m.legend._categories == ["a", "b", "c", "d", "e"], m.legend._categories

    categorical_column(points)


def main():
    from oslo import points_oslo

    test_thematicmap2()
    test_thematicmap(points_oslo())
    test_hatches()


if __name__ == "__main__":
    main()
