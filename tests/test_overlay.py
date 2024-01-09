# %%
import sys
import warnings
from pathlib import Path

import numpy as np
from helpers import cprofile_df
from IPython.display import display


src = str(Path(__file__).parent).strip("tests") + "src"

sys.path.insert(0, src)

import sgis as sg


def test_overlay(points_oslo):
    warnings.filterwarnings(action="ignore", category=UserWarning)

    p = points_oslo
    p = p.iloc[:50]

    p500 = sg.buff(p, 500)
    p500["idx1"] = 1
    p1000 = sg.buff(p, 1000)
    p1000["idx2"] = 1

    overlayed = sg.clean_overlay(p500, p1000, how="update")
    cols_should_be = ["idx", "idx1", "idx2", "geometry"]
    assert list(overlayed.columns) == cols_should_be, list(overlayed.columns)

    hows = [
        "difference",
        "symmetric_difference",
        "union",
        "identity",
        "intersection",
    ]
    cols_should_be = [
        ["idx", "idx1", "geometry"],
        ["idx_1", "idx1", "idx_2", "idx2", "geometry"],
        ["idx_1", "idx1", "idx_2", "idx2", "geometry"],
        ["idx_1", "idx1", "idx_2", "idx2", "geometry"],
        ["idx_1", "idx1", "idx_2", "idx2", "geometry"],
    ]
    for cols, how in zip(cols_should_be, hows):
        print(how)
        print(p500.columns)
        overlayed = (
            sg.clean_geoms(p500)
            .explode(ignore_index=True)
            .overlay(sg.clean_geoms(p1000).explode(ignore_index=True), how=how)
            .explode(ignore_index=True)
            .explode(ignore_index=True)
        )
        assert list(overlayed.columns) == cols, list(overlayed.columns)

        overlayed2 = sg.clean_overlay(p500, p1000, how=how)
        assert list(overlayed2.columns) == cols, list(overlayed2.columns)

        if int(overlayed.area.sum()) != int(overlayed2.area.sum()):
            raise ValueError(int(overlayed.area.sum()) != int(overlayed2.area.sum()))

        if len(overlayed) != len(overlayed2):
            raise ValueError(how, len(overlayed), len(overlayed2))

        # area is slightly different, but same area with 3 digits is good enough
        for i in [1, 2, 3]:
            if round(sum(overlayed.area), i) != round(sum(overlayed2.area), i):
                raise ValueError(
                    how,
                    i,
                    round(sum(overlayed.area), i),
                    round(sum(overlayed2.area), i),
                )

            sg.clean_overlay(
                p500.sample(1),
                p1000.sample(1),
                how=how,
                geom_type="polygon",
            )


def test_overlay_random(n=25):
    for _ in range(n):
        print(_)
        loc_num = np.random.randint(low=50, high=150)
        buff_num = np.random.randint(low=7, high=23)
        for how in [
            "intersection",
            "difference",
            "symmetric_difference",
            "union",
            "identity",
        ]:
            # these geometries are singlepart and fixed (in sg.buff), so
            # results should be identical as is
            gdf1 = sg.random_points(n, loc=loc_num).pipe(sg.buff, buff_num)
            gdf2 = sg.random_points(n, loc=loc_num).pipe(sg.buff, buff_num * 0.5)

            overlayed = (
                gdf1.overlay(gdf2, how=how)
                .explode(ignore_index=True)
                .explode(ignore_index=True)
            )
            overlayed2 = sg.clean_overlay(gdf1, gdf2, how=how)

            if len(overlayed) != len(overlayed2):
                raise ValueError(how, len(overlayed), len(overlayed2))

            i = 3
            if round(sum(overlayed.area), i) != round(sum(overlayed2.area), i):
                raise ValueError(
                    how,
                    i,
                    round(sum(overlayed.area), i),
                    round(sum(overlayed2.area), i),
                )


def partial_func():
    from oslo import points_oslo

    test_overlay(points_oslo())
    test_overlay_random(n=100)


def main():
    df = cprofile_df("partial_func()")

    display(df.sort_values("cumtime", ascending=False).head(50))
    display(df.sort_values("percall", ascending=False).head(50))


if __name__ == "__main__":
    partial_func()
    # main()
