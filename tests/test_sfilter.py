# %%

import random
import sys
import timeit
from pathlib import Path

import pandas as pd

src = str(Path(__file__).parent.parent) + "/src"

sys.path.insert(0, src)

import sgis as sg


def test_sfilter():
    def sfilter_asserts(df1, df2, should_equal):
        for predicate in ["intersects", "within"]:
            print(predicate)
            assert sg.sfilter(df1, df2, predicate=predicate).equals(
                should_equal
            ), sg.sfilter(df1, df2, predicate=predicate)
            assert sg.sfilter(df1, df2.geometry, predicate=predicate).equals(
                should_equal
            )
            assert sg.sfilter(df1, df2.geometry.values, predicate=predicate).equals(
                should_equal
            )
            assert sg.sfilter(df1, df2.geometry.to_numpy(), predicate=predicate).equals(
                should_equal
            )
            assert sg.sfilter(df1, df2.geometry.tolist(), predicate=predicate).equals(
                should_equal
            )
            assert sg.sfilter(df1, df2.unary_union, predicate=predicate).equals(
                should_equal
            )
            assert sg.sfilter(
                df1.geometry, df2.unary_union, predicate=predicate
            ).equals(should_equal.geometry)

    df1 = sg.to_gdf([(0, 0), (0, 1), (1, 1)])
    df1["idx"] = [3, 1, 2]
    df2 = sg.to_gdf([(0, 0), (0, 0), (1, 1), (1, 2)])

    should_equal = sg.to_gdf([(0, 0), (1, 1)])
    should_equal.index = [0, 2]
    should_equal["idx"] = [3, 2]

    should_not_equal = sg.to_gdf([(0, 1)])
    should_not_equal.index = [1]
    should_not_equal["idx"] = [1]

    sfilter_asserts(df1, df2, should_equal)

    intersect, not_intersect = sg.sfilter_split(df1, df2)
    assert intersect.equals(should_equal)
    assert not_intersect.equals(should_not_equal), not_intersect

    # with non-unique index
    df1.index = [0, 1, 0]
    should_equal.index = [0, 0]

    sfilter_asserts(df1, df2, should_equal)


def test_sfilter_random():
    for _ in range(25):
        gdf = sg.random_points(12)
        gdf.geometry = gdf.buffer(random.random())
        other = sg.random_points(6)
        other.geometry = other.buffer(random.random())
        intersecting, not_intersecting = sg.sfilter_split(gdf, other)

        assert intersecting.equals(sg.sfilter(gdf, other))
        assert not_intersecting.equals(sg.sfilter_inverse(gdf, other))

        filt = gdf.intersects(other.unary_union)
        assert intersecting.equals(gdf[filt]), (intersecting, gdf[filt])
        assert not_intersecting.equals(gdf[~filt]), (intersecting, gdf[~filt])


def benchmark_within_vs_intersects():
    def intersects_within(gdf, other):
        within = sg.sfilter(gdf, other, "within")
        intersects = sg.sfilter(
            gdf.loc[~gdf.index.isin(within.index)], other, "intersects"
        )
        return pd.concat([within, intersects]).sort_index()

    gdf = sg.random_points(1000).pipe(sg.buff, 0.01)
    other = sg.random_points(100).pipe(sg.buff, 0.1)

    s = sg.sfilter(gdf, other, "intersects")
    s2 = intersects_within(gdf, other)
    print(s)
    print(s2)
    assert s.equals(s2)

    l = []
    for rows in [2000]:
        for div in [50, 25, 10, 6, 4, 2, 1, 0.5]:
            for buffdiv in [50, 25, 10, 5, 2, 1, 0.5]:
                for _ in range(20):
                    gdf = sg.random_points(rows).pipe(sg.buff, 0.1 / buffdiv)
                    other = sg.random_points(rows // div).pipe(sg.buff, 0.1)

                    tid_intersects = timeit.timeit(
                        lambda: sg.sfilter(gdf, other, "intersects"), number=1
                    )
                    tid_begge = timeit.timeit(
                        lambda: intersects_within(gdf, other),
                        number=1,
                    )

                    print(rows, div, buffdiv, round(tid_intersects / tid_begge, 1))

                    l.append((rows, div, buffdiv, tid_intersects / tid_begge))

    df = pd.DataFrame(l, columns=("rows", "div", "buffdiv", "ratio"))
    print(df)
    print(df.groupby("rows")["ratio"].mean())
    print(df.groupby("div")["ratio"].mean())
    print(df.groupby("buffdiv")["ratio"].mean())


if __name__ == "__main__":
    test_sfilter()
    test_sfilter_random()
