# %%

import random
import sys
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import extract_unique_points
from shapely import unary_union

src = str(Path(__file__).parent.parent) + "/src"

sys.path.insert(0, src)

from conftest import testgdf

import sgis as sg


def not_test_dissexp_n_jobs():
    n = 1000
    gdf = (
        sg.random_points(n)
        .assign(
            x=[np.random.choice([*"abcde"]) for _ in range(n)],
            y=[np.random.choice([*"abcde"]) for _ in range(n)],
        )
        .pipe(sg.buff, 0.07)
    )

    for n_jobs in [
        1,
        2,
        4,
        8,
    ]:
        print(
            n_jobs,
            timeit.timeit(lambda: sg.clean_overlay(gdf, gdf, n_jobs=n_jobs), number=5),
        )
        print(
            n_jobs,
            "                      ",
            timeit.timeit(
                lambda: sg.clean_overlay(gdf, gdf, how="difference", n_jobs=n_jobs),
                number=5,
            ),
        )

        print(
            n_jobs,
            "                                           ",
            timeit.timeit(
                lambda: sg.dissexp(gdf, by=["x", "y"], n_jobs=n_jobs), number=5
            ),
        )


def test_dissexp_by_cluster():
    gdf = sg.random_points(100).assign(
        x=[np.random.choice([*"abc"]) for _ in range(100)],
        y=[np.random.choice([*"abc"]) for _ in range(100)],
    )

    for n_jobs in [
        1,
        3,
    ]:
        by_cluster = sg.dissexp_by_cluster(gdf, n_jobs=n_jobs)
        regular = sg.dissexp(gdf, n_jobs=n_jobs)
        assert len(by_cluster) == len(regular)
        assert round(by_cluster.area.sum(), 3) == round(regular.area.sum(), 3)

        assert list(sorted(by_cluster.columns)) == [
            "geometry",
            "x",
            "y",
        ], by_cluster.columns

        assert list(regular.columns) == ["x", "y", "geometry"], regular.columns

        diss = sg.diss_by_cluster(gdf, by="x", n_jobs=n_jobs)
        assert list(sorted(diss.columns)) == ["geometry", "y"], diss.columns
        diss = sg.diss_by_cluster(gdf, by=["x", "y"], n_jobs=n_jobs)
        assert list(diss.columns) == ["geometry"], diss.columns
        diss = sg.diss_by_cluster(gdf, by=("y",), n_jobs=n_jobs)
        assert list(sorted(diss.columns)) == ["geometry", "x"], diss.columns

        diss = sg.dissexp_by_cluster(gdf, by="x", n_jobs=n_jobs)
        assert list(sorted(diss.columns)) == ["geometry", "y"], diss.columns
        diss = sg.dissexp_by_cluster(gdf, by=["x", "y"], n_jobs=n_jobs)
        assert list(diss.columns) == ["geometry"], diss.columns
        diss = sg.dissexp_by_cluster(gdf, by=("y",), n_jobs=n_jobs)
        assert list(sorted(diss.columns)) == ["geometry", "x"], diss.columns

        sg.buffdissexp_by_cluster(gdf, 0.1, n_jobs=n_jobs).pipe(sg.buff, 0.1).pipe(
            sg.buffdissexp_by_cluster, 0.1, n_jobs=n_jobs
        )


def test_buffdissexp_by_cluster(gdf_fixture):
    for distance in [1, 10, 100, 1000, 10000]:
        by_cluster = sg.buffdissexp_by_cluster(gdf_fixture, distance)
        regular = sg.buffdissexp(gdf_fixture, distance)
        assert len(by_cluster) == len(regular)
        assert round(by_cluster.area.sum(), 3) == round(regular.area.sum(), 3)

    gdf = sg.random_points(100).assign(
        x=np.random.choice([*"abc"]), y=np.random.choice([*"abc"])
    )
    by_cluster = sg.buffdissexp_by_cluster(gdf, 0.1)
    regular = sg.buffdissexp(gdf, 0.1)
    assert len(by_cluster) == len(regular)
    assert round(by_cluster.area.sum(), 3) == round(regular.area.sum(), 3)

    assert list(sorted(by_cluster.columns)) == [
        "geometry",
        "x",
        "y",
    ], by_cluster.columns
    assert list(sorted(regular.columns)) == ["geometry", "x", "y"], regular.columns

    diss = sg.buffdissexp_by_cluster(gdf, 1, by="x")
    assert list(sorted(diss.columns)) == ["geometry", "y"], diss.columns
    diss = sg.buffdissexp_by_cluster(gdf, 1, by=["x", "y"])
    assert list(sorted(diss.columns)) == ["geometry"], diss.columns
    diss = sg.buffdissexp_by_cluster(gdf, 1, by=("y",))
    assert list(sorted(diss.columns)) == ["geometry", "x"], diss.columns


def test_buffdissexp(gdf_fixture):
    sg.buffdissexp(gdf_fixture, 10, ignore_index=True)

    for distance in [1, 10, 100, 1000, 10000]:
        copy = gdf_fixture.copy()

        # with geopandas
        copy["geometry"] = copy.buffer(distance, resolution=50).make_valid()
        copy = copy.dissolve(by="txtcol")
        copy["geometry"] = copy.make_valid()
        copy = copy.explode(index_parts=False)

        copy2 = sg.buffdissexp(gdf_fixture, distance, by="txtcol")

        copy2 = copy2.loc[:, copy.columns]

        assert copy.equals(copy2), (
            copy,
            copy2,
            sg.explore(
                copy,
                copy2,
            ),
        )


def test_buffdiss(gdf_fixture):
    sg.buffdiss(gdf_fixture, 10)

    for distance in [1, 10, 100, 1000, 10000]:
        print("distance", distance)
        copy = gdf_fixture.copy()

        # with geopandas
        copy["geometry"] = copy.buffer(distance, resolution=50).make_valid()
        copy = copy.dissolve(by="txtcol")
        copy["geometry"] = copy.make_valid()

        copy2 = sg.buffdiss(gdf_fixture, distance, by="txtcol")

        copy2 = copy2.loc[:, copy.columns]

        assert copy.equals(copy2), (
            copy,
            copy2,
            len(copy.geometry.apply(extract_unique_points).explode()),
            len(copy2.geometry.apply(extract_unique_points).explode()),
            sg.explore(
                copy,
                copy2,
                copy_p=copy.assign(
                    geometry=lambda x: extract_unique_points(x.geometry.values)
                ).explode(),
                copy2_p=copy2.assign(
                    geometry=lambda x: extract_unique_points(x.geometry.values)
                ).explode(),
                # ,
            ),
        )


def test_dissexp(gdf_fixture):
    sg.dissexp(gdf_fixture, ignore_index=True)

    copy = gdf_fixture.copy()

    # with geopandas
    copy = copy.dissolve(by="txtcol")
    copy["geometry"] = copy.make_valid()
    copy = copy.explode(index_parts=False)

    copy2 = sg.dissexp(gdf_fixture, by="txtcol")

    copy2 = copy2.loc[:, copy.columns]

    assert copy.equals(copy2), (
        copy,
        copy2,
        sg.explore(
            copy,
            copy2,
        ),
    )

    gdf = sg.random_points(10).pipe(sg.buff, 1)
    gdf.index = [0, 0, 1, 1, 1, 2, 2, 1, 2, 3]
    gdf["col"] = gdf.index
    dissexped = sg.dissexp(gdf, by="col")
    assert len(dissexped) == 4

    dissexped = sg.dissexp(gdf, level=0)
    assert len(dissexped) == 4


def test_dissexp_grid_size():
    test_buffdissexp_index(grid_size=1e-4)
    test_buffdissexp_index(grid_size=1e-2)


def test_buffdissexp_index(grid_size=None):
    gdf = sg.to_gdf([(0, 0), (1, 1), (2, 2)]).assign(cat=[*"aab"])

    # when not dissolving by column, index should be 0, 1, 2...
    singlepart = sg.buffdissexp(gdf, 0.1, grid_size=grid_size)
    assert list(singlepart.index) == [0, 1, 2], singlepart
    assert list(singlepart.columns) == ["cat", "geometry"], singlepart

    # when by and not as_index, index should be 0, 1, 2...
    singlepart = sg.buffdissexp(gdf, 0.1, by="cat", as_index=False, grid_size=grid_size)
    assert list(singlepart.index) == [0, 1, 2], singlepart
    assert list(singlepart.cat) == [*"aab"], singlepart
    assert list(singlepart.columns) == ["cat", "geometry"], singlepart

    # to respect geopandas, only specifying 'by' should give 'by' column as index
    singlepart = sg.buffdissexp(gdf, 0.1, by="cat", grid_size=grid_size)
    assert list(singlepart.index) == [*"aab"], singlepart
    assert list(singlepart.columns) == ["geometry"], singlepart

    singlepart = sg.buffdissexp(gdf[["geometry"]], 0.1, grid_size=grid_size)
    assert list(singlepart.index) == [0, 1, 2], singlepart
    assert list(singlepart.columns) == ["geometry"], singlepart

    singlepart = sg.buffdissexp(gdf, 1, grid_size=grid_size)
    assert list(singlepart.index) == [0], singlepart
    assert (list(sorted(singlepart.columns))) == (
        ["cat", "geometry"]
    ), singlepart.columns

    # if the index is set, it will be gone after dissolve (same in geopandas)
    singlepart = sg.buffdissexp(gdf.set_index("cat"), 0.1, grid_size=grid_size)
    assert list(singlepart.index) == [0, 1, 2], singlepart
    assert list(singlepart.columns) == ["geometry"], singlepart

    singlepart = sg.buffdissexp(
        gdf, 0.1, by="cat", ignore_index=True, grid_size=grid_size
    )
    assert list(singlepart.index) == [0, 1, 2], singlepart


def test_dissexp_index():
    gdf = sg.to_gdf([(0, 0), (1, 1), (2, 2)]).assign(cat=[*"aab"])

    # this is what is returned with the geopandas defaults
    singlepart = sg.dissexp(gdf, ignore_index=False)
    assert list(singlepart.index) == [0, 0, 0], singlepart
    assert (list(singlepart.columns)) == (["cat", "geometry"]), singlepart

    # if dissolve by and default explode, the by column is removed completely
    singlepart = sg.dissexp(gdf, by="cat", ignore_index=True)
    assert list(singlepart.index) == [0, 1, 2], singlepart
    assert list(singlepart.columns) == ["geometry"], singlepart

    # when not dissolving by column, index should be 0, 1, 2...
    singlepart = sg.dissexp(gdf)
    assert list(singlepart.index) == [0, 1, 2], singlepart
    assert list(singlepart.columns) == ["cat", "geometry"], singlepart

    # when by and not as_index, index should be 0, 1, 2...
    singlepart = sg.dissexp(gdf, by="cat", as_index=False)
    assert list(singlepart.index) == [0, 1, 2], singlepart
    assert list(singlepart.cat) == [*"aab"], singlepart
    assert list(singlepart.columns) == ["cat", "geometry"], singlepart

    # to respect geopandas, only specifying 'by' should give 'by' column as index
    singlepart = sg.dissexp(gdf, by="cat")
    assert list(singlepart.index) == [*"aab"], singlepart
    assert list(singlepart.columns) == ["geometry"], singlepart

    singlepart = sg.dissexp(gdf[["geometry"]])
    assert list(singlepart.index) == [0, 1, 2], singlepart
    assert list(singlepart.columns) == ["geometry"], singlepart

    singlepart = sg.dissexp(gdf.set_index("cat"))
    assert list(singlepart.index) == [0, 1, 2], singlepart
    assert list(singlepart.columns) == ["geometry"], singlepart

    singlepart = sg.dissexp(gdf, ignore_index=False)
    assert list(singlepart.index) == [0, 0, 0], singlepart
    assert list(singlepart.columns) == ["cat", "geometry"], singlepart

    singlepart = sg.dissexp(gdf, by="cat", ignore_index=True)
    assert list(singlepart.index) == [0, 1, 2], singlepart


def test_grouped_unary_union():
    number = 5

    df = sg.random_points(10000).pipe(sg.buff, 0.01)
    df["col"] = [random.choice(list(range(250))) for _ in range(len(df))]

    for _ in range(2):
        dissolved = sg.geopandas_tools.general._grouped_unary_union(df, by="col")
        dissolved2 = (
            df.groupby("col")["geometry"].agg(lambda x: (unary_union(x))).make_valid()
        )

    assert dissolved.equals(dissolved2), (dissolved, dissolved2)

    def grouped_unary_union():
        return sg.geopandas_tools.general._grouped_unary_union(df, by="col")

    def groupby_unary_union():
        return df.groupby("col")["geometry"].agg(unary_union).make_valid()

    times = {"grouped_unary_union": 0, "groupby_unary_union": 0}
    for _ in range(2):
        times["grouped_unary_union"] = timeit.timeit(grouped_unary_union, number=number)
        times["groupby_unary_union"] = timeit.timeit(groupby_unary_union, number=number)

    print(pd.Series(times).apply(lambda x: round(x, 1)))


if __name__ == "__main__":

    test_grouped_unary_union()
    gdf_fixture = testgdf()
    test_buffdiss(gdf_fixture)
    test_dissexp_index()
    test_buffdissexp(gdf_fixture)
    test_dissexp(gdf_fixture)
    test_buffdissexp_index()
    test_dissexp_by_cluster()
    test_dissexp_grid_size()
    # not_test_dissexp_n_jobs()
