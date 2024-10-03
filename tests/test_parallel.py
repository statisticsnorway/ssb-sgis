import sys
from pathlib import Path

import pandas as pd

src = str(Path(__file__).parent.parent) + "/src"

sys.path.insert(0, src)

import sgis as sg


def func(x, *args, **kwargs):
    return x


def x2(x):
    return x * 2


def x3(x):
    return x * 3


def test_attributes():
    p = sg.Parallel(2, backend="loky")
    assert p.processes == 2
    assert p.backend == "loky"
    assert p.context == "spawn"
    assert p.kwargs == {}
    assert p.funcs == []
    assert p.results == []

    res = sg.Parallel(2, backend="loky").map(x2, [])
    assert res == []


def x2_with_arg_kwarg(x, plus, minus):
    return x * 2 + plus - minus


def add(x, y):
    return x + y


def add2(x, y, z):
    return x + y + z


# def test_run():
#     for backend in ["loky", "multiprocessing", "threading"]:
#         res_x2, res_x3 = sg.Parallel(2, backend=backend).run([x2, x3], args=(3,))
#         assert res_x2 == 6, res_x2
#         assert res_x3 == 9, res_x3


def test_map():
    for backend in ["loky", "multiprocessing", "threading"]:
        print(backend)
        iterable = [1, 2, 3, 4, 5, 6]

        res = sg.Parallel(2, backend=backend).map(func, iterable)
        print(res)
        assert res == iterable, res

        p = sg.Parallel(2, backend=backend)
        results = p.map(x2, iterable)
        assert results == [2, 4, 6, 8, 10, 12], results

        p = sg.Parallel(2, backend=backend)
        results = p.map(x2_with_arg_kwarg, iterable, kwargs={"plus": 1, "minus": 2})
        assert results == [1, 3, 5, 7, 9, 11], results


def test_chunkwise():
    df = sg.random_points(100).pipe(sg.buff, 0.1)
    df2 = df.pipe(sg.buff, 0.1)
    df2["_range_idx"] = range(len(df2))
    df["_range_idx_right"] = range(len(df))
    overlayed = sg.clean_overlay(df2, df).sort_values("_range_idx")
    assert len(overlayed)
    for backend in ["loky", "multiprocessing", "threading"]:
        print(backend)

        res = (
            sg.Parallel(2, backend=backend)
            .chunkwise(sg.clean_overlay, df2, args=(df,))
            .sort_values("_range_idx")
        )
        assert res.equals(overlayed), (overlayed, res)

        res = (
            sg.Parallel(2, backend=backend)
            .chunkwise(sg.clean_overlay, df2, kwargs=dict(df2=df))
            .sort_values("_range_idx")
        )
        assert res.equals(overlayed), (overlayed, res)

        res = (
            sg.Parallel(2, backend=backend)
            .chunkwise(sg.clean_overlay, df2, args=(df,), n_chunks=10)
            .sort_values("_range_idx")
        )
        assert res.equals(overlayed), (overlayed, res)

        res = (
            sg.Parallel(2, backend=backend)
            .chunkwise(sg.clean_overlay, df2, args=(df,), max_rows_per_chunk=10)
            .sort_values("_range_idx")
        )
        assert res.equals(overlayed), (overlayed, res)


def test_args_to_kwargs():
    def func(x, y, z):
        pass

    x = 1
    y = ["xx"]
    z = {1: "a", 2: "b"}
    args = (x, y, z)
    kwargs = sg.parallel.parallel._turn_args_into_kwargs(func, args, 0)
    assert list(kwargs) == ["x", "y", "z"], kwargs
    assert list(kwargs.values()) == [x, y, z], kwargs

    kwargs = sg.parallel.parallel._turn_args_into_kwargs(func, (y, z), 1)
    assert list(kwargs) == ["y", "z"], kwargs
    assert list(kwargs.values()) == [y, z], kwargs


def test_starmap():
    iterable = [(1, 2), (2, 3), (3, 4)]

    p = sg.Parallel(1, backend="loky")
    results = p.starmap(add, iterable)
    assert results == [3, 5, 7]

    results = p.starmap(add2, iterable, kwargs=dict(z=1))
    assert results == [4, 6, 8]

    p = sg.Parallel(2, backend="loky")
    results = p.starmap(add, [])
    assert results == []
    results = p.starmap(add, iterable)
    assert results == [3, 5, 7]

    results = p.starmap(add2, iterable, kwargs=dict(z=1))
    assert results == [4, 6, 8]

    p = sg.Parallel(2, backend="multiprocessing")
    results = p.starmap(add, iterable)
    assert results == [3, 5, 7]

    results = p.starmap(add2, iterable, kwargs=dict(z=1))
    assert results == [4, 6, 8]


def test_gridloop_parallel():
    points = sg.random_points(100, loc=10000).set_crs(25833)
    points["i"] = range(len(points))

    grid = sg.make_grid(points, 2000)
    grid["grid_idx"] = range(len(grid))
    intersected = sg.clean_overlay(points, grid).sort_values("i").reset_index(drop=True)

    intersected2 = (
        pd.concat(
            sg.gridloop(
                sg.clean_overlay,
                gridsize=200,
                mask=points,
                kwargs={"df1": points, "df2": grid},
            ),
            ignore_index=True,
        )
        .sort_values("i")
        .reset_index(drop=True)
    )
    print(intersected)
    print(intersected2)

    assert intersected.equals(intersected2)

    intersected3 = (
        pd.concat(
            sg.gridloop(
                sg.clean_overlay,
                gridsize=200,
                mask=points,
                args=(points, grid),
            ),
            ignore_index=True,
        )
        .sort_values("i")
        .reset_index(drop=True)
    )

    assert intersected.equals(intersected3)

    intersected4 = (
        pd.concat(
            sg.gridloop(
                sg.clean_overlay,
                gridsize=200,
                gridbuffer=100,
                mask=points,
                args=(points, grid),
            ),
            ignore_index=True,
        )
        .sort_values("i")
        .reset_index(drop=True)
    )

    assert intersected.equals(intersected4)

    intersected5 = (
        pd.concat(
            sg.gridloop(
                sg.clean_overlay,
                gridsize=200,
                gridbuffer=100,
                mask=points,
                args=(points, grid),
                parallelizer=sg.Parallel(1, backend="multiprocessing"),
            ),
            ignore_index=True,
        )
        .sort_values("i")
        .reset_index(drop=True)
    )

    assert intersected.equals(intersected5)

    intersected6 = (
        pd.concat(
            sg.gridloop(
                sg.clean_overlay,
                gridsize=200,
                gridbuffer=100,
                mask=points,
                args=(points, grid),
                parallelizer=sg.Parallel(2, backend="multiprocessing"),
            ),
            ignore_index=True,
        )
        .sort_values("i")
        .reset_index(drop=True)
    )

    assert intersected.equals(intersected6)

    intersected7 = (
        pd.concat(
            sg.gridloop(
                sg.clean_overlay,
                gridsize=200,
                gridbuffer=100,
                mask=points,
                args=(points, grid),
                parallelizer=sg.Parallel(2, backend="loky"),
            ),
            ignore_index=True,
        )
        .sort_values("i")
        .reset_index(drop=True)
    )

    assert intersected.equals(intersected7)


def test_gridlooper_parallel():
    points = sg.random_points(100, loc=10000).set_crs(25833)
    points["i"] = range(len(points))

    grid = sg.make_grid(points, 2000)
    grid["grid_idx"] = range(len(grid))
    intersected = sg.clean_overlay(points, grid).sort_values("i").reset_index(drop=True)

    looper = sg.Gridlooper(gridsize=200, mask=points)

    intersected2 = (
        pd.concat(
            looper.run(
                sg.clean_overlay,
                **{"df1": points, "df2": grid},
            ),
            ignore_index=True,
        )
        .sort_values("i")
        .reset_index(drop=True)
    )

    assert intersected.equals(intersected2)

    looper.concat = True

    intersected3 = (
        looper.run(
            sg.clean_overlay,
            *(points, grid),
        )
        .sort_values("i")
        .reset_index(drop=True)
    )

    assert intersected.equals(intersected3)

    looper.concat = False
    looper.gridbuffer = 100

    intersected4 = (
        pd.concat(
            looper.run(
                sg.clean_overlay,
                points,
                df2=grid,
            ),
            ignore_index=True,
        )
        .sort_values("i")
        .reset_index(drop=True)
    )

    assert intersected.equals(intersected4)

    looper.parallelizer = sg.Parallel(1, backend="multiprocessing")

    intersected5 = (
        pd.concat(
            looper.run(
                sg.clean_overlay,
                points,
                grid,
            ),
            ignore_index=True,
        )
        .sort_values("i")
        .reset_index(drop=True)
    )

    assert intersected.equals(intersected5)

    intersected6 = (
        pd.concat(
            looper.run(
                sg.clean_overlay,
                points,
                grid,
            ),
            ignore_index=True,
        )
        .sort_values("i")
        .reset_index(drop=True)
    )

    assert intersected.equals(intersected6)

    intersected7 = (
        pd.concat(
            looper.run(
                sg.clean_overlay,
                points,
                grid,
            ),
            ignore_index=True,
        )
        .sort_values("i")
        .reset_index(drop=True)
    )

    assert intersected.equals(intersected7)


if __name__ == "__main__":
    # test_run()
    test_chunkwise()
    test_args_to_kwargs()
    test_starmap()
    test_map()
    test_gridloop_parallel()
    test_gridlooper_parallel()
