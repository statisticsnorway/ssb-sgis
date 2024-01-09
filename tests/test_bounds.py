# %%
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point


src = str(Path(__file__).parent.parent) + "/src"

sys.path.insert(0, src)

import sgis as sg


"""def test_dice():
    points = sg.random_points(1000, loc=10000).set_crs(25833)
    points["col"] = np.random.choice([*"abcd"], len(points))
    points["col2"] = np.random.choice([1, 2, 3, 4], len(points))

    grid = sg.make_grid(points, gridsize=1000)
    assert all(points.intersects(grid.unary_union))
    assert list(grid.columns) == ["col", "col2", "geometry"], list(grid.columns)
    print(grid)
    ss


test_dice()"""


def test_gridloop():
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
                # mask=points,
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
                # mask=points,
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
                # mask=points,
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
                # mask=points,
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
                # mask=points,
                args=(points, grid),
                parallelizer=sg.Parallel(3, backend="multiprocessing"),
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
                # mask=points,
                args=(points, grid),
                parallelizer=sg.Parallel(3, backend="loky"),
            ),
            ignore_index=True,
        )
        .sort_values("i")
        .reset_index(drop=True)
    )

    assert intersected.equals(intersected7)


def test_bounds():
    points = sg.random_points(1000, loc=10000).set_crs(25833)

    single_cell = sg.make_grid(points, gridsize=1_000_00, clip_to_bounds=True)
    assert len(single_cell) == 1, len(single_cell)

    grid = sg.make_grid(points, gridsize=1000)
    assert all(points.intersects(grid.unary_union))

    # should work with geoseries, tuple and polygon
    sg.make_grid(points.geometry, 1000),
    sg.make_grid(points.geometry.total_bounds, 1000, crs=points.crs)
    sg.make_grid(points.unary_union, 1000, crs=points.crs)

    ssb_grid = sg.make_ssb_grid(points, gridsize=1000)
    assert all(points.intersects(ssb_grid.unary_union))

    if __name__ == "__main__":
        sg.explore(grid, ssb_grid, points)

    for _ in range(100):
        p = points.sample(10).buffer(1000 * np.random.random(1))

        grid = sg.make_grid(p, gridsize=1000)
        assert p.within(grid.unary_union).all()

        grid = sg.make_ssb_grid(p, gridsize=1000)
        assert p.within(grid.unary_union).all()

    grid = sg.make_grid_from_bbox(0, 0, 1, 1, gridsize=0.1, crs=25833)
    print(grid.total_bounds)
    grid["idx"] = grid.index
    assert len(grid) == 121, len(grid)
    if __name__ == "__main__":
        sg.qtm(grid, "idx")

    # this will create grid around this grid,
    from_bounds = sg.make_grid(grid, gridsize=0.1)
    from_bounds["idx"] = from_bounds.index
    if __name__ == "__main__":
        sg.qtm(from_bounds, grid, alpha=0.5)
    assert grid.within(from_bounds.unary_union.buffer(0.00001)).all()

    gdf = sg.to_gdf([(0, 0), (1, 1), (2, 2)]).pipe(sg.buff, 0.1)
    gdf.index = [1, 3, 5]
    for boxes in [
        sg.bounds_to_polygon(gdf),
        sg.bounds_to_polygon(gdf.geometry),
        sg.bounds_to_polygon(gdf.buffer(0.01).to_frame()),
    ]:
        assert len(gdf) == len(boxes)
        assert list(gdf.index) == list(boxes.index)
        assert not any(gdf.geometry.isna())
        if __name__ == "__main__" and isinstance(boxes, gpd.GeoDataFrame):
            sg.qtm(gdf, boxes, alpha=0.5)

    points = sg.bounds_to_points(gdf)
    if __name__ == "__main__":
        sg.qtm(gdf, boxes, alpha=0.5)
    assert len(gdf) == len(points)
    assert list(gdf.index) == list(points.index)
    assert points.intersects(boxes.unary_union).all()
    assert not any(gdf.geometry.isna())


if __name__ == "__main__":
    import cProfile

    # test_gridloop()
    # test_bounds()
    cProfile.run(
        """
test_gridloop()
test_bounds()
    """,
        sort="cumtime",
    )
