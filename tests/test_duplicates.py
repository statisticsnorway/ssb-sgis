# %%

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


src = str(Path(__file__).parent.parent) + "/src"

sys.path.insert(0, src)

import sgis as sg


def test_random_get_intersections():
    # many iterations to try to break the assertion
    for i in range(100):
        circles = sg.random_points(15).set_crs(25833).buffer(0.1).to_frame()
        the_overlap = sg.get_intersections(circles)

        updated = sg.update_geometries(the_overlap)

        overlapping_now = sg.get_intersections(updated).loc[
            lambda x: x.area / x.length > 1e-12
        ]

        assert not len(overlapping_now), overlapping_now.assign(
            sliv=lambda x: x.area / x.length
        )


def test_drop_duplicate_geometries():
    circles = sg.to_gdf([(0, 0), (0, 1), (1, 1), (1, 0)]).pipe(sg.buff, 1)

    dups = sg.get_intersections(circles)
    assert len(dups) == 12, len(dups)

    no_dups = sg.drop_duplicate_geometries(dups)
    assert len(no_dups) == 6, len(no_dups)


def test_get_intersections():
    circles = sg.to_gdf([(0, 0), (0, 1), (1, 1), (1, 0)]).pipe(sg.buff, 1)

    dups = (
        sg.get_intersections(circles)
        .pipe(sg.update_geometries)
        .pipe(sg.buff, -0.01)
        .pipe(sg.clean_geoms)
    )
    print(dups)
    sg.qtm(dups.pipe(sg.buff, -0.025), alpha=0.2, column="area")
    assert len(dups) == 5, len(dups)

    dups = (
        sg.get_intersections(circles)
        .pipe(sg.update_geometries)
        .pipe(sg.buff, -0.01)
        .pipe(sg.clean_geoms)
    )
    print(dups)
    sg.qtm(dups.pipe(sg.buff, -0.025), alpha=0.2, column="area")
    assert len(dups) == 5, len(dups)

    dups = sg.get_intersections(circles)
    print(dups)
    sg.qtm(dups.pipe(sg.buff, -0.025), alpha=0.2, column="area")
    assert len(dups) == 12, len(dups)

    # should also work with points
    points = sg.to_gdf([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0), (1, 0)])
    assert len(sg.get_intersections(points)) == 4
    assert len(sg.update_geometries(sg.get_intersections(points))) == 2


def _test_get_intersections():
    with_overlap = sg.to_gdf([(0, 0), (4, 4), (1, 1)]).pipe(sg.buff, 1)
    with_overlap["col"] = 1
    if __name__ == "__main__":
        sg.explore(with_overlap)

    dissolved_overlap = sg.get_intersections(with_overlap)
    print(dissolved_overlap)
    assert len(dissolved_overlap) == 1
    assert list(dissolved_overlap.columns) == ["geometry"]
    assert round(dissolved_overlap.area.sum(), 2) == 0.57, dissolved_overlap.area.sum()

    again = sg.get_intersections(dissolved_overlap)
    print(again)
    assert not len(again)

    # index should be preserved and area should be twice
    without_dissolve = sg.get_intersections(with_overlap, dissolve=False)
    print(without_dissolve)
    assert list(without_dissolve.index) == [0, 2], list(without_dissolve.index)
    assert (
        round(without_dissolve.area.sum(), 2) == 0.57 * 2
    ), without_dissolve.area.sum()

    assert list(without_dissolve.columns) == ["col", "geometry"]

    again = sg.get_intersections(without_dissolve)
    print(again)
    assert len(again) == 1, len(again)

    once_again = sg.get_intersections(again)
    print(once_again)
    assert not len(once_again)


def test_update_geometries():
    coords = [(0, 0), (0, 1), (1, 1), (1, 0)]
    buffers = [0.9, 1.3, 0.7, 1.1]
    circles = sg.to_gdf(coords)
    circles["geometry"] = circles["geometry"].buffer(buffers)

    updated = sg.update_geometries(circles)
    area = list((updated.area * 10).astype(int))
    assert area == [25, 36, 4, 18], area

    updated_largest_first = sg.update_geometries(sg.sort_large_first(circles))
    area = list((updated_largest_first.area * 10).astype(int))
    assert area == [53, 24, 5, 2], area

    updated_smallest_first = sg.update_geometries(
        sg.sort_large_first(circles).iloc[::-1]
    )
    area = list((updated_smallest_first.area * 10).astype(int))
    assert area == [15, 24, 18, 26], area

    sg.explore(circles, updated, updated_largest_first, updated_smallest_first)


if __name__ == "__main__":
    test_drop_duplicate_geometries()
    test_get_intersections()
    test_random_get_intersections()
    test_update_geometries()
