# %%

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest


src = str(Path(__file__).parent.parent) + "/src"

sys.path.insert(0, src)

import sgis as sg


def test_random_get_duplicate_areas():
    circles = sg.random_points(15).set_crs(25833).buffer(0.1)
    the_overlap = sg.get_duplicate_areas(circles)

    the_overlap["idx"] = [str(x) for x in range(len(the_overlap))]

    overlapping_now = sg.get_duplicate_areas(the_overlap)

    assert not len(overlapping_now)


def test_get_duplicate_areas():
    circles = sg.to_gdf([(0, 0), (0, 1), (1, 1), (1, 0)]).pipe(sg.buff, 1)

    dups = sg.get_duplicate_areas(circles, keep="first")
    print(dups)
    sg.qtm(dups.pipe(sg.buff, -0.025), alpha=0.2, column="x")
    assert len(dups) == 8, len(dups)

    dups = sg.get_duplicate_areas(circles, keep="last")
    assert len(dups) == 8, len(dups)
    print(dups)
    sg.qtm(dups.pipe(sg.buff, -0.025), alpha=0.2, column="area")

    dups = sg.get_duplicate_areas(circles, keep=False)
    assert len(dups) == 12, len(dups)
    print(dups)
    sg.qtm(dups.pipe(sg.buff, -0.025), alpha=0.2, column="area")


def _test_get_duplicate_areas():
    with_overlap = sg.to_gdf([(0, 0), (4, 4), (1, 1)]).pipe(sg.buff, 1)
    with_overlap["col"] = 1
    if __name__ == "__main__":
        sg.explore(with_overlap)

    dissolved_overlap = sg.get_duplicate_areas(with_overlap)
    print(dissolved_overlap)
    assert len(dissolved_overlap) == 1
    assert list(dissolved_overlap.columns) == ["geometry"]
    assert round(dissolved_overlap.area.sum(), 2) == 0.57, dissolved_overlap.area.sum()

    again = sg.get_duplicate_areas(dissolved_overlap)
    print(again)
    assert not len(again)

    # index should be preserved and area should be twice
    without_dissolve = sg.get_duplicate_areas(with_overlap, dissolve=False)
    print(without_dissolve)
    assert list(without_dissolve.index) == [0, 2], list(without_dissolve.index)
    assert (
        round(without_dissolve.area.sum(), 2) == 0.57 * 2
    ), without_dissolve.area.sum()

    assert list(without_dissolve.columns) == ["col", "geometry"]

    again = sg.get_duplicate_areas(without_dissolve)
    print(again)
    assert len(again) == 1, len(again)

    once_again = sg.get_duplicate_areas(again)
    print(once_again)
    assert not len(once_again)


def test_close_holes():
    p = sg.to_gdf([0, 0])

    buff1 = sg.buffdissexp(p, 100)

    no_holes_closed = sg.close_all_holes(buff1)
    assert round(np.sum(no_holes_closed.area), 3) == round(np.sum(buff1.area), 3)
    no_holes_closed = sg.close_small_holes(buff1, max_area=1_000_000)
    assert round(np.sum(no_holes_closed.area), 3) == round(np.sum(buff1.area), 3)

    buff2 = sg.buffdissexp(p, 200)
    ring_with_hole = sg.clean_overlay(buff2, buff1, how="difference")

    buff0 = sg.buffdissexp(p, 30)
    ring_with_hole_and_island = pd.concat([ring_with_hole, buff0])

    # run this for different geometry input types
    def _close_the_holes(ring_with_hole):
        all_closed = sg.close_all_holes(ring_with_hole)
        assert sum(all_closed.area) > sum(ring_with_hole.area)

        # this should return the entire hole
        closed_island_ignored = sg.close_all_holes(
            ring_with_hole, without_islands=False
        )
        assert sum(closed_island_ignored.area) > sum(all_closed.area)

        hole_not_closed = sg.close_small_holes(ring_with_hole, max_area=1)
        assert sum(all_closed.area) > sum(ring_with_hole.area)

        all_closed2 = sg.close_small_holes(ring_with_hole, max_area=30_000)

        assert round(np.sum(all_closed2.area.iloc[0]), 3) == round(
            np.sum(all_closed.area.iloc[0]), 3
        ), (
            round(np.sum(all_closed2.area.iloc[0]), 3),
            round(np.sum(all_closed.area.iloc[0]), 3),
        )

        assert sum(all_closed2.area) > sum(ring_with_hole.area)

        hole_not_closed2 = sg.close_small_holes(ring_with_hole, max_area=20_000)
        assert np.sum(hole_not_closed2.area) == np.sum(ring_with_hole.area), (
            np.sum(hole_not_closed2.area),
            np.sum(ring_with_hole.area),
        )

        # this should return the entire hole
        without_islands = False
        all_closed3 = sg.close_small_holes(
            ring_with_hole, max_area=32_000, without_islands=without_islands
        )

        assert round(np.sum(all_closed3.area), 3) > round(np.sum(all_closed.area), 3)

        hole_not_closed3 = sg.close_small_holes(
            ring_with_hole, max_area=30_000, without_islands=without_islands
        )
        assert np.sum(hole_not_closed3.area) == np.sum(ring_with_hole.area)

    _close_the_holes(ring_with_hole_and_island)
    _close_the_holes(ring_with_hole_and_island.geometry)


def test_get_polygon_clusters():
    gdf = sg.to_gdf([(0, 0)]).loc[lambda x: x.index > 0]
    assert len(gdf) == 0
    c = sg.get_polygon_clusters(gdf)

    INDEX = [1, 3, 5, 7, 9, 11, 13]
    # this should give three clusters
    gdf = sg.to_gdf(
        [(0, 0), (1, 1), (0, 1), (1, 0), (10, 10), (20, 20), (21, 21)]
    ).pipe(sg.buff, 2)
    gdf.index = INDEX

    should_give = pd.Series([4, 2, 1], name="cluster", index=[0, 2, 1])

    gdf_clustered = sg.get_polygon_clusters(gdf)

    if __name__ == "__main__":
        sg.explore(gdf_clustered, "cluster")

    print(gdf_clustered)
    assert gdf_clustered.cluster.value_counts().equals(
        should_give
    ), gdf_clustered.cluster.value_counts()

    # two gdfs at the same time
    clustered1, clustered2 = sg.get_polygon_clusters(gdf, gdf)
    print(clustered1, clustered2)
    assert clustered1.cluster.value_counts().equals(
        should_give
    ), clustered1.cluster.value_counts()
    assert clustered2.cluster.value_counts().equals(
        should_give
    ), clustered2.cluster.value_counts()

    assert list(clustered1.index) == INDEX
    assert list(clustered2.index) == INDEX

    # check that index is preserved if MultiIndex on one
    multiindex = pd.MultiIndex.from_arrays([(0, 0, 0, 1, 1, 1, 2), list(range(7))])
    gdf2 = gdf.copy()
    gdf2.index = multiindex
    clustered1, clustered2 = sg.get_polygon_clusters(gdf, gdf2)
    assert list(clustered1.index) == INDEX
    assert clustered2.index.equals(multiindex)

    s_clustered = sg.get_polygon_clusters(gdf.geometry)
    print("s_clustered")
    print(s_clustered)
    assert s_clustered.cluster.value_counts().equals(
        should_give
    ), s_clustered.cluster.value_counts()

    should_give = pd.Series([7], name="cluster", index=[0])

    buffered_clustered = sg.get_polygon_clusters(gdf.buffer(10))
    print(buffered_clustered)
    assert buffered_clustered.cluster.value_counts().equals(
        should_give
    ), buffered_clustered.cluster.value_counts()

    if __name__ == "__main__":
        sg.explore(buffered_clustered, "cluster")

    buffered_clustered = sg.get_polygon_clusters(gdf, "col")
    print(buffered_clustered)
    assert list(buffered_clustered.columns) == ["col", "geometry"], list(
        buffered_clustered.columns
    )

    with pytest.raises(ValueError):
        sg.get_polygon_clusters(gdf.dissolve())

    sg.get_polygon_clusters(gdf.dissolve(), allow_multipart=True)

    c = sg.get_polygon_clusters(gdf, wkt_col=True)
    assert c["cluster"].isin(["0_0", "9_9", "20_20"]).all()


def test_eliminate():
    from shapely.geometry import Polygon

    sliver = sg.to_gdf(Polygon([(0, 0), (0.1, 1), (0, 2), (-0.1, 1)])).assign(
        what="sliver", num=1
    )
    small_poly = sg.to_gdf(
        Polygon([(0, 0), (-0.1, 1), (0, 2), (-1, 2), (-2, 2), (-1, 1)])
    ).assign(what="small", num=2)
    large_poly = sg.to_gdf(
        Polygon([(0, 0), (0.1, 1), (1, 2), (2, 2), (3, 2), (3, 0)])
    ).assign(what="large", num=3)
    isolated = sg.to_gdf(
        Polygon([(10, 10), (-10.1, 11), (10, 12), (-11, 12), (-12, 12), (-11, 11)])
    ).assign(what="isolated", num=4)

    polys1 = pd.concat([small_poly, large_poly], ignore_index=True)
    polys2 = pd.concat([sliver, small_poly, large_poly], ignore_index=True)

    if __name__ == "__main__":
        sg.qtm(polys2, "what", alpha=0.8)
    polys1.index = [5, 7]
    polys2.index = [3, 5, 7]
    assert list(polys2.area) == [0.2, 1.9, 5.4], list(polys2.area)

    for polys in [polys1, polys2]:
        eliminated = sg.eliminate_by_longest(polys, sliver)

        if __name__ == "__main__":
            sg.qtm(eliminated, "what", title="after eliminate_by_longest", alpha=0.8)
        assert list(eliminated.index) == [5, 7], list(eliminated.index)
        assert list(eliminated.num) == [2, 3], list(eliminated.num)
        assert list(eliminated.what) == ["small", "large"], list(eliminated.what)
        assert list(round(eliminated.area, 1)) == [2.1, 5.4], list(eliminated.area)

        eliminated = sg.eliminate_by_longest(
            polys, sliver, aggfunc={"num": "sum", "what": "first"}
        )
        assert list(eliminated.num) == [3, 3], list(eliminated.num)

        eliminated = sg.eliminate_by_largest(polys, sliver)
        if __name__ == "__main__":
            sg.qtm(eliminated, "what", title="after eliminate_by_largest", alpha=0.8)
        assert list(eliminated.index) == [5, 7], list(eliminated.index)
        assert list(eliminated.num) == [2, 3], list(eliminated.num)
        assert list(eliminated.what) == ["small", "large"], list(eliminated.what)
        assert list(round(eliminated.area, 1)) == [1.9, 5.6], list(eliminated.area)

        eliminated = sg.eliminate_by_largest(
            polys, sliver, aggfunc={"num": "sum", "what": "first"}
        )
        assert list(eliminated.num) == [2, 4], list(eliminated.num)

        eliminated = sg.eliminate_by_smallest(
            polys, sliver, aggfunc={"num": "sum", "what": "first"}
        )
        if __name__ == "__main__":
            sg.qtm(eliminated, "what", title="after eliminate_by_smallest", alpha=0.8)
        assert list(eliminated.index) == [5, 7], list(eliminated.index)
        assert list(eliminated.num) == [3, 3], list(eliminated.num)
        assert list(eliminated.what) == ["small", "large"], list(eliminated.what)
        assert list(round(eliminated.area, 1)) == [2.1, 5.4], list(eliminated.area)

    eliminated = sg.eliminate_by_longest(polys1, isolated)
    assert list(eliminated.what) == ["small", "large", "isolated"], list(
        eliminated.what
    )
    assert list(eliminated.index) == [5, 7, 0], list(eliminated.index)
    if __name__ == "__main__":
        sg.qtm(eliminated, "what", title="with isolated", alpha=0.8)

    eliminated = sg.eliminate_by_largest(polys1, isolated)
    assert list(eliminated.what) == ["small", "large", "isolated"], list(
        eliminated.what
    )
    assert list(eliminated.index) == [5, 7, 0], list(eliminated.index)


if __name__ == "__main__":
    test_get_duplicate_areas()
    test_random_get_duplicate_areas()
    sss
    test_close_holes()
    test_get_polygon_clusters()
    test_eliminate()
