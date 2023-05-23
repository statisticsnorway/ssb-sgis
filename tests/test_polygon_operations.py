# %%

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest


src = str(Path(__file__).parent.parent) + "/src"

sys.path.insert(0, src)

import sgis as sg


def test_get_overlapping_polygons():
    with_overlap = sg.to_gdf([(0, 0), (4, 4), (1, 1)]).pipe(sg.buff, 1)
    with_overlap["col"] = 1
    if __name__ == "__main__":
        sg.explore(with_overlap)

    the_overlap = sg.get_overlapping_polygons(with_overlap)
    print(the_overlap)
    if __name__ == "__main__":
        sg.explore(the_overlap)
    assert list(the_overlap.index) == [0, 2]
    assert list(the_overlap.columns) == ["col", "geometry"]
    assert round(the_overlap.area.sum(), 5) == 1.14108, round(the_overlap.area.sum(), 5)

    overlapping_polygons = sg.get_overlapping_polygon_indices(with_overlap)
    print(overlapping_polygons)
    assert isinstance(overlapping_polygons, pd.Index)
    assert list(overlapping_polygons) == [0, 2], overlapping_polygons
    without_the_overlapping = with_overlap.loc[
        ~with_overlap.index.isin(overlapping_polygons)
    ]
    assert list(without_the_overlapping.index) == [1], without_the_overlapping

    overlapping_polygons = sg.get_overlapping_polygon_product(with_overlap)
    print(overlapping_polygons)
    should_be = pd.Series([0, 2], index=[2, 0])
    assert overlapping_polygons.equals(should_be), overlapping_polygons
    without_the_overlapping = with_overlap.loc[
        ~with_overlap.index.isin(overlapping_polygons)
    ]
    assert list(without_the_overlapping.index) == [1], without_the_overlapping


def test_close_holes():
    p = sg.to_gdf([0, 0])

    buff1 = sg.buffdissexp(p, 100)

    no_holes_closed = sg.close_all_holes(buff1)
    assert round(sum(no_holes_closed.area), 3) == round(sum(buff1.area), 3)
    no_holes_closed = sg.close_small_holes(buff1, 10000 * 1_000_000)
    assert round(sum(no_holes_closed.area), 3) == round(sum(buff1.area), 3)

    buff2 = sg.buffdissexp(p, 200)
    rings_with_holes = sg.clean_overlay(buff2, buff1, how="difference")

    # run this for different geometry input types
    def _close_the_holes(rings_with_holes):
        holes_closed = sg.close_all_holes(rings_with_holes)
        if hasattr(holes_closed, "area"):
            assert sum(holes_closed.area) > sum(rings_with_holes.area)

        holes_closed2 = sg.close_small_holes(
            rings_with_holes, max_area=10000 * 1_000_000
        )
        if hasattr(holes_closed, "area"):
            assert round(sum(holes_closed2.area), 3) == round(sum(holes_closed.area), 3)
            assert sum(holes_closed2.area) > sum(rings_with_holes.area)

        holes_not_closed = sg.close_small_holes(rings_with_holes, max_area=1)
        if hasattr(holes_closed, "area"):
            assert sum(holes_not_closed.area) == sum(rings_with_holes.area)
        else:
            assert sum(gpd.GeoSeries(holes_not_closed).area) == sum(
                rings_with_holes.area
            )

    _close_the_holes(rings_with_holes)
    _close_the_holes(rings_with_holes.geometry)


def test_get_polygon_clusters():
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


if __name__ == "__main__":
    test_get_polygon_clusters()
    test_get_overlapping_polygons()

    test_close_holes()
