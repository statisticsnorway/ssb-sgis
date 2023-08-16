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

        updated = sg.update_geometries(the_overlap, grid_size=1e-4)

        overlapping_now = sg.get_intersections(updated).loc[
            lambda x: x.area / x.length > 1e-8
        ]

        assert not len(overlapping_now), overlapping_now.assign(
            sliv=lambda x: x.area / x.length
        )


def not_test_bug():
    import geopandas as gpd
    import networkx as nx
    from shapely import STRtree, area, buffer, intersection
    from shapely.geometry import Point

    # print(gpd.show_versions())

    points = [Point(x, y) for x, y in [(0, 0), (1, 0), (2, 0)]]
    circles = buffer(points, 1.2)

    print(area(circles))

    tree = STRtree(circles)
    left, right = tree.query(circles, predicate="intersects")

    intersections = intersection(circles[left], circles[right])
    print(area(intersections))

    tree = STRtree(intersections)
    left, right = tree.query(intersections, predicate="within")
    print(len(left))
    print(len(right))
    print(left)
    print(right)

    sg.to_gdf(intersections).explore()
    sss
    import geopandas as gpd

    print(gpd.show_versions())

    circles = sg.to_gdf([(0, 0), (1, 0), (2, 0)]).pipe(sg.buff, 1.2)
    gdf = sg.get_intersections(circles)
    gdf = gdf.reset_index(drop=True)
    assert len(gdf) == 6
    gdf.to_file(r"c:/users/ort/downloads/linux_windows.gpkg")
    joined = gdf.sjoin(gdf, predicate="within")
    print(joined)
    print(len(joined))
    assert len(joined) == 12
    assert list(sorted(joined.index.unique())) == [0, 1, 2, 3, 4, 5]

    import networkx as nx
    from shapely import STRtree

    tree = STRtree(gdf.geometry.values)
    left, right = tree.query(gdf.geometry.values, predicate="within")
    print(left)
    print(len(left))
    print(right)
    print(len(right))

    edges = list(zip(left, right))
    print(edges)

    graph = nx.Graph()
    graph.add_edges_from(edges)

    component_mapper = {
        j: i
        for i, component in enumerate(nx.connected_components(graph))
        for j in component
    }

    gdf["duplicate_group"] = component_mapper

    gdf["duplicate_group"] = gdf["duplicate_group"].astype(str)

    print(gdf)
    sg.explore(gdf, "duplicate_group")


def not_test_bug2():
    import geopandas as gpd
    from shapely import STRtree

    circles = sg.to_gdf([(0, 0), (1, 0), (2, 0)]).pipe(sg.buff, 1.2)
    gdf = sg.get_intersections(circles)
    print([x.wkt for x in gdf.geometry])
    gdf = gdf.reset_index(drop=True)
    assert len(gdf) == 6
    # gdf.to_file(r"c:/users/ort/downloads/linux_windows.gpkg")

    # gdf = gpd.read_file(r"c:/users/ort/downloads/linux_windows.gpkg")
    tree = STRtree(gdf.geometry.values)
    left, right = tree.query(gdf.geometry.values, predicate="within")
    print(left)
    print(len(left))
    print(right)
    print(len(right))


def not_test_drop_duplicate_geometries():
    circles = sg.to_gdf([(0, 0), (1, 0), (2, 0)]).pipe(sg.buff, 1.2)
    dups = sg.get_intersections(circles)
    assert len(dups) == 6
    # 3 unique intersections
    no_dups = sg.drop_duplicate_geometries(dups)
    assert len(no_dups) == 3, len(no_dups)

    # 3 pairs == 6 rows
    dups2 = sg.get_intersections(no_dups)
    assert len(dups2) == 6, len(dups2)
    no_dups2 = sg.drop_duplicate_geometries(dups2)
    assert len(no_dups2) == 1, len(no_dups2)

    dups3 = sg.drop_duplicate_geometries(no_dups2)
    sg.explore(no_dups2, dups3)
    assert len(dups3) == 1, len(dups3)

    no_dups3 = sg.get_intersections(dups3)
    assert len(no_dups3) == 0, len(no_dups3)


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
    not_test_bug2()
    test_get_intersections()
    test_update_geometries()
    test_random_get_intersections()
    not_test_drop_duplicate_geometries()
