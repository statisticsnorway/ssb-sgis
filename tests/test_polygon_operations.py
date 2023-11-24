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


def test_polygonsasrings():
    p = sg.to_gdf([0, 0])

    buff1 = sg.buffdissexp(p, 100)

    no_holes_closed = sg.close_all_holes(buff1)
    assert round(np.sum(no_holes_closed.area), 3) == round(np.sum(buff1.area), 3), (
        no_holes_closed,
        buff1,
    )
    no_holes_closed = sg.close_small_holes(buff1, max_area=1_000_000)
    assert round(np.sum(no_holes_closed.area), 3) == round(np.sum(buff1.area), 3)

    buff2 = sg.buffdissexp(p, 200)
    ring_with_hole = sg.clean_overlay(buff2, buff1, how="difference")

    buff0 = sg.buffdissexp(p, 30)
    ring_with_hole_and_island = pd.concat([ring_with_hole, buff0])

    p2 = sg.to_gdf([150, 0]).buffer(10).to_frame()
    two_holes = sg.clean_overlay(ring_with_hole_and_island, p2, how="difference")

    rings = sg.PolygonsAsRings(two_holes).get_rings()
    assert int(rings.length.sum()) == 2136, rings.length.sum()
    assert isinstance(rings, gpd.GeoDataFrame), type(rings)

    rings = sg.PolygonsAsRings(two_holes.geometry).get_rings()
    assert int(rings.length.sum()) == 2136, rings.length.sum()
    assert isinstance(rings, gpd.GeoSeries), type(rings)

    rings = sg.PolygonsAsRings(two_holes.geometry.values).get_rings()
    assert int(rings.length.sum()) == 2136, rings.length.sum()
    assert isinstance(rings, gpd.array.GeometryArray), type(rings)


def test_close_holes():
    p = sg.to_gdf([0, 0])

    buff1 = sg.buffdissexp(p, 100)

    no_holes_closed = sg.close_all_holes(buff1)
    assert round(np.sum(no_holes_closed.area), 3) == round(np.sum(buff1.area), 3), (
        no_holes_closed,
        buff1,
    )
    no_holes_closed = sg.close_small_holes(buff1, max_area=1_000_000)
    assert round(np.sum(no_holes_closed.area), 3) == round(np.sum(buff1.area), 3)

    buff2 = sg.buffdissexp(p, 200)
    ring_with_hole = sg.clean_overlay(buff2, buff1, how="difference")

    buff0 = sg.buffdissexp(p, 30)
    ring_with_hole_and_island = pd.concat([ring_with_hole, buff0])

    p2 = sg.to_gdf([150, 0]).buffer(10).to_frame()
    two_holes = sg.clean_overlay(ring_with_hole_and_island, p2, how="difference")

    assert len(sg.get_holes(buff1)) == 0
    assert len(sg.get_holes(ring_with_hole)) == 1
    assert len(sg.get_holes(two_holes)) == 2

    # run this for different geometry input types
    def _close_the_holes(ring_with_hole):
        all_closed = sg.close_all_holes(ring_with_hole)

        assert sum(all_closed.area) > sum(ring_with_hole.area)
        all_closed2 = sg.close_all_holes(two_holes)
        assert sum(all_closed.area) == sum(all_closed.area)

        # this should return the entire hole
        closed_island_ignored = sg.close_all_holes(ring_with_hole, ignore_islands=True)
        assert sum(closed_island_ignored.area) > sum(all_closed.area)

        hole_not_closed = sg.close_small_holes(ring_with_hole, max_area=1)
        assert sum(all_closed.area) > sum(hole_not_closed.area)

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
        ignore_islands = True
        all_closed3 = sg.close_small_holes(
            ring_with_hole, max_area=32_000, ignore_islands=ignore_islands
        )
        print(type(all_closed3))

        assert round(np.sum(all_closed3.area), 3) > round(np.sum(all_closed.area), 3)

        hole_not_closed3 = sg.close_small_holes(
            ring_with_hole, max_area=30_000, ignore_islands=ignore_islands
        )
        assert np.sum(hole_not_closed3.area) == np.sum(ring_with_hole.area), (
            hole_not_closed3,
            ring_with_hole,
        )

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

    c = sg.get_polygon_clusters(gdf, as_string=True)
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

    polys = pd.concat([small_poly, large_poly], ignore_index=True)

    polys.index = [5, 7]

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
    if __name__ == "__main__":
        sg.qtm(eliminated, "num", title="", alpha=0.8)

    assert list(eliminated.num) == [3, 3], list(eliminated.num)
    assert list(sorted(eliminated.columns)) == ["geometry", "num", "what"], list(
        sorted(eliminated.columns)
    )

    eliminated = sg.eliminate_by_largest(polys, sliver)
    if __name__ == "__main__":
        sg.qtm(eliminated, "what", title="after eliminate_by_largest", alpha=0.8)
    assert list(eliminated.index) == [5, 7], list(eliminated.index)
    assert list(eliminated.num) == [2, 3], list(eliminated.num)
    assert list(eliminated.what) == ["small", "large"], list(eliminated.what)
    assert list(round(eliminated.area, 1)) == [1.9, 5.6], list(eliminated.area)
    assert list(sorted(eliminated.columns)) == ["geometry", "num", "what"], list(
        sorted(eliminated.columns)
    )

    eliminated = sg.eliminate_by_largest(
        polys, sliver, aggfunc={"num": "sum", "what": "first"}
    )
    assert list(eliminated.num) == [2, 4], list(eliminated.num)
    assert list(sorted(eliminated.columns)) == ["geometry", "num", "what"], list(
        sorted(eliminated.columns)
    )
    eliminated = sg.eliminate_by_smallest(
        polys, sliver, aggfunc={"num": "sum", "what": "first"}
    )
    if __name__ == "__main__":
        sg.qtm(eliminated, "what", title="after eliminate_by_smallest", alpha=0.8)
    assert list(eliminated.index) == [5, 7], list(eliminated.index)
    assert list(eliminated.num) == [3, 3], list(eliminated.num)
    assert list(eliminated.what) == ["small", "large"], list(eliminated.what)
    assert list(round(eliminated.area, 1)) == [2.1, 5.4], list(eliminated.area)
    assert list(sorted(eliminated.columns)) == ["geometry", "num", "what"], list(
        sorted(eliminated.columns)
    )
    missing_value = polys.assign(what=pd.NA)
    eliminated = sg.eliminate_by_smallest(missing_value, sliver)
    assert eliminated["what"].isna().all()

    eliminated = sg.eliminate_by_longest(polys, isolated)

    if __name__ == "__main__":
        sg.qtm(eliminated, "what", title="with isolated", alpha=0.8)
    assert list(eliminated.index) == [5, 7, 0], list(eliminated.index)
    assert list(eliminated.what) == ["small", "large", "isolated"], list(
        eliminated.what
    )

    eliminated = sg.eliminate_by_largest(polys, isolated)
    assert list(eliminated.what) == ["small", "large", "isolated"], list(
        eliminated.what
    )
    assert list(eliminated.index) == [5, 7, 0], list(eliminated.index)

    eliminated = sg.eliminate_by_longest(polys, sg.buff(sliver, 0.1), fix_double=True)
    double = sg.get_intersections(eliminated)
    if __name__ == "__main__":
        sg.qtm(
            eliminated, double, "what", title="with buffer and fix double", alpha=0.5
        )

    assert double.area.sum() < 1e-10, double

    eliminated = sg.eliminate_by_largest(polys, sg.buff(sliver, 0.1), fix_double=True)
    double = sg.get_intersections(eliminated)
    if __name__ == "__main__":
        sg.qtm(
            eliminated, double, "what", title="with buffer and fix double", alpha=0.5
        )
    assert list(sorted(eliminated.columns)) == ["geometry", "num", "what"], list(
        sorted(eliminated.columns)
    )
    assert double.area.sum() < 1e-10, double


if __name__ == "__main__":
    test_eliminate()
    test_polygonsasrings()

    test_close_holes()
    test_get_polygon_clusters()
