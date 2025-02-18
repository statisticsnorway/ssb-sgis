# %%

import sys
from pathlib import Path

src = str(Path(__file__).parent.parent) + "/src"

sys.path.insert(0, src)

import sgis as sg


def test_false_duplicate():
    df = sg.to_gdf(
        [
            "POLYGON ((-41791.03450000007 6560935.7707, -41801.31859999988 6560936.7687, -41803.17750000022 6560936.593900001, -41805.15990000032 6560937.0655000005, -41807.12509999983 6560937.604499999, -41809.07070000004 6560938.2103, -41810.99450000003 6560938.882200001, -41812.89429999981 6560939.6193, -41814.767799999565 6560940.4209, -41816.612800000235 6560941.2859000005, -41818.42729999963 6560942.213400001, -41820.2089999998 6560943.202299999, -41821.95590000041 6560944.2513999995, -41823.6660000002 6560945.3596, -41825.33729999978 6560946.5254999995, -41826.96779999975 6560947.7477, -41828.555599999614 6560949.024900001, -41830.09889999963 6560950.355599999, -41831.59590000007 6560951.7381, -41833.044800000265 6560953.171, -41834.444000000134 6560954.6524, -41917.51709999982 6561045.6811999995, -41949.2276999997 6561102.8188000005, -41963.58069999982 6561096.5593, -41972.38900000043 6561106.468599999, -41981.164400000125 6561116.4070999995, -41989.90689999983 6561126.374500001, -41998.61639999971 6561136.3708999995, -42083.50889999978 6561234.140900001, -42084.960400000215 6561235.868100001, -42086.35529999994 6561237.6413, -42087.691999999806 6561239.458799999, -42088.9693 6561241.318499999, -42090.185800000094 6561243.218599999, -42091.34009999968 6561245.1570999995, -42092.431099999696 6561247.131899999, -42093.457600000314 6561249.140900001, -42094.41860000044 6561251.1821, -42095.31309999991 6561253.2534, -42096.140100000426 6561255.352499999, -42096.89869999979 6561257.4772, -42097.588200000115 6561259.625399999, -42098.207799999975 6561261.7947, -42098.51649999991 6561262.9507, -42084.001099999994 6561302.503699999, -41966.81049999967 6561244.797900001, -41888.98190000001 6561208.751499999, -41887.840900000185 6561208.2256000005, -41877.511900000274 6561204.262399999, -41831.20930000022 6561179.600199999, -41796.11679999996 6561162.9793, -41751.73849999998 6561174.8303, -41749.38659999985 6561167.6283, -41743.05620000046 6561141.887499999, -41742.414599999785 6561128.6757, -41741.38239999954 6561119.125, -41739.34439999983 6561105.2031, -41737.132100000046 6561088.858200001, -41733.088000000454 6561071.270400001, -41730.83719999995 6561062.052999999, -41730.821700000204 6561062.0013, -41727.84179999959 6561039.0908, -41724.97750000004 6561024.0569, -41723.3180999998 6561012.525900001, -41721.37789999973 6561000.758099999, -41718.00459999964 6560977.630799999, -41715.903300000355 6560964.1866, -41717.54349999968 6560962.8641, -41763.933899999596 6560941.767000001, -41770.46829999983 6560939.2239, -41777.272800000384 6560937.259199999, -41784.08430000022 6560936.1108, -41791.03450000007 6560935.7707), (-41908.82129999995 6561158.1865, -41863.62200000044 6561132.123199999, -41846.09219999984 6561162.551000001, -41891.28239999991 6561188.603499999, -41908.82129999995 6561158.1865))",
            "POLYGON ((-41863.62200000044 6561132.123199999, -41908.82129999995 6561158.1865, -41891.28239999991 6561188.603499999, -41846.09219999984 6561162.551000001, -41863.62200000044 6561132.123199999))",
        ]
    )
    dups = sg.get_intersections(df)[lambda x: x.area > 1e-6]
    assert not len(dups), dups.area


def test_random_get_intersections():
    grid_size = 1e-4

    # many iterations to try to break the assertion
    for _ in range(100):
        circles = sg.random_points(15).set_crs(25833).buffer(0.1).to_frame()
        the_overlap = sg.get_intersections(circles)

        updated = sg.update_geometries(the_overlap, grid_size=grid_size)

        overlapping_now = sg.get_intersections(updated).loc[
            lambda x: x.area / x.length > grid_size
        ]
        assert not len(overlapping_now), (
            overlapping_now.assign(sliv=lambda x: x.area / x.length),
            sg.explore(overlapping_now),
        )


def not_test_bug():
    import geopandas as gpd
    import networkx as nx
    from shapely import STRtree
    from shapely import area
    from shapely import buffer
    from shapely import intersection
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

    from shapely import STRtree

    tree = STRtree(gdf.geometry.values)
    left, right = tree.query(gdf.geometry.values, predicate="within")
    print(left)
    print(len(left))
    print(right)
    print(len(right))

    edges = list(zip(left, right, strict=False))
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
    from shapely import STRtree

    circles = sg.to_gdf([(0, 0), (1, 0), (2, 0)]).pipe(sg.buff, 1.2)
    gdf = sg.get_intersections(circles)
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

    n_jobs = 2

    dups = (
        sg.get_intersections(circles, n_jobs=n_jobs)
        .pipe(sg.update_geometries)
        .pipe(sg.buff, -0.01)
        .pipe(sg.clean_geoms)
    )
    print(dups)
    sg.qtm(dups.pipe(sg.buff, -0.025), alpha=0.2, column="area")
    assert len(dups) == 5, len(dups)

    dups = (
        sg.get_intersections(circles, n_jobs=n_jobs)
        .pipe(sg.update_geometries, n_jobs=n_jobs)
        .pipe(sg.buff, -0.01)
        .pipe(sg.clean_geoms)
    )
    print(dups)
    sg.qtm(dups.pipe(sg.buff, -0.025), alpha=0.2, column="area")
    assert len(dups) == 5, len(dups)

    dups = sg.get_intersections(circles, n_jobs=n_jobs)
    print(dups)
    sg.qtm(dups.pipe(sg.buff, -0.025), alpha=0.2, column="area")
    assert len(dups) == 12, len(dups)


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


def test_random_update_geometries(n=100):
    for i in range(n):
        print(i, end="\r")
        circles = sg.random_points(n).buffer(0.05).to_frame("geometry")
        updated = sg.update_geometries(circles)
        duplicates = sg.get_intersections(updated).loc[
            lambda x: ~x.buffer(-1e-6).is_empty
        ]
        assert not len(duplicates), (
            sg.explore(duplicates, updated, circles),
            duplicates.area,
            duplicates.geometry,
        )


def test_update_geometries():
    coords = [(0, 0), (0, 1), (1, 1), (1, 0)]
    buffers = [0.9, 1.3, 0.7, 1.1]
    circles = sg.to_gdf(coords)
    circles["geometry"] = circles["geometry"].buffer(buffers)

    n_jobs = 2

    updated = sg.update_geometries(circles, n_jobs=n_jobs)
    area = list((updated.area * 10).astype(int))
    assert area == [25, 36, 4, 18], area

    updated_largest_first = sg.update_geometries(
        sg.sort_large_first(circles), n_jobs=n_jobs
    )
    area = list((updated_largest_first.area * 10).astype(int))
    assert area == [53, 24, 5, 2], area

    updated_smallest_first = sg.update_geometries(
        sg.sort_large_first(circles).iloc[::-1], n_jobs=n_jobs
    )
    area = updated_smallest_first.area * 10
    assert [int(x) for x in area] == [15, 24, 18, 0, 26], area

    sg.explore(circles, updated, updated_largest_first, updated_smallest_first)


if __name__ == "__main__":
    test_false_duplicate()
    test_get_intersections()
    test_random_get_intersections()
    test_update_geometries()
    test_random_update_geometries(200)
    not_test_bug2()
    not_test_drop_duplicate_geometries()
