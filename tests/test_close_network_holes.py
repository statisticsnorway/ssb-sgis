# %%
import sys
import warnings
from pathlib import Path

from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from shapely.wkt import loads

src = str(Path(__file__).parent).replace("tests", "") + "src"


sys.path.insert(0, src)
import sgis as sg


def test_very_small_network_hole():
    df = sg.to_gdf(
        [
            "LINESTRING Z (491452.45539342123 7598608.108440004 4.364247304392665, 491393.9000000004 7598596.809999999 4.26600000000326, 491393.556966218 7598596.748927162 4.26920592320249)",
            "LINESTRING (491420.4326655079 7598572.554228766, 491420.73000000045 7598581.239999998, 491421.1699999999 7598586.949999999, 491421.3700000001 7598591.59, 491421.54000000004 7598595.870000001, 491422.36000000034 7598599.48, 491423.0099999998 7598602.43)",
            "LINESTRING Z (491452.45420054055 7598608.116481753 4.49720152812222, 491393.9000000004 7598596.809999999 4.399999999994179, 491393.556966218 7598596.748927162 4.403205923193643)",
        ],
        25833,
    )
    print(df)
    points_close_to_deadends = sg.get_k_nearest_points_for_deadends(
        df, k=5, max_distance=15
    )
    sg.explore(points_close_to_deadends)
    closed = sg.close_network_holes(df, 15, max_angle=120)
    assert closed.length.sum() > df.length.sum() + 0.01


def test_failing_line_along_road():
    df = sg.to_gdf(
        [
            "LINESTRING (292939.5983805864 6926480.755566795, 292932.9000000004 6926483.1000000015, 292922.7999999998 6926487.699999999, 292917.7999999998 6926490.1000000015, 292913.7000000002 6926493.6000000015, 292911.2000000002 6926496.300000001, 292909 6926499.6000000015, 292907.5 6926503.199999999, 292906.5999999996 6926507.5, 292906.2999999998 6926510.8999999985, 292906.4000000004 6926514.199999999, 292907 6926516.800000001, 292907.7999999998 6926519.5, 292909.7999999998 6926523.300000001, 292913 6926528, 292918.4000000004 6926534.3999999985, 292921.36508169596 6926537.7886647945)",
            "LINESTRING (292875.5138994143 6926527.857438979, 292875.9287999999 6926527.513599999, 292879.57469999976 6926524.370999999, 292887.01680000033 6926517.649799999, 292895.40079999994 6926510.0370000005, 292903.03639999963 6926503.559, 292905.85979999974 6926502.050299998)",
        ],
        25833,
    )
    closest_node = sg.get_k_nearest_points_for_deadends(df, k=1, max_distance=3)
    sg.explore(closest_node, df)
    assert len(closest_node) == 1, closest_node
    assert int(df.length.sum()) == 120, df.length.sum()
    df = sg.split_lines_by_nearest_point(df, closest_node)
    assert int(df.length.sum()) == 120, df.length.sum()
    closed = sg.close_network_holes(df, 3, max_angle=130)
    assert int(closed.length.sum()) == 122, closed.length.sum()


def test_line_angle_0():
    lines_angle_0 = sg.to_gdf(
        MultiLineString(
            [
                LineString([(-1, 0), (0, 0), (1, 0)]),
                LineString([(2, 0), (3, 0), (4, 0)]),
            ]
        ),
        crs=25833,
    )
    should_work = sg.close_network_holes(lines_angle_0, max_distance=1, max_angle=90)
    if __name__ == "__main__":
        lines_angle_0.plot()
        should_work.plot("hole")
    assert len(should_work) == 4, len(should_work)

    directed_should_give_same = sg.close_network_holes(
        lines_angle_0,
        max_distance=1,
        max_angle=90,
    )
    if __name__ == "__main__":
        lines_angle_0.plot()
        directed_should_give_same.plot("hole")
    assert len(directed_should_give_same) == 4, len(directed_should_give_same)

    cannot_reach = sg.close_network_holes(lines_angle_0, max_distance=0.5, max_angle=10)
    assert len(cannot_reach) == 2, len(cannot_reach)

    can_reach = sg.close_network_holes(lines_angle_0, max_distance=1, max_angle=10)
    assert len(can_reach) == 4, len(can_reach)

    angle_0_should_be_fine = sg.close_network_holes(
        lines_angle_0, max_distance=1, max_angle=0
    )
    assert len(angle_0_should_be_fine) == 4, len(angle_0_should_be_fine)


def test_line_angle_90():
    lines_angle_90 = sg.to_gdf(
        MultiLineString([LineString([(0, 0), (1, 0)]), LineString([(1, 1), (1, 2)])]),
        crs=25833,
    )

    nw = sg.close_network_holes(lines_angle_90, max_distance=1, max_angle=180)
    if __name__ == "__main__":
        lines_angle_90.plot()
        nw.plot("hole")

    assert len(nw) == 4, len(nw)

    # when 180 degrees allowed and 2 units, it should create duplicate lines on top of
    # existing ones
    nw = sg.close_network_holes(lines_angle_90, max_distance=2, max_angle=180)
    assert len(nw) == 6, len(nw)

    nw = sg.close_network_holes(lines_angle_90, max_distance=1, max_angle=45)
    if __name__ == "__main__":
        nw.plot("hole")
    assert len(nw) == 3, len(nw)

    lines_angle_90_both = sg.to_gdf(
        MultiLineString([LineString([(0, 0), (1, 0)]), LineString([(1, 1), (2, 1)])]),
        crs=25833,
    )

    nw = sg.close_network_holes(lines_angle_90_both, max_distance=1, max_angle=45)
    if __name__ == "__main__":
        lines_angle_90_both.plot()
        nw.plot("hole")
    assert len(nw) == 2, len(nw)

    nw = sg.close_network_holes(lines_angle_90_both, max_distance=1, max_angle=90)
    if __name__ == "__main__":
        nw.plot("hole")
    assert len(nw) == 4, len(nw)

    lines_angle_90_other_side = sg.to_gdf(
        MultiLineString(
            [LineString([(0, 0), (-1, 0)]), LineString([(-1, -1), (-2, -1)])]
        ),
        crs=25833,
    )

    nw = sg.close_network_holes(lines_angle_90_other_side, max_distance=1, max_angle=45)
    if __name__ == "__main__":
        lines_angle_90_other_side.plot()
        nw.plot("hole")
    assert len(nw) == 2, len(nw)

    nw = sg.close_network_holes(lines_angle_90_other_side, max_distance=1, max_angle=90)
    if __name__ == "__main__":
        nw.plot("hole")
    assert len(nw) == 4, len(nw)

    should_fill_diagonals = sg.close_network_holes(
        lines_angle_90_other_side, max_distance=2, max_angle=160
    )
    if __name__ == "__main__":
        should_fill_diagonals.plot("hole")
    assert len(should_fill_diagonals) == 6, len(should_fill_diagonals)


def test_line_angle_45():
    lines_angle_45 = sg.to_gdf(
        MultiLineString(
            [LineString([(0, 0), (1, 0)]), LineString([(-1, -1), (-2, -1)])]
        ),
        crs=25833,
    ).explode(index_parts=False)

    should_not_reach = sg.close_network_holes(
        lines_angle_45, max_distance=1, max_angle=45
    )
    if __name__ == "__main__":
        lines_angle_45.plot()
        should_not_reach.plot("hole")
    assert len(should_not_reach) == len(lines_angle_45)

    should_reach = sg.close_network_holes(lines_angle_45, max_distance=2, max_angle=45)
    if __name__ == "__main__":
        should_reach.plot("hole")
    assert len(should_reach) == 4, len(should_reach)

    not_small_enough_angle = sg.close_network_holes(
        lines_angle_45, max_distance=2, max_angle=20
    )
    if __name__ == "__main__":
        not_small_enough_angle.plot("hole")
    assert len(not_small_enough_angle) == 2, len(not_small_enough_angle)

    lines_angle_45_opposite = sg.to_gdf(
        MultiLineString([LineString([(0, 0), (-1, 0)]), LineString([(1, 1), (2, 1)])]),
        crs=25833,
    )

    should_not_reach = sg.close_network_holes(
        lines_angle_45_opposite, max_distance=1, max_angle=45
    )
    if __name__ == "__main__":
        lines_angle_45_opposite.plot()
        should_not_reach.plot("hole")
    assert len(should_not_reach) == 2, len(should_not_reach)

    should_reach = sg.close_network_holes(
        lines_angle_45_opposite, max_distance=2, max_angle=45
    )
    if __name__ == "__main__":
        should_reach.plot("hole")
    assert len(should_reach) == 4, len(should_reach)

    not_small_enough_angle = sg.close_network_holes(
        lines_angle_45_opposite, max_distance=2, max_angle=20
    )
    if __name__ == "__main__":
        not_small_enough_angle.plot("hole")
    assert len(not_small_enough_angle) == 2, len(not_small_enough_angle)


def test_close_network_holes(roads_oslo, points_oslo):
    warnings.filterwarnings(action="ignore", category=UserWarning)
    warnings.filterwarnings(action="ignore", category=FutureWarning)

    r = roads_oslo
    p = points_oslo

    p = p.iloc[[0]]

    r = sg.clean_clip(r, p.buffer(600))

    nw = sg.get_connected_components(r)

    if __name__ == "__main__":
        sg.qtm(nw, "connected", title="before filling holes")

    assert sum(nw.connected == 1) == 650, sum(nw.connected == 1)
    assert sum(nw.connected == 0) == 104, sum(nw.connected == 0)

    nw = sg.close_network_holes_to_deadends(r.copy(), max_distance=1.1)
    assert sum(nw.hole == 1) == 68, sum(nw.hole == 1)

    nw = sg.close_network_holes(r.copy(), max_distance=1.1, max_angle=0)
    assert sum(nw.hole == 1) == 0, sum(nw.hole == 1)

    nw = sg.close_network_holes(r.copy(), max_distance=1.1, max_angle=10)
    assert sum(nw.hole == 1) == 57, sum(nw.hole == 1)

    nw = sg.close_network_holes(r.copy(), max_distance=1.1, max_angle=90)
    assert sum(nw.hole == 1) == 67, sum(nw.hole == 1)

    nw2 = sg.close_network_holes(r.copy(), max_distance=1.1, max_angle=180)
    assert sum(nw2.hole == 1) == 68, sum(nw2.hole == 1)

    nw = sg.close_network_holes_to_deadends(r.copy(), max_distance=10)
    assert sum(nw.hole == 1) == 95, sum(nw.hole == 1)

    nw = sg.close_network_holes(r.copy(), max_distance=10, max_angle=90)
    sg.explore(nw.assign(hole=nw["hole"].astype(str)), "hole")
    nw = sg.get_connected_components(nw)
    sg.explore(nw, "connected")
    # assert sum(nw.hole == 1) == 93, sum(nw.hole == 1)

    # nodes = r.extract_unique_points().explode().to_frame()
    # # nw = sg.close_network_holes(r.copy(), max_distance=10.33, max_angle=90)
    # nw["length"] = nw.length
    # sg.explore(nw, nodes)

    nw = sg.get_connected_components(nw)

    if __name__ == "__main__":
        sg.qtm(nw, "connected", title="after filling holes")
        sg.explore(nw, "connected")

    assert sum(nw["connected"] == 1) == 827, sum(nw["connected"] == 1)
    assert sum(nw["connected"] == 0) == 20, sum(nw["connected"] == 0)


def test_sharp_angle():
    lines = sg.to_gdf(
        [
            LineString(
                [
                    loads("POINT (706599.21 7706862.550000001)"),
                    loads("POINT (706623.8099999996 7706901.710000001)"),
                ]
            ),
            LineString(
                [
                    loads("POINT (706623.8099999996 7706901.710000001)"),
                    loads("POINT (706625.2400000002 7706902.690000001)"),
                ]
            ),
            LineString(
                [
                    loads("POINT (706625.2400000002 7706902.690000001)"),
                    loads("POINT (706626.4199999999 7706903.1499999985)"),
                ]
            ),
            LineString(
                [
                    loads("POINT (706626.4199999999 7706903.1499999985)"),
                    loads("POINT (706627.7000000002 7706903.73)"),
                ]
            ),
        ],
        crs=25833,
    )

    deadends_closed = sg.close_network_holes_to_deadends(lines, 15)
    all_closed = sg.close_network_holes(lines, 15, max_angle=120)
    assert len(lines) == len(deadends_closed) == len(all_closed)


def main():
    from oslo import points_oslo
    from oslo import roads_oslo

    test_very_small_network_hole()

    test_failing_line_along_road()

    test_line_angle_0()
    test_line_angle_90()
    test_line_angle_45()
    test_sharp_angle()
    test_close_network_holes(roads_oslo(), points_oslo())


if __name__ == "__main__":
    main()
