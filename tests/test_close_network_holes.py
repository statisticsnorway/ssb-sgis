# %%
import sys
import warnings
from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString
from shapely.geometry import MultiLineString

src = str(Path(__file__).parent).strip("tests") + "src"


sys.path.insert(0, src)
import sgis as sg


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
    assert sum(nw.hole == 1) == 93, sum(nw.hole == 1)

    nw = sg.get_connected_components(nw)

    if __name__ == "__main__":
        sg.qtm(nw, "connected", title="after filling holes")

    assert sum(nw["connected"] == 1) == 827, sum(nw["connected"] == 1)
    assert sum(nw["connected"] == 0) == 20, sum(nw["connected"] == 0)


def main():
    from oslo import points_oslo
    from oslo import roads_oslo

    test_line_angle_0()
    test_line_angle_90()
    test_line_angle_45()
    test_close_network_holes(roads_oslo(), points_oslo())


if __name__ == "__main__":
    main()
