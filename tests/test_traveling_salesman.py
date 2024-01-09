# %%
import sys
from pathlib import Path

from shapely.geometry import LineString, Point


src = str(Path(__file__).parent.parent) + "/src"

sys.path.insert(0, src)

import sgis as sg


def test_traveling_salesman():
    points = sg.to_gdf(
        [
            (0, 0),
            (10, -10),
            (10, 10),
            (0, 10),
            (0, -10),
            (10, 0),
            (20, 0),
            (0, 20),
        ]
    )
    oneway_should_equal = [
        Point(0, 20),
        Point(0, 10),
        Point(10, 10),
        Point(0, 0),
        Point(10, 0),
        Point(20, 0),
        Point(10, -10),
        Point(0, -10),
    ]
    roundtrip_should_equal = [
        Point(0, 0),
        Point(10, -10),
        Point(0, -10),
        Point(10, 0),
        Point(20, 0),
        Point(10, 10),
        Point(0, 10),
        Point(0, 20),
        Point(0, 0),
    ]
    roundtrip = sg.traveling_salesman_problem(points, return_to_start=True)
    oneway = sg.traveling_salesman_problem(points, return_to_start=False)

    print(oneway)
    print(LineString(oneway))
    print(roundtrip)
    print(LineString(roundtrip))

    print(LineString(roundtrip).length)
    print(LineString(oneway).length)
    sg.qtm(points, sg.to_gdf(LineString(roundtrip)))
    sg.qtm(points, sg.to_gdf(LineString(oneway)))
    assert oneway == oneway_should_equal, oneway
    assert roundtrip == roundtrip_should_equal, roundtrip

    points.index = [3, 8, 1, 0, 12, 20, 15, 32]
    distances = sg.get_all_distances(points, points)
    roundtrip = sg.traveling_salesman_problem(
        points, return_to_start=True, distances=distances
    )
    oneway = sg.traveling_salesman_problem(
        points, return_to_start=False, distances=distances
    )

    print(LineString(roundtrip).length)
    print(LineString(oneway).length)
    sg.qtm(points, sg.to_gdf(LineString(roundtrip)))
    sg.qtm(points, sg.to_gdf(LineString(oneway)))

    assert oneway == oneway_should_equal, oneway
    assert roundtrip == roundtrip_should_equal, roundtrip


if __name__ == "__main__":
    test_traveling_salesman()
