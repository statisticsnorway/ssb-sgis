# %%
import sys
import warnings
from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString


src = str(Path(__file__).parent).strip("tests") + "src"


sys.path.insert(0, src)
import sgis as sg


def test_close_network_holes(roads_oslo, points_oslo):
    lines_angle_0 = sg.to_gdf(
        MultiLineString(
            [
                LineString([(-1, 0), (0, 0), (1, 0)]),
                LineString([(2, 0), (3, 0), (4, 0)]),
            ]
        ),
        crs=25833,
    )
    nw = sg.Network(lines_angle_0).close_network_holes(1, fillna=0, max_angle=90)
    if __name__ == "__main__":
        lines_angle_0.plot()
        nw.gdf.plot("hole")
    # assert len(nw) == 4, len(nw)

    nw = sg.DirectedNetwork(lines_angle_0).close_network_holes(
        1,
        max_angle=90,
        fillna=0,
    )
    if __name__ == "__main__":
        lines_angle_0.plot()
        nw.gdf.plot("hole")
    # assert len(nw) == 4, len(nw)

    nw = sg.Network(lines_angle_0).close_network_holes(1, fillna=0, max_angle=10)
    # assert len(nw) == 4, len(nw)
    nw = sg.Network(lines_angle_0).close_network_holes(1, fillna=0, max_angle=0)
    # assert len(nw) == 4, len(nw)

    lines_angle_90 = sg.to_gdf(
        MultiLineString([LineString([(0, 0), (1, 0)]), LineString([(1, 1), (1, 2)])]),
        crs=25833,
    )
    lines_angle_90.plot(cmap="plasma")

    nw.gdf.plot("hole", cmap="plasma")
    nw = sg.Network(lines_angle_90).close_network_holes(1, max_angle=180, fillna=0)
    if __name__ == "__main__":
        lines_angle_90.plot()
        nw.gdf.plot("hole")
    # assert len(nw) == 4, len(nw)

    nw = sg.Network(lines_angle_90).close_network_holes(1, fillna=0, max_angle=45)
    if __name__ == "__main__":
        nw.gdf.plot("hole")
    # assert len(nw) == 3, len(nw)

    lines_angle_90_both = sg.to_gdf(
        MultiLineString([LineString([(0, 0), (1, 0)]), LineString([(1, 1), (2, 1)])]),
        crs=25833,
    )

    nw = sg.Network(lines_angle_90_both).close_network_holes(1, fillna=0, max_angle=45)
    if __name__ == "__main__":
        lines_angle_90_both.plot()
        nw.gdf.plot("hole")
    # assert len(nw) == 2, len(nw)

    nw = sg.Network(lines_angle_90_both).close_network_holes(1, fillna=0, max_angle=90)
    if __name__ == "__main__":
        lines_angle_90_both.plot()
        nw.gdf.plot("hole")
    # assert len(nw) == 4, len(nw)

    warnings.filterwarnings(action="ignore", category=UserWarning)
    warnings.filterwarnings(action="ignore", category=FutureWarning)

    r = roads_oslo
    p = points_oslo

    p = p.iloc[[0]]

    r = sg.clean_clip(r, p.buffer(600))

    nw = sg.Network(r)

    nw = nw.get_largest_component()

    if __name__ == "__main__":
        sg.qtm(nw.gdf, "connected")

    # assert sum(nw.gdf.connected == 1) == 650
    # assert sum(nw.gdf.connected == 0) == 104

    nw = nw.close_network_holes_to_deadends(1.1, fillna=0)
    print("n", sum(nw.gdf.hole == 1))
    # assert sum(nw.gdf.hole == 1) == 68

    nw = nw.close_network_holes(1.1, max_angle=90, fillna=0)
    print("n", sum(nw.gdf.hole == 1))
    # assert sum(nw.gdf.hole == 1) == 68

    nw = nw.close_network_holes_to_deadends(10, fillna=0)
    print("n", sum(nw.gdf.hole == 1))
    # assert sum(nw.gdf.hole == 1) == 93

    nw = nw.close_network_holes(10, max_angle=90, fillna=0)
    print("n", sum(nw.gdf.hole == 1))
    # assert sum(nw.gdf.hole == 1) == 103

    nw = nw.get_largest_component()

    if __name__ == "__main__":
        sg.qtm(nw.gdf, "connected")

    # assert sum(nw.gdf.connected == 1) == 836, sum(nw.gdf.connected == 1)
    # assert sum(nw.gdf.connected == 0) == 21, sum(nw.gdf.connected == 0)


def main():
    from oslo import points_oslo, roads_oslo

    test_close_network_holes(roads_oslo(), points_oslo())


if __name__ == "__main__":
    main()
