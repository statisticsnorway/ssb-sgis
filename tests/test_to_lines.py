import sys
from pathlib import Path

import pytest
from shapely.geometry import LineString
from shapely.geometry import Polygon

src = str(Path(__file__).parent.parent) + "/src"

sys.path.insert(0, src)

import sgis as sg


def test_polygons_with_interiors():
    # from geopandas import *
    # import shapely
    # from shapely.geometry import *
    # import numpy as np
    # from sgis import *

    # sg.to_lines = to_lines
    # sg.polygons_to_lines = polygons_to_lines

    poly1 = sg.to_gdf(Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]))

    inner_poly = sg.to_gdf(
        (
            Polygon(
                [(0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25)],
            )
        )
        .difference(
            Polygon(
                [(0.35, 0.35), (0.4, 0.35), (0.4, 0.4), (0.35, 0.4)],
            )
        )
        .difference(
            Polygon(
                [(0.55, 0.55), (0.45, 0.55), (0.45, 0.45), (0.55, 0.45)],
            )
        )
    )

    poly2 = sg.to_gdf(Polygon([(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)]))
    poly1["x"] = 1
    inner_poly["x"] = 2
    poly2["x"] = 3
    poly1["y"] = 4

    # import itertools
    import pandas as pd

    samla = pd.concat([poly1, poly2, inner_poly]).pipe(
        sg.geopandas_tools.general.polygons_to_lines
    )

    res = sg.to_lines(poly1, poly2, inner_poly)
    sg.qtm(res.assign(idx=[str(i) for i in range(len(res))]), "idx")
    sg.explore(res.assign(idx=[str(i) for i in range(len(res))]), "idx")
    sg.explore(samla.assign(idx=[str(i) for i in range(len(samla))]), "idx")
    assert len(res) == 17, len(res)


def test_to_lines():
    poly1 = sg.to_gdf(Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]))

    as_lines = sg.to_lines(poly1)
    print(poly1)
    print(as_lines)
    assert int(poly1.length.sum()) == int(as_lines.length.sum()), (
        int(poly1.length.sum()),
        int(as_lines.length.sum()),
    )

    inner_poly = sg.to_gdf(
        Polygon([(0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25)])
    )

    poly1_diff = poly1.overlay(inner_poly, how="difference")

    poly2 = sg.to_gdf(Polygon([(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)]))

    line = sg.to_gdf(LineString([(0.5, 0.5), (0.5, 1.5)]))
    lines = sg.to_lines(poly1_diff)
    lines["l"] = lines.length.astype(str)
    print(lines)

    if __name__ == "__main__":
        sg.qtm(lines, "l", title=len(lines))

    if __name__ == "__main__":
        sg.qtm(poly1, poly2, inner_poly)

    with pytest.raises(ValueError):
        sg.to_lines(line, poly1, inner_poly, poly1_diff, sg.to_gdf([0, 0]))

    # lines = sg.to_lines(line, poly1, inner_poly, poly1_diff)

    # lines["l"] = lines.length.astype(str)
    # print(lines)

    # if __name__ == "__main__":
    #     sg.qtm(lines, "l", title=len(lines))

    lines = sg.to_lines(poly1, poly2, inner_poly)
    lines["l"] = lines.length.astype(str)
    print(lines)
    if __name__ == "__main__":
        sg.qtm(lines, "l", title=len(lines))

    lines = sg.to_lines(poly1, poly2)
    lines["l"] = lines.length.astype(str)
    print(lines)
    if __name__ == "__main__":
        sg.qtm(lines, "l", title=len(lines))

    lines = sg.to_lines(poly1, poly2, inner_poly, poly2)
    lines["l"] = lines.length.astype(str)
    print(lines)
    if __name__ == "__main__":
        sg.qtm(lines, "l", title=len(lines))

    lines = sg.to_lines(lines)
    lines["l"] = lines.length.astype(str)
    print(lines)
    if __name__ == "__main__":
        sg.qtm(lines, "l", title=len(lines))

    # empty
    poly1["col"] = 1
    poly2["col"] = 1
    empty1 = poly1[poly1["col"] != 1]
    empty2 = poly2[poly2["col"] != 1]
    lines = sg.to_lines(empty1, empty2, empty2, empty1)
    assert lines.shape == (0, 2)
    assert list(sorted(lines.columns)) == ["col", "geometry"]


def main():
    test_polygons_with_interiors()
    test_to_lines()


if __name__ == "__main__":
    main()

# %%
