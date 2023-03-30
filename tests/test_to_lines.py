# %%
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Polygon


src = str(Path(__file__).parent.parent) + "/src"

sys.path.insert(0, src)

import sgis as sg


def test_to_lines():
    poly1 = sg.to_gdf(Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]))

    inner_poly = sg.to_gdf(
        Polygon([(0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25)])
    )

    poly1_diff = poly1.overlay(inner_poly, how="difference")

    line = sg.to_gdf(LineString([(0.5, 0.5), (0.5, 1.5)]))
    lines = sg.to_lines(poly1_diff)
    lines["l"] = lines.length.astype(str)
    print(lines)

    if __name__ == "__main__":
        sg.qtm(lines, "l", legend_title=len(lines))

    poly2 = sg.to_gdf(Polygon([(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)]))

    if __name__ == "__main__":
        sg.qtm(poly1, poly2, inner_poly)

    lines = sg.to_lines(line, poly1, inner_poly, poly1_diff)

    lines["l"] = lines.length.astype(str)
    print(lines)

    if __name__ == "__main__":
        sg.qtm(lines, "l", legend_title=len(lines))

    lines = sg.to_lines(poly1, poly2, inner_poly)
    lines["l"] = lines.length.astype(str)
    print(lines)
    if __name__ == "__main__":
        sg.qtm(lines, "l", legend_title=len(lines))

    lines = sg.to_lines(poly1, poly2)
    lines["l"] = lines.length.astype(str)
    print(lines)
    if __name__ == "__main__":
        sg.qtm(lines, "l", legend_title=len(lines))

    lines = sg.to_lines(poly1, poly2, inner_poly, poly2)
    lines["l"] = lines.length.astype(str)
    print(lines)
    if __name__ == "__main__":
        sg.qtm(lines, "l", legend_title=len(lines))

    lines = sg.to_lines(lines)
    lines["l"] = lines.length.astype(str)
    print(lines)
    if __name__ == "__main__":
        sg.qtm(lines, "l", legend_title=len(lines))

    # TODO: MIXED


def main():
    test_to_lines()


if __name__ == "__main__":
    main()
