# %%
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString


src = str(Path(__file__).parent).strip("tests") + "src"

sys.path.insert(0, src)

import sgis as sg


def speed_col_to_minutes():
    # 1 km, 60 hm/h. Should give 1 in the minute column
    line_1000m = sg.to_gdf(LineString([(0, 0), (0, 1000)]), crs=25833)
    line_1000m["oneway"] = "B"

    should_take_1_min = sg.make_directed_network(
        line_1000m.assign(speed=60),
        direction_col="oneway",
        direction_vals_bft=("B", "FT", "TF"),
        speed_col_kmh="speed",
    )

    assert (
        int(round(should_take_1_min.minutes.mean(), 0)) == 1
    ), should_take_1_min.minutes.mean()


def flat_speed_to_minutes():
    # 1 km, 60 hm/h. Should give 1 in the minute column
    line_1000m = sg.to_gdf(LineString([(0, 0), (0, 1000)]), crs=25833)
    line_1000m["oneway"] = "B"
    should_take_1_min = sg.make_directed_network(
        line_1000m,
        direction_col="oneway",
        direction_vals_bft=("B", "FT", "TF"),
        flat_speed_kmh=60,
    )
    assert (
        int(round(should_take_1_min.minutes.mean(), 0)) == 1
    ), should_take_1_min.minutes.mean()


def test_directed_network():
    flat_speed_to_minutes()
    speed_col_to_minutes()

    rules = sg.NetworkAnalysisRules(weight="minutes", directed=True)

    points = sg.to_gdf([(0, 0), (0, 1), (0, 2), (0, 3)], crs=25833)

    # creating a network with a length of 1 meter, a drivetime of 1
    lines = sg.to_gdf(
        [
            LineString([(0, 0), (0, 1)]),
            LineString([(0, 1), (0, 2)]),
            LineString([(0, 2), (0, 3)]),
        ],
        crs=25833,
    )
    lines["oneway"] = ["B", "FT", "TF"]
    lines["drivetime_fw"] = [1, 1, -1]
    lines["drivetime_bw"] = [1, -1, 1]

    # now to create the same directed lines in different ways

    directed_lines = sg.make_directed_network_norway(lines)

    _run_od_costs(directed_lines, rules, points)

    directed_lines = sg.make_directed_network(
        lines,
        direction_col="oneway",
        direction_vals_bft=("B", "FT", "TF"),
        flat_speed_kmh=1 / 16.666666666,
    )
    directed_lines["minutes"] = round(directed_lines.minutes, 0).astype(int)
    _run_od_costs(directed_lines, rules, points)

    directed_lines = sg.make_directed_network(
        lines.assign(minutes=1),
        direction_col="oneway",
        direction_vals_bft=("B", "FT", "TF"),
        minute_cols="minutes",
    )
    _run_od_costs(directed_lines, rules, points)

    directed_lines = sg.make_directed_network(
        lines.assign(minutes=1),
        direction_col="oneway",
        direction_vals_bft=("B", "FT", "TF"),
        minute_cols=["minutes", "minutes"],
    )
    _run_od_costs(directed_lines, rules, points)

    directed_lines = sg.make_directed_network(
        lines,
        direction_col="oneway",
        direction_vals_bft=("B", "FT", "TF"),
        flat_speed_kmh=1 / 16.666666666,
    )
    directed_lines["minutes"] = round(directed_lines.minutes, 0).astype(int)
    _run_od_costs(directed_lines, rules, points)

    directed_lines = sg.make_directed_network(
        lines,
        direction_col="oneway",
        direction_vals_bft=("B", "FT", "TF"),
        minute_cols=("drivetime_fw", "drivetime_bw"),
        reverse_tofrom=False,
    )
    with pytest.raises(AssertionError):
        _run_od_costs(directed_lines, rules, points)


def _run_od_costs(directed_lines, rules, points):
    nwa = sg.NetworkAnalysis(directed_lines, rules=rules)
    print(nwa)

    od = nwa.od_cost_matrix(points.iloc[[0]], points.iloc[[1]])
    print(od)
    assert od.minutes.sum() == 1

    od = nwa.od_cost_matrix(points.iloc[[1]], points.iloc[[0]])
    print(od)
    assert od.minutes.sum() == 1

    od = nwa.od_cost_matrix(points.iloc[[0]], points.iloc[[2]])
    print(od)
    assert od.minutes.sum() == 2

    od = nwa.od_cost_matrix(points.iloc[[0]], points.iloc[[3]])
    print(od)
    assert od.minutes.isna().all()

    od = nwa.od_cost_matrix(points.iloc[[3]], points.iloc[[0]])
    print(od)
    assert od.minutes.isna().all()

    od = nwa.od_cost_matrix(points.iloc[[3]], points.iloc[[1]])
    print(od)
    assert od.minutes.isna().all()

    od = nwa.od_cost_matrix(points.iloc[[3]], points.iloc[[2]])
    print(od)
    assert od.minutes.sum() == 1


def main():
    test_directed_network()


if __name__ == "__main__":
    main()
