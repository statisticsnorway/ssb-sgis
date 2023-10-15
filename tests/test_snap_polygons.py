# %%

import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


src = str(Path(__file__).parent).strip("tests") + "src"

import sys


sys.path.insert(0, src)


import sgis as sg
from sgis.geopandas_tools.polygons_to_lines import get_cheap_centerlines
from sgis.geopandas_tools.snap_polygons import _concat_gaps_double_and_slivers


def test_remove_points_on_straight_lines():
    line = sg.to_gdf([(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (3, 3), (4, 4), (5, 5)])
    line.index = [1] * len(line)

    t = remove_points_on_straight_lines(t, gap_lines)


def test_coverage_clean():
    AREA_SHOULD_BE = 104

    with open(Path(__file__).parent / "testdata/dissolve_error.txt") as f:
        df = sg.to_gdf(f.readlines(), 25833)

    dissappears = sg.to_gdf([5.95201, 62.41451], 4326).to_crs(25833).buffer(100)
    df_problem_area = sg.sfilter(df, dissappears.buffer(0.1))

    assert len(df_problem_area) == 3

    assert (area := int(df_problem_area.area.sum())) == AREA_SHOULD_BE, area

    cleaned = sg.coverage_clean(df, 0.1)

    snapped = sg.snap_polygons(df, 0.1).assign(
        idx=lambda x: [str(i) for i in range(len(x))]
    )

    cleaned_problem_area = sg.sfilter(cleaned, dissappears.buffer(0.1))
    snapped_problem_area = sg.sfilter(snapped, dissappears.buffer(0.1))

    assert len(cleaned_problem_area) == 1
    assert len(snapped_problem_area) == 1

    assert (area := int(cleaned_problem_area.area.sum())) == AREA_SHOULD_BE, area
    assert (area := int(snapped_problem_area.area.sum())) == AREA_SHOULD_BE, area

    cleaned_dissolved_problem_area = sg.sfilter(
        sg.dissexp(cleaned), dissappears.buffer(0.1)
    )
    snapped_dissolved_problem_area = sg.sfilter(
        sg.dissexp(snapped), dissappears.buffer(0.1)
    )

    assert len(cleaned_dissolved_problem_area) == 1, cleaned_dissolved_problem_area
    assert len(snapped_dissolved_problem_area) == 1, snapped_dissolved_problem_area

    assert (
        area := int(cleaned_dissolved_problem_area.area.sum())
    ) == AREA_SHOULD_BE, area
    assert (
        area := int(snapped_dissolved_problem_area.area.sum())
    ) == AREA_SHOULD_BE, area


def test_snap():
    mask = sg.to_gdf("POINT (905139.722 7878785.909)", crs=25833).buffer(330)

    df = gpd.read_parquet(Path(__file__).parent / "testdata" / "polygon_snap.parquet")
    gaps = gpd.read_parquet(Path(__file__).parent / "testdata" / "gap_lines.parquet")

    holes = sg.to_gdf(
        [
            "POINT (905200 7878700)",
            "POINT (905250 7878780)",
            "POINT (905242.961 7878773.758)",
        ],
        25833,
    ).pipe(sg.buff, 3)

    df = sg.clean_overlay(df, holes, how="difference")

    sg.qtm(df.clip(mask))

    sg.qtm(df=df.clip(mask).pipe(sg.buff, -0.5))

    tolerance = 5

    gaps, _ = _concat_gaps_double_and_slivers(df, tolerance)
    snap_to = get_cheap_centerlines(gaps)

    snapped = sg.snap_polygons(df, tolerance).pipe(sg.snap_polygons, tolerance)

    sg.qtm(
        snapped=snapped.clip(mask).pipe(sg.buff, -0.5),
    )

    gaps = sg.get_gaps(snapped)
    double = sg.get_intersections(snapped)

    sg.qtm(
        snapped=snapped.clip(mask),
        double=double.clip(mask),
        gaps=gaps.clip(mask),
        alpha=0.5,
    )
    sg.explore(snap_to, snapped, double, gaps)

    snapped.explode().explode().to_parquet(
        Path(__file__).parent / "testdata" / "polygon_snap_result1.parquet"
    )

    assert list(sorted(snapped.columns)) == ["geometry"], snapped.columns

    assert double.area.sum() < 1e-6, double.area.sum()
    assert gaps.area.sum() < 1e-2, gaps.area.sum()
    assert snapped.area.sum() > df.area.sum()

    assert sg.get_geom_type(snapped) == "polygon", sg.get_geom_type(snapped)

    for g in snapped.explode().explode().geometry:
        sg.qtm(snapped=snapped.clip(g.buffer(1)), g=sg.to_gdf(g), alpha=0.5)

    print("\n\nnå snapped2\n")

    snapped2 = sg.snap_polygons(snapped, tolerance=tolerance).assign(
        idx=lambda x: [str(i) for i in range(len(x))]
    )

    snapped2.to_parquet(
        Path(__file__).parent / "testdata" / "polygon_snap_result2.parquet"
    )

    mask = sg.to_gdf("POINT (905139.722 7878785.909)", crs=25833).buffer(110)

    sg.qtm(snapped2=snapped2.clip(mask), alpha=0.5)
    sg.qtm(
        snapped=snapped.clip(mask),
        snapped2=snapped2.clip(mask),
        alpha=0.5,
        # column="idx",
    )

    snapped3 = sg.snap_polygons(df, snap_to=snapped2, tolerance=tolerance).assign(
        idx=lambda x: [str(i) for i in range(len(x))]
    )

    snapped3.to_parquet(
        Path(__file__).parent / "testdata" / "polygon_snap_result3.parquet"
    )

    mask = sg.to_gdf("POINT (905139.722 7878785.909)", crs=25833).buffer(110)

    sg.qtm(snapped3=snapped3.clip(mask), alpha=0.5)
    sg.qtm(
        snapped=snapped.clip(mask),
        snapped2=snapped2.clip(mask),
        snapped3=snapped3.clip(mask),
        alpha=0.5,
        # column="idx",
    )
    assert (dup := sg.get_intersections(snapped)).area.sum() < 1e-6, (
        dup.assign(areal=lambda x: x.area),
        dup.area.sum(),
    )
    assert (dup := sg.get_intersections(snapped2)).area.sum() < 1e-6, (
        dup.assign(areal=lambda x: x.area),
        dup.area.sum(),
    )
    assert (dup := sg.get_intersections(snapped3)).area.sum() < 1e-6, (
        dup.assign(areal=lambda x: x.area),
        dup.area.sum(),
    )


def main():
    test_snap()
    test_coverage_clean()


if __name__ == "__main__":
    main()
