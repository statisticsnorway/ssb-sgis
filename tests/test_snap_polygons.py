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


def not_test_coverage_clean():
    AREA_SHOULD_BE = 104

    with open(Path(__file__).parent / "testdata/dissolve_error.txt") as f:
        df = sg.to_gdf(f.readlines(), 25833)

    dissappears = sg.to_gdf([5.95201, 62.41451], 4326).to_crs(25833).buffer(100)
    df_problem_area = sg.sfilter(df, dissappears.buffer(0.1))

    assert len(df_problem_area) == 3

    assert (area := int(df_problem_area.area.sum())) == AREA_SHOULD_BE, area

    cleaned = sg.coverage_clean(df, 0.1, duplicate_action="fix")

    cleaned_problem_area = sg.sfilter(cleaned, dissappears.buffer(0.1))

    assert (area := int(cleaned_problem_area.area.sum())) == AREA_SHOULD_BE, area

    cleaned_dissolved_problem_area = sg.sfilter(
        sg.dissexp(cleaned), dissappears.buffer(0.1)
    )

    assert len(cleaned_dissolved_problem_area) == 1, cleaned_dissolved_problem_area

    assert (
        area := int(cleaned_dissolved_problem_area.area.sum())
    ) == AREA_SHOULD_BE, area


def not_test_snap():
    mask = sg.to_gdf("POINT (905139.722 7878785.909)", crs=25833).buffer(330)

    df = gpd.read_parquet(Path(__file__).parent / "testdata" / "polygon_snap.parquet")

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

    cleaned = sg.coverage_clean(df, tolerance)  # .pipe(sg.coverage_clean, tolerance)

    sg.qtm(
        cleaned=cleaned.clip(mask).pipe(sg.buff, -0.5),
    )

    gaps = sg.get_gaps(cleaned)
    double = sg.get_intersections(cleaned)

    sg.qtm(
        cleaned=cleaned.clip(mask),
        double=double.clip(mask),
        gaps=gaps.clip(mask),
        alpha=0.5,
    )
    sg.explore(cleaned, double, gaps)

    """cleaned.explode().explode().to_parquet(
        Path(__file__).parent / "testdata" / "polygon_snap_result1.parquet"
    )"""

    assert list(sorted(cleaned.columns)) == ["geometry"], cleaned.columns

    assert double.area.sum() < 1e-6, double.area.sum()
    assert gaps.area.sum() < 1e-2, gaps.area.sum()
    assert int(cleaned.area.sum()) == 431076, cleaned.area.sum()

    assert sg.get_geom_type(cleaned) == "polygon", sg.get_geom_type(cleaned)

    for g in cleaned.explode().explode().geometry:
        sg.qtm(cleaned=cleaned.clip(g.buffer(1)), g=sg.to_gdf(g), alpha=0.5)

    print("\n\nnå cleaned2\n")

    cleaned2 = sg.coverage_clean(cleaned, tolerance=tolerance)

    """cleaned2.to_parquet(
        Path(__file__).parent / "testdata" / "polygon_snap_result2.parquet"
    )"""

    gaps = sg.get_gaps(cleaned2)
    double = sg.get_intersections(cleaned2)
    assert list(sorted(cleaned2.columns)) == ["geometry"], cleaned2.columns

    assert double.area.sum() < 1e-6, double.area.sum()
    assert gaps.area.sum() < 1e-2, gaps.area.sum()
    assert cleaned2.area.sum() > df.area.sum()
    assert int(cleaned2.area.sum()) == 431076, cleaned2.area.sum()

    assert sg.get_geom_type(cleaned2) == "polygon", sg.get_geom_type(cleaned2)

    mask = sg.to_gdf("POINT (905139.722 7878785.909)", crs=25833).buffer(110)

    sg.qtm(cleaned2=cleaned2.clip(mask), alpha=0.5)
    sg.qtm(
        cleaned=cleaned.clip(mask),
        cleaned2=cleaned2.clip(mask),
        alpha=0.5,
        # column="idx",
    )

    snapped = sg.snap_polygons(df, snap_to=cleaned2, tolerance=tolerance * 3)

    """snapped.to_parquet(
        Path(__file__).parent / "testdata" / "polygon_snap_result3.parquet"
    )"""

    mask = sg.to_gdf("POINT (905139.722 7878785.909)", crs=25833).buffer(110)

    sg.qtm(snapped=snapped.clip(mask), alpha=0.5)
    sg.qtm(
        snapped=snapped.clip(mask),
        cleaned2=cleaned2.clip(mask),
        cleaned=cleaned.clip(mask),
        alpha=0.5,
    )
    gaps = sg.get_gaps(snapped)

    double = sg.get_intersections(snapped)
    assert list(sorted(snapped.columns)) == ["geometry"], snapped.columns

    assert double.area.sum() < 1e-6, double.area.sum()
    assert gaps.area.sum() < 1e-2, gaps.area.sum()
    assert snapped.area.sum() > df.area.sum()
    assert int(snapped.area.sum()) == 431076, snapped.area.sum()

    assert sg.get_geom_type(snapped) == "polygon", sg.get_geom_type(snapped)


def main():
    not_test_snap()
    not_test_coverage_clean()


if __name__ == "__main__":
    import cProfile

    # cProfile.run("main()", sort="cumtime")

    main()
