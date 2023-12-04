# %%

import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import extract_unique_points


src = str(Path(__file__).parent).strip("tests") + "src"


import sys


sys.path.insert(0, src)


import sgis as sg


def test_coverage_clean():
    AREA_SHOULD_BE = 104

    with open(Path(__file__).parent / "testdata/dissolve_error.txt") as f:
        df = sg.to_gdf(f.readlines(), 25833)

    dissappears = sg.to_gdf([5.95201, 62.41451], 4326).to_crs(25833).buffer(100)
    df_problem_area = sg.sfilter(df, dissappears.buffer(0.1))

    assert len(df_problem_area) == 3

    assert (area := int(df_problem_area.area.sum())) == AREA_SHOULD_BE, area

    cleaned = sg.coverage_clean(df, 0.1, duplicate_action="fix")

    cleaned_problem_area = sg.sfilter(cleaned, dissappears.buffer(0.1))

    sg.explore(cleaned, cleaned_problem_area, dissappears, df_problem_area)
    assert (area := int(cleaned_problem_area.area.sum())) == AREA_SHOULD_BE, area

    cleaned_dissolved_problem_area = sg.sfilter(
        sg.dissexp(cleaned), dissappears.buffer(0.1)
    )

    assert len(cleaned_dissolved_problem_area) == 1, cleaned_dissolved_problem_area

    assert (
        area := int(cleaned_dissolved_problem_area.area.sum())
    ) == AREA_SHOULD_BE, area


def test_snap_problem_area():
    df = gpd.read_parquet(
        Path(__file__).parent / "testdata" / "snap_problem_area_1144.parquet"
    )

    tolerance = 0.1

    cleaned = sg.coverage_clean(df, tolerance)

    gaps = sg.get_gaps(cleaned)
    double = sg.get_intersections(cleaned)

    # sg.explore(
    #     df,  # =cleaned.assign(wkt=lambda x: x.geometry.to_wkt()),
    #     cleaned,  # =cleaned.assign(wkt=lambda x: x.geometry.to_wkt()),
    #     double,  # =double.assign(wkt=lambda x: x.to_wkt()),
    #     gaps,  # =gaps.assign(wkt=lambda x: x.to_wkt()),
    # )

    cols = [
        "ARGRUNNF",
        "ARJORDBR",
        "ARKARTSTD",
        "ARSKOGBON",
        "ARTRESLAG",
        "ARTYPE",
        "ARVEGET",
        "ASTSSB",
        "geometry",
        "kilde",
    ]

    assert list(sorted(cleaned.columns)) == cols, list(sorted(cleaned.columns))

    assert double.area.sum() < 1e-6, double.area.sum()
    assert gaps.area.sum() < 1e-2, gaps.area.sum()
    assert int(cleaned.area.sum()) == 154240, cleaned.area.sum()
    assert int(df.area.sum()) == 154240, df.area.sum()

    assert sg.get_geom_type(cleaned) == "polygon", sg.get_geom_type(cleaned)

    cleaned2 = sg.coverage_clean(cleaned, tolerance=tolerance)

    gaps = sg.get_gaps(cleaned2)
    double = sg.get_intersections(cleaned2)

    assert list(sorted(cleaned2.columns)) == cols, cleaned2.columns

    assert double.area.sum() < 1e-6, double.area.sum()
    assert gaps.area.sum() < 1e-2, gaps.area.sum()
    assert int(cleaned2.area.sum()) == 154240, cleaned2.area.sum()

    assert sg.get_geom_type(cleaned2) == "polygon", sg.get_geom_type(cleaned2)


def test_snap():
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

    sg.qtm(df.clip(mask), alpha=0.5)

    sg.qtm(df=df.clip(mask).pipe(sg.buff, -0.5), alpha=0.5)

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

    sg.explore(
        cleaned=cleaned,  # .assign(wkt=lambda x: x.to_wkt()),
        double=double,  # .assign(wkt=lambda x: x.to_wkt()),
        gaps=gaps,  # .assign(wkt=lambda x: x.to_wkt()),
    )

    assert list(sorted(cleaned.columns)) == ["geometry"], cleaned.columns

    assert double.area.sum() < 1e-6, double.area.sum()
    assert gaps.area.sum() < 1e-2, gaps.area.sum()
    assert int(cleaned.area.sum()) == 431076, cleaned.area.sum()

    assert sg.get_geom_type(cleaned) == "polygon", sg.get_geom_type(cleaned)

    for g in cleaned.explode().explode().geometry:
        sg.qtm(cleaned=cleaned.clip(g.buffer(1)), g=sg.to_gdf(g), alpha=0.5)

    cleaned2 = sg.coverage_clean(cleaned, tolerance=tolerance)

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
    sg.explore(
        cleaned=cleaned.clip(mask),
        cleaned2=cleaned2.clip(mask),
        # alpha=0.5,
        # column="idx",
    )

    # add a double surface
    df = pd.concat([df, sg.buff(df.iloc[[0]], 10)], ignore_index=True)

    cleaned3 = sg.coverage_clean(df, tolerance=tolerance)

    gaps = sg.get_gaps(cleaned3)
    double = sg.get_intersections(cleaned3)

    if __name__ == "__main__":
        sg.explore(
            gaps=gaps,  # .clip(mask),
            double=double,  # .clip(mask),
            cleaned3=cleaned3,  # .clip(mask),
            p=sg.to_gdf(extract_unique_points(cleaned3.geometry)).pipe(sg.buff, 1),
            # alpha=0.5,
            # column="idx",
        )

    assert list(sorted(cleaned3.columns)) == ["geometry"], cleaned3.columns

    assert double.area.sum() < 1e-6, double.area.sum()
    assert gaps.area.sum() < 1e-2, gaps.area.sum()
    assert int(cleaned3.area.sum()) == 441769, cleaned3.area.sum()

    assert sg.get_geom_type(cleaned3) == "polygon", sg.get_geom_type(cleaned3)

    mask = sg.to_gdf("POINT (905139.722 7878785.909)", crs=25833).buffer(110)

    if __name__ == "__main__":
        sg.qtm(cleaned3=cleaned3.clip(mask), alpha=0.5)
        sg.explore(
            cleaned=cleaned.clip(mask),
            cleaned2=cleaned2.clip(mask),
            cleaned3=cleaned3.clip(mask),
            # alpha=0.5,
            # column="idx",
        )

    return

    snapped = sg.snap_polygons(df, snap_to=cleaned2, tolerance=tolerance)

    """snapped.to_parquet(
        Path(__file__).parent / "testdata" / "polygon_snap_result3.parquet"
    )"""

    mask = sg.to_gdf("POINT (905139.722 7878785.909)", crs=25833).buffer(110)

    sg.qtm(snapped=snapped.clip(mask), alpha=0.5)

    sg.qtm(
        snapped=snapped.clip(mask),
        cleaned2=cleaned2.clip(mask),
        cleaned=cleaned.clip(mask),
        µalpha=0.5,
    )
    gaps = sg.get_gaps(snapped)

    double = sg.get_intersections(snapped)

    sg.explore(snapped, double, gaps)

    assert list(sorted(snapped.columns)) == ["geometry"], snapped.columns

    assert double.area.sum() < 1e-6, double.area.sum()

    assert gaps.area.sum() < 1e-2, gaps.area.sum()

    assert snapped.area.sum() > df.area.sum()

    assert int(snapped.area.sum()) == 431076, snapped.area.sum()

    assert sg.get_geom_type(snapped) == "polygon", sg.get_geom_type(snapped)


def main():
    test_snap()
    test_coverage_clean()
    test_snap_problem_area()


if __name__ == "__main__":
    import cProfile

    # cProfile.run("main()", sort="cumtime")

    main()
