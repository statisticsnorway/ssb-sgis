# %%

import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import extract_unique_points


src = str(Path(__file__).parent).strip("tests") + "src"


sys.path.insert(0, src)

import sgis as sg


def test_clean_dissappearing_polygon():
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


def test_clean_1144():
    df = gpd.read_parquet(
        Path(__file__).parent / "testdata" / "snap_problem_area_1144.parquet"
    )

    for tolerance in [1, 0.1, 0.01, 0.001]:
        print(tolerance)

        cleaned = sg.coverage_clean(df, tolerance)

        gaps = sg.get_gaps(cleaned)
        double = sg.get_intersections(cleaned)

        sg.explore(
            df,  # =cleaned.assign(wkt=lambda x: x.geometry.to_wkt()),
            cleaned,  # =cleaned.assign(wkt=lambda x: x.geometry.to_wkt()),
            double,  # =double.assign(wkt=lambda x: x.to_wkt()),
            gaps,  # =gaps.assign(wkt=lambda x: x.to_wkt()),
        )

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

        assert double.area.sum() < 1e-3, double.area.sum()
        assert gaps.area.sum() < 1e-2, (
            gaps.area.sum(),
            gaps.area.max(),
            gaps.area,
            gaps,
        )
        assert int(cleaned.area.sum()) == 154240, cleaned.area.sum()
        assert int(df.area.sum()) == 154240, df.area.sum()

        assert sg.get_geom_type(cleaned) == "polygon", sg.get_geom_type(cleaned)

        cleaned2 = sg.coverage_clean(cleaned, tolerance=tolerance)

        gaps = sg.get_gaps(cleaned2)
        double = sg.get_intersections(cleaned2)

        assert list(sorted(cleaned2.columns)) == cols, cleaned2.columns

        assert double.area.sum() < 1e-3, double.area.sum()
        assert gaps.area.sum() < 1e-2, gaps.area.sum()
        assert int(cleaned2.area.sum()) == 154240, cleaned2.area.sum()

        assert sg.get_geom_type(cleaned2) == "polygon", sg.get_geom_type(cleaned2)

        # snapped = sg.snap_polygons(df, cleaned, tolerance)


def test_clean():
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

    # sg.qtm(df.clip(mask), alpha=0.5)
    # sg.qtm(df=df.clip(mask).pipe(sg.buff, -0.5), alpha=0.5)

    tolerance = 5

    cleaned = sg.coverage_clean(df, tolerance)


def test_spikes():
    from shapely.geometry import Point, Polygon

    sliver = sg.to_gdf(Polygon([(0, 0), (0.1, 1), (0, 2), (-0.1, 1)])).assign(
        what="sliver", num=1
    )
    poly_with_spike = sg.to_gdf(
        Polygon(
            [
                (0, 0),
                (-0.1, 1),
                (0, 2),
                (-0.99, 2),
                (-0.99, 1.5),
                (-1.01, 1.5),
                (-1.01, 2),
                (-1.51, 2),
                (-1.51, 1.7),
                (-1.52, 2),
                (-2, 2),
                (-1, 1),
            ],
            holes=[
                (
                    [
                        (-0.5, 1.25),
                        (-0.5, 1.65),
                        (-0.49, 1.65),
                        (-0.49, 1.25),
                    ]
                ),
            ],
        )
    ).assign(what="small", num=2)
    poly_filling_the_spike = sg.to_gdf(
        Polygon(
            [
                (0, 2),
                (-0.99, 2),
                (-0.99, 1.5),
                (-1.01, 1.5),
                (-1.01, 2),
                (-2, 2),
                (-2, 6),
                (0, 6),
                (0, 2),
            ],
        )
    ).assign(what="small", num=2)

    df = pd.concat([sliver, poly_with_spike, poly_filling_the_spike])
    holes = sg.buff(sg.to_gdf([(-0.84, 3), (-0.84, 4.4)]), [0.4, 0.3])
    df = sg.clean_overlay(df, holes, how="update")

    tolerance = 0.09

    cleaned = sg.coverage_clean(df, tolerance)
    if __name__ == "__main__":
        sg.explore(
            cleaned=cleaned,
            df=df,
        )

    assert (area := sorted([round(x, 3) for x in cleaned.area])) == sorted(
        [
            7.225,
            1.89,
            0.503,
            0.283,
            0.2,
        ]
    ), area
    assert (length := sorted([round(x, 3) for x in cleaned.length])) == sorted(
        [
            17.398,
            7.838,
            2.513,
            1.885,
            4.02,
        ]
    ), length

    spikes_fixed = sg.split_spiky_polygons(df, tolerance)
    fixed_and_cleaned = sg.coverage_clean(spikes_fixed, tolerance)

    if __name__ == "__main__":
        sg.explore(
            fixed_and_cleaned=fixed_and_cleaned,
            spikes_fixed=spikes_fixed,
            cleaned=cleaned,
            df=df,
        )

    assert (area := sorted([round(x, 3) for x in fixed_and_cleaned.area])) == sorted(
        [
            7.215,
            1.9,
            0.503,
            0.283,
            0.2,
        ]
    ), area
    assert (
        length := sorted([round(x, 3) for x in fixed_and_cleaned.length])
    ) == sorted(
        [
            16.398,
            6.838,
            2.513,
            1.885,
            4.02,
        ]
    ), length


def main():
    test_spikes()
    test_clean()
    test_clean_1144()
    test_clean_dissappearing_polygon()


if __name__ == "__main__":
    import cProfile

    # cProfile.run("main()", sort="cumtime")

    main()

# %%
()
