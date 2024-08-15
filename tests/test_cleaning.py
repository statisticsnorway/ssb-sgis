# %%

import random
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

src = str(Path(__file__).parent).replace("tests", "") + "src"


sys.path.insert(0, src)

import sgis as sg

# sizes = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
# coords = (
#     [(x, y) for x, y in zip(sizes[::-1], sizes)]
#     + [(x, y) for x, y in zip(sizes, sizes[::-1])]
#     + [(x, x) for x in sizes]
#     + [(0, x) for x in sizes]
#     + [(x, 0) for x in sizes]
#     + [(0, 0)]
# )

# df = sg.to_gdf(coords)
# df.geometry = df.buffer(10).difference(df.buffer(9.9))
# dissolved = df.dissolve()
# sg.explore(df, dissolved)
# sss


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

    # cleaned_dissolved_problem_area.to_parquet(
    #     "c:/users/ort/downloads/cleaned_dissolved_problem_area.parquet"
    # )

    assert len(cleaned_dissolved_problem_area) == 1, (
        sg.explore(
            cleaned_dissolved_problem_area.assign(
                col=lambda x: [str(i) for i in range(len(x))]
            ),
            "col",
        ),
        cleaned_dissolved_problem_area,
    )

    assert (
        area := int(cleaned_dissolved_problem_area.area.sum())
    ) == AREA_SHOULD_BE, area


def test_clean_complicated_roads():
    for tolerance in [
        0.5,
    ]:

        _test_clean_complicated_roads_base(
            Path(__file__).parent / "testdata/roads_difficult_to_dissolve4.txt",
            "POLYGON ((-32050 6557614, -32050 6556914, -32750 6556914, -32750 6557614, -32050 6557614))",
            tolerance=tolerance,
        )

        _test_clean_complicated_roads_base(
            Path(__file__).parent / "testdata/roads_difficult_to_dissolve3.txt",
            "POLYGON ((28120 6945720, 28120 6945020, 27420 6945020, 27420 6945720, 28120 6945720))",
            tolerance=tolerance,
        )

        _test_clean_complicated_roads_base(
            Path(__file__).parent / "testdata/roads_difficult_to_dissolve2.txt",
            "POLYGON ((270257 6654842, 270257 6654142, 269557 6654142, 269557 6654842, 270257 6654842))",
            tolerance=tolerance,
        )

        _test_clean_complicated_roads_base(
            Path(__file__).parent / "testdata/roads_difficult_to_dissolve.txt",
            "POLYGON ((-49922 6630166, -49922 6629466, -50622 6629466, -50622 6630166, -49922 6630166))",
            tolerance=tolerance,
        )


def test_clean_dissexp():
    from geopandas import GeoDataFrame
    from shapely import extract_unique_points
    from shapely import get_parts

    df = sg.to_gdf(
        [
            "POLYGON ((373693.16000000015 7321024.640000001, 373690.5999999996 7321023.460000001, 373688.5499999998 7321022.210000001, 373686.01999999955 7321021.34, 373685.04000000004 7321020.43, 373684.76999999955 7321019.190000001, 373681.96999999974 7321015.460000001, 373680.11000000034 7321012.82, 373677.33999999985 7321010.59, 373673.21999999974 7321003.699999999, 373671.70999999996 7321002.870000001, 373667.29000000004 7321001.620000001, 373677.5 7321015, 373695 7321030, 373700.8520873802 7321030, 373695.46999999974 7321027.460000001, 373694.63999999966 7321026.039999999, 373693.16000000015 7321024.640000001))",
            "POLYGON ((373700.4003424102 7321029.786805352, 373700.8520873802 7321030, 373700.85208738025 7321030, 373700.4003424102 7321029.786805352))",
        ],
        25833,
    )

    original_points = GeoDataFrame(
        {"geometry": get_parts(extract_unique_points(df.geometry.values))}
    )[lambda x: ~x.geometry.duplicated()]

    cleaned = sg.clean_dissexp(df, dissolve_func=sg.dissexp, by=None)

    print(cleaned)
    sg.explore(cleaned)

    cleaned.geometry = extract_unique_points(cleaned.geometry.values)
    assert cleaned.index.is_unique
    cleaned = cleaned.explode(index_parts=True)

    still_in, gone = sg.sfilter_split(cleaned, original_points.buffer(1e-10))
    sg.explore(still_in, gone, df, cleaned)


def _test_clean_complicated_roads_base(path, mask, tolerance):
    print()
    print(path)

    with open(path) as f:
        df = sg.to_gdf(f.readlines(), 25833)

    mask = sg.to_gdf(
        mask,
        25833,
    )
    df["ARTYPE"] = 12
    cleaned = sg.coverage_clean(df, tolerance, mask=mask)

    gaps = sg.get_gaps(cleaned)
    double = sg.get_intersections(cleaned)
    # missing = get_missing(sg.clean_clip(df, bbox.buffer(-tolerance * 1.1)), cleaned)
    missing = get_missing(df, cleaned)
    sg.explore(df, cleaned, gaps, missing, double)

    print(
        f"tolerance {tolerance}",
        "gaps",
        gaps.area.sum(),
        "dup",
        double.area.sum(),
        "missing",
        missing.area.sum(),
    )
    assert gaps.area.sum() <= 1e-6, f"tolerance {tolerance}, gaps: {gaps.area.sum()}"
    assert (
        double.area.sum() <= 1e-6
    ), f"tolerance {tolerance}, double: {double.area.sum()}"
    assert (
        missing.area.sum() <= 1e-6
    ), f"tolerance {tolerance}, missing: {missing.area.sum()}"


def test_clean_1144():
    df = gpd.read_parquet(
        Path(__file__).parent / "testdata" / "snap_problem_area_1144.parquet"
    )

    bbox = sg.to_gdf(shapely.minimum_rotated_rectangle(df.union_all()), df.crs)

    kommune_utenhav = gpd.read_parquet(
        Path(__file__).parent / "testdata" / "kommune_utenhav_1144_2023.parquet"
    )

    kommune_utenhav = sg.clean_clip(kommune_utenhav, bbox, geom_type="polygon")

    assert sg.get_intersections(df).dissolve().area.sum() == 0
    assert int(df.area.sum()) == 154240, df.area.sum()

    cols = [
        "ARGRUNNF",
        "ARJORDBR",
        "ARKARTSTD",
        "ARSKOGBON",
        "ARTRESLAG",
        "ARTYPE",
        "ARVEGET",
        "ASTSSB",
        "df_idx",
        "geometry",
        "kilde",
    ]

    df["df_idx"] = range(len(df))

    for tolerance in [
        0.51,
        0.5,
        0.91,
        0.57,
        5,
        1,
        2,
        0.75,
        1.5,
        2.25,
        *[round(random.random() + 0.5, 2) for _ in range(10)],
        *[round(x, 2) for x in np.arange(0.4, 1, 0.01)],
    ]:
        print("\ntolerance")
        print(tolerance)
        # cleaned = sg.coverage_clean(df, tolerance, pre_dissolve_func=_buff).pipe(
        #     sg.clean_clip, df, geom_type="polygon"

        # )

        # allow near-thin polygons to dissappear. this happens because snapping makes them thin
        # before eliminate

        thick_df_indices = df.loc[
            lambda x: ~x.buffer(-tolerance / 1.3).is_empty, "df_idx"
        ]

        cleaned = sg.coverage_clean(
            df, tolerance, mask=kommune_utenhav
        )  # .pipe(sg.coverage_clean, tolerance)

        # cleaned = sg.coverage_clean(
        #     sg.sort_large_first(df), tolerance, mask=kommune_utenhav
        # ).pipe(sg.snap_polygons, 0.1, mask=kommune_utenhav)

        # allow edge cases
        cleaned_clipped = sg.clean_clip(cleaned, bbox.buffer(-tolerance * 1.1))

        gaps = sg.get_gaps(cleaned_clipped)

        double = sg.get_intersections(cleaned_clipped)
        missing = get_missing(
            sg.clean_clip(df, bbox.buffer(-tolerance * 1.1)), cleaned_clipped
        )

        cleaned_points = cleaned.extract_unique_points().to_frame("geometry").explode()
        df_points = df.extract_unique_points().to_frame("geometry").explode()

        sg.explore(
            cleaned,
            gaps,
            double,
            missing,
            df,
            kommune_utenhav,
            center=sg.debug_config._DEBUG_CONFIG["center"],
        )
        sg.explore(
            cleaned,
            gaps,
            double,
            missing,
            df,
            kommune_utenhav,
            cleaned_points,
            df_points,
            gaps_buff=sg.buff(gaps, np.log(gaps.area.values + 2) ** 2),
            missing_buff=sg.buff(missing, np.log(missing.area.values + 2) ** 2),
            double_buff=sg.buff(double, np.log(double.area.values + 2) ** 2),
        )

        print(
            f"tolerance {tolerance}",
            "gaps",
            gaps.area.sum(),
            "dup",
            double.area.sum(),
            "missing",
            missing.area.sum(),
        )
        assert (
            gaps.area.sum() <= 1e-6
        ), f"tolerance {tolerance}, gaps: {gaps.area.sum()}"
        assert (
            double.area.sum() <= 1e-6
        ), f"tolerance {tolerance}, double: {double.area.sum()}"
        assert (
            missing.area.sum() <= 1e-3
        ), f"tolerance {tolerance}, missing: {missing.area.sum()}"

        assert thick_df_indices.isin(cleaned_clipped["df_idx"]).all(), sg.explore(
            df,
            cleaned,
            missing_polygons=df[
                (df["df_idx"].isin(thick_df_indices))
                & (~df["df_idx"].isin(cleaned_clipped["df_idx"]))
            ],
        )

        notna_df = df.notna().all()
        cols_notna = list(notna_df[lambda x: x == True].index)
        notna_df_relevant_cols = df[cols_notna].notna().all()
        notna_cleaned = cleaned[cols_notna].notna().all()
        assert notna_cleaned.equals(notna_df_relevant_cols), (
            notna_cleaned,
            notna_df_relevant_cols,
            cleaned[cols_notna].sort_values(by=cols_notna),
        )

        assert list(sorted(cleaned.columns)) == list(sorted(cols)), cleaned.columns


def get_missing(df, other):
    return (
        sg.clean_overlay(df, other, how="difference", geom_type="polygon")
        # .pipe(sg.buff, -0.0001)
        # .pipe(sg.clean_overlay, other, how="difference", geom_type="polygon")
        .pipe(sg.sfilter_inverse, other.buffer(-0.001))
    )


def test_clean():

    df = gpd.read_parquet(Path(__file__).parent / "testdata" / "polygon_snap.parquet")

    bbox = sg.to_gdf(shapely.minimum_rotated_rectangle(df.union_all()), df.crs)

    kommune_utenhav = gpd.read_parquet(
        Path(__file__).parent / "testdata" / "kommune_utenhav_5435_2023.parquet"
    )
    kommune_utenhav = sg.clean_clip(kommune_utenhav, bbox, geom_type="polygon")

    holes = sg.to_gdf(
        [
            "POINT (905200 7878700)",
            "POINT (905250 7878780)",
            "POINT (905242.961 7878773.758)",
        ],
        25833,
    ).pipe(sg.buff, 3)

    df = sg.clean_overlay(df, holes, how="difference")

    mask = sg.close_all_holes(sg.dissexp_by_cluster(df)).dissolve()

    for tolerance in [5, 6, 7, 8, 9, 10]:
        print("tolerance:", tolerance)

        cleaned = sg.coverage_clean(df, tolerance)
        assert sg.get_geom_type(cleaned) == "polygon", sg.get_geom_type(cleaned)

        double = sg.get_intersections(cleaned).loc[lambda x: ~x.buffer(-1e-9).is_empty]
        gaps = sg.get_gaps(cleaned).loc[lambda x: ~x.buffer(-1e-9).is_empty]
        missing = get_missing(df, cleaned)

        print(double.area.sum(), missing.area.sum(), gaps.area.sum())

        sg.explore(
            df=df.to_crs(25833),
            cleaned=cleaned.to_crs(25833),
            double=double.to_crs(25833),
            missing=missing,
            gaps=gaps.to_crs(25833),
        )

        assert (a := max(list(double.area) + [0])) < 1e-5, a
        assert (a := max(list(missing.area) + [0])) < 1e-5, a
        assert (a := max(list(gaps.area) + [0])) < 1e-5, a

        notna_cleaned = cleaned[df.columns].notna().all()
        notna_df = df.notna().all()
        assert notna_cleaned.equals(notna_df), (notna_cleaned, notna_df)

    sg.explore(
        cleaned1=sg.coverage_clean(df, 1),
        cleaned3=sg.coverage_clean(df, 3),
        cleaned5=sg.coverage_clean(df, 5),
        df=df,
    )


def not_test_spikes():
    from shapely.geometry import Polygon

    factor = 10000

    sliver = sg.to_gdf(
        Polygon(
            [
                (0, 0),
                (0.1 * factor, 1 * factor),
                (0, 2 * factor),
                (-0.1 * factor, 1 * factor),
            ]
        )
    ).assign(what="sliver", num=1)
    poly_with_spike = sg.to_gdf(
        Polygon(
            [
                (0 * factor, 0 * factor),
                (-0.1 * factor, 1 * factor),
                (0 * factor, 2 * factor),
                (-0.99 * factor, 2.001 * factor),
                (-0.99 * factor, 1.5 * factor),
                (-1.01 * factor, 1.5 * factor),
                (-1.01 * factor, 2.001 * factor),
                (-1.51 * factor, 2.001 * factor),
                (-1.51 * factor, 1.7 * factor),
                (-1.52 * factor, 2.001 * factor),
                (-2 * factor, 2.001 * factor),
                (-1 * factor, 1 * factor),
            ],
            holes=[
                (
                    [
                        (-0.5 * factor, 1.25 * factor),
                        (-0.5 * factor, 1.65 * factor),
                        (-0.49 * factor, 1.65 * factor),
                        (-0.49 * factor, 1.25 * factor),
                    ]
                ),
            ],
        )
    ).assign(what="small", num=2)
    poly_filling_the_spike = sg.to_gdf(
        Polygon(
            [
                (0, 2.001 * factor),
                (-0.99 * factor, 2 * factor),
                (-0.99 * factor, 1.5 * factor),
                (-1.01 * factor, 1.5 * factor),
                (-1.01 * factor, 2 * factor),
                (-2 * factor, 2 * factor),
                (-2 * factor, 6 * factor),
                (0, 6 * factor),
                (0, 2.001 * factor),
            ],
        )
    ).assign(what="large", num=2)

    df = pd.concat([sliver, poly_with_spike, poly_filling_the_spike])
    holes = sg.buff(
        sg.to_gdf([(-0.84 * factor, 3 * factor), (-0.84 * factor, 4.4 * factor)]),
        [0.4 * factor, 0.3 * factor],
    )
    df = sg.clean_overlay(df, holes, how="update")
    df.crs = 25833

    tolerance = 0.09 * factor

    cleaned = sg.coverage_clean(df, tolerance)
    gaps = sg.get_gaps(cleaned, True)

    if __name__ == "__main__":
        sg.explore(
            cleaned=cleaned,
            gaps=gaps,
        )

    def is_close_enough(num1, num2):
        if num1 >= num2 - 1e-3 and num1 <= num2 + 1e-3:
            return True
        return False

    return
    area_should_be = [
        725264293.6535025,
        20000000.0,
        190000000.0,
        48285369.993336275,
        26450336.353161283,
    ]
    print(list(cleaned.area))
    for area1, area2 in zip(
        sorted(cleaned.area),
        sorted(area_should_be),
        strict=False,
    ):
        assert is_close_enough(area1, area2), (area1, area2)

    length_should_be = [
        163423.91054766334,
        40199.502484483564,
        68384.02248970368,
        24882.8908851665,
        18541.01966249684,
    ]

    print(list(cleaned.length))

    for length1, length2 in zip(
        sorted(cleaned.length),
        sorted(length_should_be),
        strict=False,
    ):
        assert is_close_enough(length1, length2), (length1, length2)


def main():
    test_clean_dissexp()
    test_clean_1144()
    test_clean_complicated_roads()
    test_clean()
    test_clean_dissappearing_polygon()
    not_test_spikes()


if __name__ == "__main__":

    # df = cprofile_df("main()")
    # print(df.iloc[:50])
    # print(df.iloc[50:100])

    # import cProfile

    # cProfile.run("main()", sort="cumtime")
    _time = time.perf_counter()
    main()
    print("seconds passed:", time.perf_counter() - _time)


# %%
