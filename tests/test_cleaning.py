# %%

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import extract_unique_points, minimum_rotated_rectangle


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


def test_clean_1144():
    df = gpd.read_parquet(
        Path(__file__).parent / "testdata" / "snap_problem_area_1144.parquet"
    )

    bbox = sg.to_gdf(minimum_rotated_rectangle(df.unary_union), df.crs)

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

    for tolerance in [2, 1, 5]:
        print("\ntolerance")
        print(tolerance)
        # cleaned = sg.coverage_clean(df, tolerance, pre_dissolve_func=_buff).pipe(
        #     sg.clean_clip, df, geom_type="polygon"

        # )

        thick_df_indices = df.loc[
            lambda x: ~x.buffer(-tolerance / 2).is_empty, "df_idx"
        ]

        cleaned = sg.coverage_clean(df, tolerance)

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

        print(
            "cleaned",
            gaps.area.sum(),
            double.area.sum(),
            missing.area.sum(),
        )

        assert thick_df_indices.isin(cleaned_clipped["df_idx"]).all(), sg.explore(
            df,
            cleaned,
            missing_polygons=df[
                (df["df_idx"].isin(thick_df_indices))
                & (~df["df_idx"].isin(cleaned_clipped["df_idx"]))
            ],
        )

        snapped_to_mask = sg.snap_to_mask(
            sg.sort_large_first(df), tolerance, mask=kommune_utenhav
        )

        continue

        gaps = sg.get_gaps(snapped_to_mask)
        double = sg.get_intersections(snapped_to_mask)
        missing = sg.clean_overlay(
            sg.clean_clip(df, bbox.buffer(-tolerance * 1.1)),
            snapped_to_mask,
            how="difference",
            geom_type="polygon",
        )

        if missing.area.sum() > 1:
            sg.explore(
                missing,
                gaps,
                double,
                snapped_to_mask,
                df,
                kommune_utenhav,
            )

        assert double.area.sum() < 1e-4, double.area.sum()
        assert missing.area.sum() < 1e-4, missing.area.sum()
        assert gaps.area.sum() < 1e-4, gaps.area.sum()

        # assert thick_df_indices.isin(snapped_to_mask["df_idx"]).all(), sg.explore(
        #     df,
        #     snapped_to_mask,
        #     missing_polygons=df[
        #         (df["df_idx"].isin(thick_df_indices))
        #         & (~df["df_idx"].isin(snapped_to_mask["df_idx"]))
        #     ],
        # )

        print(
            "snapped_to_mask",
            gaps.area.sum(),
            double.area.sum(),
            missing.area.sum(),
        )

        snapped = sg.snap_polygons(
            sg.sort_large_first(df), tolerance, mask=kommune_utenhav
        )

        gaps = sg.get_gaps(snapped)
        double = sg.get_intersections(snapped)

        missing = get_missing(sg.clean_clip(df, bbox.buffer(-tolerance * 1.1)), snapped)

        sg.explore(
            missing,
            gaps,
            double,
            snapped,
            df,
            kommune_utenhav,
        )

        # assert thick_df_indices.isin(snapped["df_idx"]).all(), sg.explore(
        #     df,
        #     snapped_to_mask,
        #     missing_polygons=df[
        #         (df["df_idx"].isin(thick_df_indices))
        #         & (~df["df_idx"].isin(snapped["df_idx"]))
        #     ],
        # )
        assert double.area.sum() < 1e-4, double.area.sum()
        assert missing.area.sum() < 1e-4, missing.area.sum()
        assert gaps.area.sum() < 1e-4, gaps.area.sum()

        print(
            "snapped",
            gaps.area.sum(),
            double.area.sum(),
            missing.area.sum(),
        )

        cleaned_and_snapped_to_mask = sg.snap_to_mask(
            cleaned, tolerance, mask=kommune_utenhav
        )

        # cleaned = sg.coverage_clean(
        #     sg.sort_large_first(df), tolerance, mask=kommune_utenhav
        # ).pipe(sg.snap_polygons, 0.1, mask=kommune_utenhav)

        # allow edge cases
        cleaned_and_snapped_to_mask = sg.clean_clip(
            cleaned_and_snapped_to_mask, bbox.buffer(-tolerance * 1.1)
        )

        gaps = sg.get_gaps(cleaned_and_snapped_to_mask)
        double = sg.get_intersections(cleaned_and_snapped_to_mask)
        missing = get_missing(
            sg.clean_clip(df, bbox.buffer(-tolerance * 1.1)),
            cleaned_and_snapped_to_mask,
        )

        print(
            "cleaned_and_snapped_to_mask",
            gaps.area.sum(),
            double.area.sum(),
            missing.area.sum(),
        )

        sg.explore(
            df,  # =cleaned.assign(wkt=lambda x: x.geometry.to_wkt()),
            snapped_to_mask,  # =cleaned.assign(wkt=lambda x: x.geometry.to_wkt()),
            cleaned,  # =cleaned.assign(wkt=lambda x: x.geometry.to_wkt()),
            double,  # =double.assign(wkt=lambda x: x.to_wkt()),
            gaps,  # =gaps.assign(wkt=lambda x: x.to_wkt()),
            missing,
            kommune_utenhav,
            thick_missing=missing[~missing.buffer(-0.01).is_empty].assign(
                area=lambda x: x.area
            ),
        )

        assert list(sorted(cleaned.columns)) == cols, list(sorted(cleaned.columns))
        assert thick_df_indices.isin(
            cleaned_and_snapped_to_mask["df_idx"]
        ).all(), sg.explore(
            df,
            snapped_to_mask,
            missing_polygons=df[
                (df["df_idx"].isin(thick_df_indices))
                & (~df["df_idx"].isin(cleaned_and_snapped_to_mask["df_idx"]))
            ],
        )

        assert double.area.sum() < 1e-4, double.area.sum()
        assert gaps.area.sum() < 1e-3, (
            gaps.area.sum(),
            gaps.area.max(),
            gaps.area,
            gaps,
        )
        assert missing.area.sum() < 1e-3 or (missing.area.sum() + 1e-10) < 1e-3, (
            missing.area.sum(),
            missing.area.sort_values(),
        )

        # assert int(cleaned.area.sum()) == 154240, (
        #     cleaned.area.sum(),
        #     f"tolerance: {tolerance}",
        # )

        assert sg.get_geom_type(cleaned) == "polygon", sg.get_geom_type(cleaned)

        print("cleaning cleaned")
        cleaned_and_snapped_to_mask = sg.snap_polygons(
            cleaned, tolerance, mask=kommune_utenhav
        )

        gaps = sg.get_gaps(cleaned_and_snapped_to_mask)
        double = sg.get_intersections(cleaned_and_snapped_to_mask)
        missing = get_missing(sg.clean_clip(df, bbox.buffer(-tolerance * 1.1)), cleaned)

        sg.explore(
            df,  # =cleaned.assign(wkt=lambda x: x.geometry.to_wkt()),
            cleaned,  # =cleaned.assign(wkt=lambda x: x.geometry.to_wkt()),
            double,  # =double.assign(wkt=lambda x: x.to_wkt()),
            gaps,  # =gaps.assign(wkt=lambda x: x.to_wkt()),
            missing,
            kommune_utenhav,
            thick_missing=missing[~missing.buffer(-0.01).is_empty].assign(
                area=lambda x: x.area
            ),
            # mask=sg.to_gdf([5.36750884, 59.00830496], 4326).to_crs(25833).buffer(10),
            # browser=True,
            max_zoom=50,
        )

        assert (
            list(sorted(cleaned_and_snapped_to_mask.columns)) == cols
        ), cleaned_and_snapped_to_mask.columns

        assert double.area.sum() < 1e-3, double.area.sum()
        assert gaps.area.sum() < 1e-3, gaps.area.sum()
        assert missing.area.sum() < 1e-3 or (missing.area.sum() + 1e-10), (
            missing.area.sum(),
            missing.area.sort_values(),
        )

        # assert int(cleaned_and_snapped_to_mask.area.sum()) == 154240, (
        #     cleaned_and_snapped_to_mask.area.sum(),
        #     f"tolerance: {tolerance}",
        # )

        assert (
            sg.get_geom_type(cleaned_and_snapped_to_mask) == "polygon"
        ), sg.get_geom_type(cleaned_and_snapped_to_mask)

        # cleaned = sg.coverage_clean(df, tolerance)

        # gaps = sg.get_gaps(cleaned)
        # double = sg.get_intersections(cleaned)
        # missing = sg.clean_overlay(df, cleaned, how="difference", geom_type="polygon")

        # assert list(sorted(cleaned.columns)) == cols, list(sorted(cleaned.columns))

        # assert double.area.sum() < 1e-4, double.area.sum()
        # assert gaps.area.sum() < 1e-3, (
        #     gaps.area.sum(),
        #     gaps.area.max(),
        #     gaps.area,
        #     gaps,
        # )
        # assert missing.area.sum() < 1e-3, (
        #     missing.area.sum(),
        #     missing.area.sort_values(),
        # )


def get_missing(df, other):
    return (
        sg.clean_overlay(df, other, how="difference", geom_type="polygon")
        .pipe(sg.buff, -0.0001)
        .pipe(sg.clean_overlay, other, how="difference", geom_type="polygon")
    )


def test_clean():
    df = gpd.read_parquet(Path(__file__).parent / "testdata" / "polygon_snap.parquet")

    bbox = sg.to_gdf(minimum_rotated_rectangle(df.unary_union), df.crs)

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

    for tolerance in [5, 10]:
        print("tolerance:", tolerance)

        # from shapely import segmentize

        # df.geometry = segmentize(df.geometry, tolerance)

        snapped = sg.coverage_clean(df, tolerance).pipe(sg.coverage_clean, tolerance)
        assert sg.get_geom_type(snapped) == "polygon", sg.get_geom_type(snapped)

        double = sg.get_intersections(snapped).loc[lambda x: ~x.buffer(-1e-9).is_empty]
        gaps = sg.get_gaps(snapped).loc[lambda x: ~x.buffer(-1e-9).is_empty]
        missing = get_missing(df, snapped)

        print(double.area.sum(), missing.area.sum(), gaps.area.sum())

        sg.explore(
            df,
            snapped,
            double,
            missing,
            gaps,
        )

        assert (a := max(list(double.area) + [0])) < 1e-4, a
        assert (a := max(list(missing.area) + [0])) < 1e-4, a
        assert (a := max(list(gaps.area) + [0])) < 1e-4, a

        snapped = sg.snap_polygons(
            df, tolerance, mask=mask.buffer(0.1, resolution=1, join_style=2)
        )
        assert sg.get_geom_type(snapped) == "polygon", sg.get_geom_type(snapped)

        continue

        double = sg.get_intersections(snapped).loc[lambda x: ~x.buffer(-1e-9).is_empty]
        gaps = sg.get_gaps(snapped).loc[lambda x: ~x.buffer(-1e-9).is_empty]
        missing = get_missing(df, snapped)

        print(double.area.sum(), missing.area.sum(), gaps.area.sum())

        sg.explore(
            df,
            snapped,
            double,
            missing,
            gaps,
        )

        assert (a := max(list(gaps.area) + [0])) < 1e-4, a
        assert (a := max(list(double.area) + [0])) < 1e-4, a
        assert (a := max(list(missing.area) + [0])) < 1e-4, a

        # snapped = sg.snap_to_mask(
        #     df, tolerance, mask=mask.buffer(0.1, resolution=1, join_style=2)
        # )

        # double = sg.get_intersections(snapped)
        # missing = sg.clean_overlay(df, snapped, how="difference", geom_type="polygon")
        # gaps = sg.get_gaps(snapped)

        # print(double.area.sum(), missing.area.sum(), gaps.area.sum())

        # sg.explore(
        #     df,

        #     snapped,
        #     double,

        #     missing,
        # )

        snapped_twice = sg.snap_polygons(snapped, tolerance)
        assert sg.get_geom_type(snapped_twice) == "polygon", sg.get_geom_type(
            snapped_twice
        )

        double = sg.get_intersections(snapped_twice)
        missing = get_missing(df, snapped_twice)

        sg.explore(
            df,
            snapped,
            snapped_twice,
            double,
            missing,
        )

        # assert (a := max(list(double.area) + [0])) < 1e-4, a
        # assert (a := max(list(missing.area) + [0])) < 1e-4, a

        snapped = sg.snap_polygons(
            df, tolerance, mask=mask.buffer(0.1, resolution=1, join_style=2)
        )

        sg.explore(snapped.assign(idx=lambda x: [str(i) for i in range(len(x))]), "idx")

        gaps = sg.get_gaps(snapped)
        double = sg.get_intersections(snapped)
        missing = get_missing(df, snapped)

        sg.explore(
            df,
            snapped,
            gaps,
            double,
            missing,
        )

        sg.explore(snapped.assign(idx=lambda x: [str(i) for i in range(len(x))]), "idx")

        assert sg.get_geom_type(snapped) == "polygon", sg.get_geom_type(snapped)
        assert (a := max(list(gaps.area) + [0])) < 1e-4, a
        assert (a := max(list(double.area) + [0])) < 1e-4, a
        assert (a := max(list(missing.area) + [0])) < 1e-4, a

        snapped_to_snapped = sg.snap_to_mask(snapped, tolerance, mask=snapped)

        sg.explore(
            df,
            snapped,
            snapped_to_snapped,
        )
        sg.explore(
            snapped.assign(idx=lambda x: [str(i) for i in range(len(x))]),
            "idx",
        )

        assert sg.get_geom_type(snapped_to_snapped) == "polygon", sg.get_geom_type(
            snapped_to_snapped
        )

        # assert (nums1 := [round(num, 3) for num in sorted(snapped.area)]) == (
        #     nums2 := [round(num, 3) for num in sorted(snapped_to_snapped.area)]
        # ), (nums1, nums2)


def not_test_spikes():
    from shapely.geometry import Point, Polygon

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
                (-0.99 * factor, 2 * factor),
                (-0.99 * factor, 1.5 * factor),
                (-1.01 * factor, 1.5 * factor),
                (-1.01 * factor, 2 * factor),
                (-1.51 * factor, 2 * factor),
                (-1.51 * factor, 1.7 * factor),
                (-1.52 * factor, 2 * factor),
                (-2 * factor, 2 * factor),
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
                (0, 2 * factor),
                (-0.99 * factor, 2 * factor),
                (-0.99 * factor, 1.5 * factor),
                (-1.01 * factor, 1.5 * factor),
                (-1.01 * factor, 2 * factor),
                (-2 * factor, 2 * factor),
                (-2 * factor, 6 * factor),
                (0, 6 * factor),
                (0, 2 * factor),
            ],
        )
    ).assign(what="small", num=2)

    df = pd.concat([sliver, poly_with_spike, poly_filling_the_spike])
    holes = sg.buff(
        sg.to_gdf([(-0.84 * factor, 3 * factor), (-0.84 * factor, 4.4 * factor)]),
        [0.4 * factor, 0.3 * factor],
    )
    df = sg.clean_overlay(df, holes, how="update")
    df.crs = 25833

    tolerance = 0.09 * factor

    snapped = sg.snap_polygons(df, tolerance)
    spikes_removed = sg.remove_spikes(df, tolerance)
    spikes_fixed = sg.split_spiky_polygons(df, tolerance)
    fixed_and_cleaned = sg.coverage_clean(
        spikes_fixed, tolerance  # , pre_dissolve_func=_buff
    )  # .pipe(sg.remove_spikes, tolerance / 100)

    if __name__ == "__main__":
        sg.explore(
            fixed_and_cleaned=fixed_and_cleaned,
            snapped=snapped,
            spikes_removed=spikes_removed,
            # spikes_fixed=spikes_fixed,
            df=df,
        )

    def is_close_enough(num1, num2):
        if num1 >= num2 - 1e-3 and num1 <= num2 + 1e-3:
            return True
        return False

    area_should_be = [
        725264293.6535025,
        20000000.0,
        190000000.0,
        48285369.993336275,
        26450336.353161283,
    ]
    print(list(fixed_and_cleaned.area))
    for area1, area2 in zip(
        sorted(fixed_and_cleaned.area),
        sorted(area_should_be),
    ):
        assert is_close_enough(area1, area2), (area1, area2)

    length_should_be = [
        163423.91054766334,
        40199.502484483564,
        68384.02248970368,
        24882.8908851665,
        18541.01966249684,
    ]

    print(list(fixed_and_cleaned.length))
    for length1, length2 in zip(
        sorted(fixed_and_cleaned.length),
        sorted(length_should_be),
    ):
        assert is_close_enough(length1, length2), (length1, length2)

    # cleaned = sg.coverage_clean(df, tolerance)
    # if __name__ == "__main__":
    #     sg.explore(
    #         cleaned=cleaned,
    #         df=df,
    #     )

    # assert (area := sorted([round(x, 3) for x in cleaned.area])) == sorted(

    #     [

    #         7.225,
    #         1.89,
    #         0.503,
    #         0.283,
    #         0.2,
    #     ]
    # ), area
    # assert (length := sorted([round(x, 3) for x in cleaned.length])) == sorted(
    #     [
    #         17.398,
    #         7.838,
    #         2.513,
    #         1.885,

    #         4.02,
    #     ]

    # ), length


def main():
    test_clean()
    test_clean_1144()
    test_clean_dissappearing_polygon()

    not_test_spikes()


if __name__ == "__main__":
    import cProfile

    # cProfile.run("main()", sort="cumtime")

    main()
