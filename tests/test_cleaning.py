# %%

import random
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from geopandas import GeoDataFrame
from shapely import extract_unique_points
from shapely import get_coordinates
from shapely import get_parts
from shapely.geometry import Polygon

src = str(Path(__file__).parent).replace("tests", "") + "src"

sys.path.insert(0, src)

import sgis as sg

# def no_explore(*args, **kwargs):
#     pass

# sg.explore = no_explore


@pytest.mark.skip(reason="This test fails, need to investigate")
def test_clean_closing_hole():
    df = sg.to_gdf(
        [
            "POLYGON ((375502.75 7490105.760000002, 375502.33999999985 7490104.309999999, 375506.5800000001 7490099.3500000015, 375512.46999999974 7490096.059999999, 375521.4299999997 7490090.170000002, 375528.5 7490089.460000001, 375537.45999999996 7490085.449999999, 375551.8300000001 7490087.809999999, 375563.8499999996 7490090.870000001, 375570.9299999997 7490095.120000001, 375576.3300000001 7490095.109999999, 375577.29000000004 7490091.34, 375575.63999999966 7490085.449999999, 375565.98000000045 7490075.789999999, 375556.3099999996 7490071.079999998, 375545 7490069.8999999985, 375539.33999999985 7490067.07, 375529.91000000015 7490063.530000001, 375520.01999999955 7490061.18, 375507.51999999955 7490057.640000001, 375495.5 7490057.170000002, 375483.71999999974 7490061.6499999985, 375478.0599999996 7490067.780000001, 375469.5800000001 7490072.460000001, 375462.5099999998 7490076.969999999, 375461.3300000001 7490082.859999999, 375457.5599999996 7490085.449999999, 375452.83999999985 7490094.41, 375458.5 7490101.949999999, 375468.63999999966 7490107.140000001, 375485.3700000001 7490114.41, 375493.8499999996 7490109.949999999, 375502.3300000001 7490106.170000002, 375502.6299999999 7490105.829999998, 375502.75 7490105.760000002))",
            "POLYGON ((375485.3700000001 7490114.41, 375468.63999999966 7490107.140000001, 375458.5 7490101.949999999, 375452.83999999985 7490094.41, 375457.5599999996 7490085.449999999, 375461.3300000001 7490082.859999999, 375462.5099999998 7490076.969999999, 375469.5800000001 7490072.460000001, 375478.0599999996 7490067.780000001, 375483.71999999974 7490061.6499999985, 375495.5 7490057.170000002, 375507.51999999955 7490057.640000001, 375520.01999999955 7490061.18, 375529.91000000015 7490063.530000001, 375539.33999999985 7490067.07, 375545 7490069.8999999985, 375556.3099999996 7490071.079999998, 375565.98000000045 7490075.789999999, 375575.63999999966 7490085.449999999, 375577.29000000004 7490091.34, 375576.3300000001 7490095.109999999, 375570.9299999997 7490095.120000001, 375563.8499999996 7490090.870000001, 375551.8300000001 7490087.809999999, 375537.45999999996 7490085.449999999, 375528.5 7490089.460000001, 375521.4299999997 7490090.170000002, 375512.46999999974 7490096.059999999, 375506.5800000001 7490099.3500000015, 375502.33999999985 7490104.309999999, 375502.75 7490105.760000002, 375505.8300000001 7490103.870000001, 375512.33999999985 7490102.059999999, 375516.4199999999 7490101.1000000015, 375517.78000000026 7490100.6000000015, 375520.5 7490099, 375522.16000000015 7490097.379999999, 375524.46999999974 7490095.800000001, 375527.1699999999 7490095.09, 375534.2000000002 7490091.039999999, 375535.98000000045 7490090.68, 375540.20999999996 7490091.469999999, 375543.11000000034 7490092.719999999, 375548.86000000034 7490096.239999998, 375551.5499999998 7490097.219999999, 375558.2599999998 7490097.890000001, 375562.33999999985 7490098.16, 375569.03000000026 7490099.129999999, 375572.53000000026 7490100.050000001, 375579.33999999985 7490100.890000001, 375581.4500000002 7490101.280000001, 375582.8099999996 7490102.010000002, 375587.2000000002 7490103.41, 375590.1799999997 7490100.109999999, 375588.5700000003 7490098.620000001, 375587.29000000004 7490097.800000001, 375590.2400000002 7490095.93, 375592.7000000002 7490093.32, 375589.0099999998 7490090.07, 375584.58999999985 7490088.18, 375583.0099999998 7490087.260000002, 375584.5800000001 7490085.48, 375584.98000000045 7490084.68, 375583 7490083.780000001, 375584.26999999955 7490082.59, 375584.4400000004 7490081.969999999, 375580.13999999966 7490076.879999999, 375574.25 7490073.239999998, 375572.01999999955 7490072.199999999, 375566.3099999996 7490068.719999999, 375562.08999999985 7490066.77, 375556.9400000004 7490065.010000002, 375558.73000000045 7490062.98, 375557.9400000004 7490062.039999999, 375555.5099999998 7490060.6000000015, 375550.6799999997 7490059.890000001, 375544.70999999996 7490057.57, 375538.25 7490055.109999999, 375528.8099999996 7490052.280000001, 375522.13999999966 7490051.640000001, 375515.29000000004 7490050.77, 375508.54000000004 7490050.219999999, 375501.98000000045 7490049.530000001, 375500.51999999955 7490049.710000001, 375494.21999999974 7490051.710000001, 375491.7599999998 7490052.309999999, 375490.1500000004 7490053.359999999, 375484 7490056.489999998, 375479.96999999974 7490058.800000001, 375477.5700000003 7490060.920000002, 375474.98000000045 7490063.949999999, 375469.9199999999 7490067.43, 375467.2000000002 7490070.43, 375464.79000000004 7490072.460000001, 375460.76999999955 7490074.77, 375459.3499999996 7490075.760000002, 375457.3200000003 7490077.8999999985, 375456.73000000045 7490079.530000001, 375456.73000000045 7490080.1499999985, 375456.0599999996 7490081.170000002, 375454.7000000002 7490081.670000002, 375452.7999999998 7490082.77, 375449.29000000004 7490086.07, 375448.5999999996 7490087.539999999, 375448.5700000003 7490088.98, 375449.3499999996 7490092.82, 375448.5499999998 7490095.120000001, 375448.5099999998 7490095.699999999, 375449.11000000034 7490096.77, 375453.6799999997 7490101.629999999, 375458.9299999997 7490106.25, 375462.5800000001 7490108.84, 375467.8099999996 7490111.289999999, 375468.96999999974 7490111.550000001, 375475.2400000002 7490114.219999999, 375481.6200000001 7490116.510000002, 375487.6900000004 7490119.239999998, 375493.9000000004 7490122.32, 375499.2000000002 7490124.440000001, 375507.5499999998 7490120.539999999, 375509.33999999985 7490119.120000001, 375512.2599999998 7490117.379999999, 375512.7599999998 7490116.890000001, 375512.8099999996 7490116.3999999985, 375512.3300000001 7490115.8999999985, 375509.5800000001 7490115.260000002, 375503.36000000034 7490112.710000001, 375501.8300000001 7490111.8999999985, 375501.0999999996 7490111.25, 375500.8099999996 7490110.530000001, 375501.79000000004 7490106.789999999, 375502.3300000001 7490106.170000002, 375493.8499999996 7490109.949999999, 375485.3700000001 7490114.41))",
        ],
        25833,
    )

    df["df_index"] = range(len(df))

    mask = sg.to_gdf(
        [
            "POLYGON ((375540.20999999996 7490091.469999999, 375543.11000000034 7490092.719999999, 375548.86000000034 7490096.239999998, 375551.5499999998 7490097.219999999, 375558.2599999998 7490097.890000001, 375562.33999999985 7490098.16, 375569.03000000026 7490099.129999999, 375572.53000000026 7490100.050000001, 375579.33999999985 7490100.890000001, 375581.4500000002 7490101.280000001, 375582.8099999996 7490102.010000002, 375587.2000000002 7490103.41, 375590.1799999997 7490100.109999999, 375588.5700000003 7490098.620000001, 375587.29000000004 7490097.800000001, 375590.2400000002 7490095.93, 375592.7000000002 7490093.32, 375589.0099999998 7490090.07, 375584.58999999985 7490088.18, 375583.0099999998 7490087.260000002, 375584.5800000001 7490085.48, 375584.98000000045 7490084.68, 375583 7490083.780000001, 375584.26999999955 7490082.59, 375584.4400000004 7490081.969999999, 375580.13999999966 7490076.879999999, 375574.25 7490073.239999998, 375572.01999999955 7490072.199999999, 375566.3099999996 7490068.719999999, 375562.08999999985 7490066.77, 375556.9400000004 7490065.010000002, 375558.73000000045 7490062.98, 375557.9400000004 7490062.039999999, 375555.5099999998 7490060.6000000015, 375550.6799999997 7490059.890000001, 375544.70999999996 7490057.57, 375538.25 7490055.109999999, 375528.8099999996 7490052.280000001, 375522.13999999966 7490051.640000001, 375515.29000000004 7490050.77, 375508.54000000004 7490050.219999999, 375501.98000000045 7490049.530000001, 375500.51999999955 7490049.710000001, 375494.21999999974 7490051.710000001, 375491.7599999998 7490052.309999999, 375490.1500000004 7490053.359999999, 375484 7490056.489999998, 375479.96999999974 7490058.800000001, 375477.5700000003 7490060.920000002, 375474.98000000045 7490063.949999999, 375469.9199999999 7490067.43, 375467.2000000002 7490070.43, 375464.79000000004 7490072.460000001, 375460.76999999955 7490074.77, 375459.3499999996 7490075.760000002, 375457.3200000003 7490077.8999999985, 375456.73000000045 7490079.530000001, 375456.73000000045 7490080.1499999985, 375456.0599999996 7490081.170000002, 375454.7000000002 7490081.670000002, 375452.7999999998 7490082.77, 375449.29000000004 7490086.07, 375448.5999999996 7490087.539999999, 375448.5700000003 7490088.98, 375449.3499999996 7490092.82, 375448.5499999998 7490095.120000001, 375448.5099999998 7490095.699999999, 375449.11000000034 7490096.77, 375453.6799999997 7490101.629999999, 375458.9299999997 7490106.25, 375462.5800000001 7490108.84, 375467.8099999996 7490111.289999999, 375468.96999999974 7490111.550000001, 375475.2400000002 7490114.219999999, 375481.6200000001 7490116.510000002, 375487.6900000004 7490119.239999998, 375493.9000000004 7490122.32, 375499.2000000002 7490124.440000001, 375507.5499999998 7490120.539999999, 375509.33999999985 7490119.120000001, 375512.2599999998 7490117.379999999, 375512.7599999998 7490116.890000001, 375512.8099999996 7490116.3999999985, 375512.3300000001 7490115.8999999985, 375509.5800000001 7490115.260000002, 375503.36000000034 7490112.710000001, 375501.8300000001 7490111.8999999985, 375501.0999999996 7490111.25, 375500.8099999996 7490110.530000001, 375501.79000000004 7490106.789999999, 375502.6299999999 7490105.829999998, 375505.8300000001 7490103.870000001, 375512.33999999985 7490102.059999999, 375516.4199999999 7490101.1000000015, 375517.78000000026 7490100.6000000015, 375520.5 7490099, 375522.16000000015 7490097.379999999, 375524.46999999974 7490095.800000001, 375527.1699999999 7490095.09, 375534.2000000002 7490091.039999999, 375535.98000000045 7490090.68, 375540.20999999996 7490091.469999999))",
        ],
        25833,
    )

    cleaned = sg.coverage_clean(df, 0.5, mask)

    sg.explore(mask, df, cleaned)

    intersected = sg.clean_overlay(df, cleaned, how="intersection", geom_type="polygon")
    area_same_index = intersected[
        lambda x: x["df_index_1"] == x["df_index_2"]
    ].area.sum()
    area_same_index_ratio = area_same_index / intersected.area.sum()
    assert area_same_index_ratio > 0.995, area_same_index_ratio

    gaps = sg.get_gaps(cleaned)
    double = sg.get_intersections(cleaned)
    missing = get_missing(mask, cleaned)
    sg.explore(df, cleaned, gaps, missing, double)

    # check that the geometries still have same column values by ensuring that the range index is the same
    intersected = sg.clean_overlay(df, cleaned, how="intersection", geom_type="polygon")
    area_same_index = intersected[
        lambda x: x["df_index_1"] == x["df_index_2"]
    ].area.sum()

    area_same_index_ratio = area_same_index / intersected.area.sum()
    assert area_same_index_ratio > 0.999, area_same_index_ratio

    assert gaps.area.sum() == 0, f"gaps: {gaps.area.sum()}"
    assert double.area.sum() == 0, f"double: {double.area.sum()}"
    assert missing.area.sum() == 0, f"missing: {missing.area.sum()}"


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


@pytest.mark.skip(reason="This test fails, need to investigate")
def test_clean_complicated_land_use():
    for tolerance in [
        0.5,
        0.4,
        0.3,
    ]:
        print(tolerance)

        _test_clean_complicated_land_use_base(
            Path(__file__).parent / "testdata/disappearing.txt",
            None,
            tolerance=tolerance,
        )

        _test_clean_complicated_land_use_base(
            Path(__file__).parent / "testdata/roads_difficult_to_dissolve4.txt",
            "POLYGON ((-32050 6557614, -32050 6556914, -32750 6556914, -32750 6557614, -32050 6557614))",
            tolerance=tolerance,
        )

        _test_clean_complicated_land_use_base(
            Path(__file__).parent / "testdata/roads_difficult_to_dissolve3.txt",
            "POLYGON ((28120 6945720, 28120 6945020, 27420 6945020, 27420 6945720, 28120 6945720))",
            tolerance=tolerance,
        )

        _test_clean_complicated_land_use_base(
            Path(__file__).parent / "testdata/roads_difficult_to_dissolve2.txt",
            "POLYGON ((270257 6654842, 270257 6654142, 269557 6654142, 269557 6654842, 270257 6654842))",
            tolerance=tolerance,
        )

        _test_clean_complicated_land_use_base(
            Path(__file__).parent / "testdata/roads_difficult_to_dissolve.txt",
            "POLYGON ((-49922 6630166, -49922 6629466, -50622 6629466, -50622 6630166, -49922 6630166))",
            tolerance=tolerance,
        )


def test_clean_dissexp():

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


def _test_clean_complicated_land_use_base(path, mask, tolerance):

    print(path)

    with open(path) as f:
        df = sg.to_gdf(f.readlines(), 25833)

    if mask is not None:
        mask = sg.to_gdf(mask, 25833)
    df["df_index"] = range(len(df))

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

    # check that the geometries still have same column values by ensuring that the range index is the same
    intersected = sg.clean_overlay(df, cleaned, how="intersection", geom_type="polygon")
    area_same_index = intersected[
        lambda x: x["df_index_1"] == x["df_index_2"]
    ].area.sum()
    area_same_index_ratio = area_same_index / intersected.area.sum()
    assert area_same_index_ratio > 0.999, area_same_index_ratio

    assert (
        gaps.area.sum() == 0
    ), f"path: {Path(path).stem}, tolerance {tolerance}, gaps: {gaps.area.sum()}"
    assert (
        missing.area.sum() == 0
    ), f"path: {Path(path).stem}, tolerance {tolerance}, missing: {missing.area.sum()}"
    assert (
        double.area.sum() == 0
    ), f"path: {Path(path).stem}, tolerance {tolerance}, double: {double.area.sum()}"


@pytest.mark.skip(reason="This test fails, need to investigate")
def test_clean_1144():
    df = gpd.read_parquet(
        Path(__file__).parent / "testdata" / "snap_problem_area_1144.parquet"
    )

    bbox = sg.to_gdf(
        shapely.minimum_rotated_rectangle(shapely.union_all(df.geometry.values)), df.crs
    )

    kommune_utenhav = gpd.read_parquet(
        Path(__file__).parent / "testdata" / "kommune_utenhav_1144_2023.parquet"
    )

    # kommune_utenhav = sg.buff(
    #     kommune_utenhav,
    #     0.001,
    #     resolution=1,
    #     join_style=2,
    # )
    kommune_utenhav = sg.clean_clip(
        kommune_utenhav,
        bbox,
        geom_type="polygon",
    )
    kommune_utenhav_points = (
        kommune_utenhav.extract_unique_points()
        .to_frame("geometry")
        .explode()
        .assign(wkt=lambda x: x.geometry.to_wkt())
    )

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
        "df_index",
        "geometry",
        "kilde",
    ]

    df["df_index"] = range(len(df))

    for tolerance in [
        2.25,
        0.5,
        0.51,
        0.91,
        0.57,
        5,
        1,
        2,
        0.75,
        1.5,
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
            lambda x: ~x.buffer(-tolerance / 1.3).is_empty, "df_index"
        ]

        cleaned = sg.coverage_clean(df, tolerance, mask=kommune_utenhav).pipe(
            lambda x: x  # sg.coverage_clean, tolerance, mask=kommune_utenhav
        )

        # allow edge cases
        cleaned_clipped = sg.clean_clip(cleaned, bbox.buffer(-tolerance * 1.1))

        gaps = sg.get_gaps(cleaned_clipped)

        double = sg.get_intersections(cleaned_clipped)
        missing = get_missing(
            # kommune_utenhav, cleaned
            sg.clean_clip(kommune_utenhav, bbox.buffer(-tolerance * 1.1)),
            cleaned,
            # sg.clean_clip(df, bbox.buffer(-tolerance * 1.1)), cleaned_clipped
        )

        cleaned_points = (
            cleaned.extract_unique_points()
            .to_frame("geometry")
            .explode()
            .assign(wkt=lambda x: x.geometry.to_wkt())
        )
        df_points = (
            df.extract_unique_points()
            .to_frame("geometry")
            .explode()
            .assign(wkt=lambda x: x.geometry.to_wkt())
        )

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
            center=(-52074.0241, 6580847.4464, 0.1),
            max_zoom=40,
        )
        sg.explore(
            cleaned,
            gaps,
            double,
            missing,
            df,
            kommune_utenhav,
            center=(5.38389153, 59.00548223, 1),
            max_zoom=40,
        )

        sg.explore(
            cleaned,
            gaps,
            double,
            missing,
            df,
            kommune_utenhav,
            cleaned_points,
            kommune_utenhav_points,
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
            missing.area.sum() <= 1e-6
        ), f"tolerance {tolerance}, missing: {missing.area.sum()}"

        assert thick_df_indices.isin(cleaned_clipped["df_index"]).all(), sg.explore(
            df,
            cleaned,
            missing_polygons=df[
                (df["df_index"].isin(thick_df_indices))
                & (~df["df_index"].isin(cleaned_clipped["df_index"]))
            ],
        )

        intersected = sg.clean_overlay(
            df, cleaned, how="intersection", geom_type="polygon"
        )
        area_same_index = intersected[
            lambda x: x["df_index_1"] == x["df_index_2"]
        ].area.sum()
        area_same_index_ratio = area_same_index / intersected.area.sum()
        assert area_same_index_ratio > 0.998 - tolerance * 0.035, area_same_index_ratio

        notna_df: pd.DataFrame = df.notna().all()
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
        .pipe(sg.sfilter_inverse, other.buffer(-0.002))
        .pipe(sg.sfilter_inverse, other.buffer(-0.003))
        .pipe(sg.sfilter_inverse, other.buffer(-0.004))
        .pipe(
            sg.buff,
            -0.001,
            resolution=1,
            join_style=2,
        )
        .pipe(
            sg.buff,
            0.001,
            resolution=1,
            join_style=2,
        )
        .pipe(sg.clean_geoms)
    )


@pytest.mark.skip(reason="This test fails, need to investigate")
def test_clean():

    df = gpd.read_parquet(Path(__file__).parent / "testdata" / "polygon_snap.parquet")

    bbox = sg.to_gdf(
        shapely.minimum_rotated_rectangle(shapely.union_all(df.geometry.values)), df.crs
    )

    kommune_utenhav = gpd.read_parquet(
        Path(__file__).parent / "testdata" / "kommune_utenhav_5435_2023.parquet"
    )
    kommune_utenhav = sg.clean_clip(kommune_utenhav, bbox, geom_type="polygon")

    holes = sg.to_gdf(
        [
            "POINT (905200 7878700)",
            # "POINT (905250 7878780)",
            "POINT (905275 7878800)",
            "POINT (905242.961 7878773.758)",
        ],
        25833,
    ).pipe(sg.buff, 3)

    df = sg.clean_overlay(df, holes, how="difference")
    df["df_index"] = range(len(df))

    mask: GeoDataFrame = sg.close_all_holes(
        sg.dissexp_by_cluster(df[["geometry"]])
    ).pipe(sg.make_all_singlepart)
    mask = GeoDataFrame(
        {
            "geometry": [
                mask.union_all()
                .buffer(
                    1e-3,
                    resolution=1,
                    join_style=2,
                )
                .buffer(
                    1e-3,
                    resolution=1,
                    join_style=2,
                )
            ]
        },
        crs=df.crs,
    ).pipe(sg.make_all_singlepart)

    for tolerance in [10, 9, 8, 7, 6, 5]:  # 5, 7, 6, 8, 9]:
        print("tolerance:", tolerance)

        cleaned = sg.coverage_clean(df, tolerance, mask=mask)
        assert sg.get_geom_type(cleaned) == "polygon", sg.get_geom_type(cleaned)

        double = sg.get_intersections(cleaned).loc[lambda x: ~x.buffer(-1e-9).is_empty]
        gaps = sg.get_gaps(cleaned).loc[lambda x: ~x.buffer(-1e-9).is_empty]
        missing = get_missing(df, cleaned).dissolve()[
            lambda x: x.buffer(-tolerance / 2).is_empty
        ]

        print(
            f"tolerance: {tolerance}, double: {double.area.sum()}, "
            f"missing: {missing.area.sum()}, gaps: {gaps.area.sum()}"
        )

        sg.explore(
            df=df.to_crs(25833),
            cleaned=cleaned.to_crs(25833),
            double=double.to_crs(25833),
            missing=missing,
            gaps=gaps.to_crs(25833),
            points=sg.to_gdf(extract_unique_points(cleaned.geometry).explode()),
        )

        assert (
            a := max(list(double.area) + [0])
        ) < 1e-5, f"tolerance: {tolerance}, double: {a}"
        assert (
            a := max(list(missing.area) + [0])
        ) < 1e-5, f"tolerance: {tolerance}, missing: {a}. {missing.area.sort_values()}"
        assert (
            a := max(list(gaps.area) + [0])
        ) < 1e-5, f"tolerance: {tolerance}, gaps: {a}"

        notna_cleaned = cleaned[df.columns].notna().all()
        notna_df = df.notna().all()
        assert notna_cleaned.equals(notna_df), (notna_cleaned, notna_df)

        intersected = sg.clean_overlay(
            df, cleaned, how="intersection", geom_type="polygon"
        )
        area_same_index = intersected[
            lambda x: x["df_index_1"] == x["df_index_2"]
        ].area.sum()
        area_same_index_ratio = area_same_index / intersected.area.sum()
        assert area_same_index_ratio > 0.995, area_same_index_ratio

        cleaned_again = sg.coverage_clean(cleaned, tolerance, mask=mask)
        assert sg.get_geom_type(cleaned_again) == "polygon", sg.get_geom_type(
            cleaned_again
        )

        double = sg.get_intersections(cleaned_again).loc[
            lambda x: ~x.buffer(-1e-9).is_empty
        ]
        gaps = sg.get_gaps(cleaned_again).loc[lambda x: ~x.buffer(-1e-9).is_empty]
        missing = get_missing(df, cleaned_again).dissolve()[
            lambda x: x.buffer(-tolerance / 2).is_empty
        ]

        print(
            f"tolerance: {tolerance}, double: {double.area.sum()}, "
            f"missing: {missing.area.sum()}, gaps: {gaps.area.sum()}"
        )

        sg.explore(
            df=df.to_crs(25833),
            cleaned=cleaned.to_crs(25833),
            cleaned_again=cleaned_again.to_crs(25833),
            double=double.to_crs(25833),
            missing=missing,
            gaps=gaps.to_crs(25833),
            points=sg.to_gdf(extract_unique_points(cleaned_again.geometry).explode()),
        )

        assert (
            a := max(list(double.area) + [0])
        ) < 1e-5, f"tolerance: {tolerance}, double: {a}"
        assert (
            a := max(list(missing.area) + [0])
        ) < 1e-5, f"tolerance: {tolerance}, missing: {a}. {missing.area.sort_values()}"
        assert (
            a := max(list(gaps.area) + [0])
        ) < 1e-5, f"tolerance: {tolerance}, gaps: {a}"

        notna_cleaned = cleaned_again[df.columns].notna().all()
        notna_df = df.notna().all()
        assert notna_cleaned.equals(notna_df), (notna_cleaned, notna_df)

        intersected = sg.clean_overlay(
            df, cleaned_again, how="intersection", geom_type="polygon"
        )
        area_same_index = intersected[
            lambda x: x["df_index_1"] == x["df_index_2"]
        ].area.sum()
        area_same_index_ratio = area_same_index / intersected.area.sum()
        assert area_same_index_ratio > 0.995, area_same_index_ratio

    sg.explore(
        cleaned1=sg.coverage_clean(df, 1),
        cleaned3=sg.coverage_clean(df, 3),
        cleaned5=sg.coverage_clean(df, 5),
        df=df,
    )


def not_test_spikes():

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


@pytest.mark.skip(reason="This test fails, need to investigate")
def test_snappping(_test=False):

    if _test:
        loop = np.arange(0.2, 1, 0.01)
    else:
        loop = np.arange(0.25, 0.5, 0.05)

    a = sg.to_gdf(
        Polygon(
            [
                (0, -2),
                (0, 0),
                (-0.02, 1),
                (-0.03, 1.2),
                (-0.04, 1.4),
                (-0.02, 1.6),
                (-0.1, 1.8),
                (0, 2),
                (0, 4),
                (-4, 4),
                (-4, -2),
                (0, -2),
            ]
        )
    )
    thin = sg.to_gdf(
        Polygon(
            [
                (-3, -4),
                (0, -2),
                (0, 0),
                (0.01, 1),
                (0, 4),
                (0.1, 4),
                (0.1, -2),
                (3, -4),
            ]
        )
    )
    thick = sg.to_gdf(
        Polygon(
            [
                (0.1, -2),
                (0.1001, 0),
                (0.1002, 1),
                (0.1000001, 4),
                (4, 4),
                (4, -2),
                (0.1, -2),
            ]
        )
    )

    b = sg.to_gdf(
        Polygon(
            [
                (0, -2),
                (0, 0),
                (0.1, 1),
                (0, 2),
                (0, 4),
                (4, 4),
                (4, -2),
                (0, -2),
            ]
        )
    )
    c = sg.to_gdf(
        Polygon(
            [
                (0, -2),
                # (0, 0),
                # (0, 2),
                (0, 4),
                (4, 4),
                (4, -2),
                (0, -2),
            ]
        )
    )
    for i, df in {
        "dfm1": pd.concat([a, thick, thin]),
        "df0": pd.concat([a, c]),
        "df1": pd.concat([a, b]),
        "df2": pd.concat(
            [
                a,
                c.assign(
                    geometry=lambda x: [
                        Polygon(get_coordinates(g)[::-1]) for g in x.geometry
                    ]
                ),
            ]
        ),
        "df3": pd.concat(
            [
                a,
                b.assign(
                    geometry=lambda x: [
                        Polygon(get_coordinates(g)[::-1]) for g in x.geometry
                    ]
                ),
            ]
        ),
        "df4": pd.concat(
            [
                a.assign(
                    geometry=lambda x: [
                        Polygon(get_coordinates(g)[::-1]) for g in x.geometry
                    ]
                ),
                c,
            ]
        ),
        "df5": pd.concat(
            [
                a.assign(
                    geometry=lambda x: [
                        Polygon(get_coordinates(g)[::-1]) for g in x.geometry
                    ]
                ),
                b,
            ]
        ),
    }.items():
        # if i != "df5":
        #     continue

        print(i)
        df["idx"] = [str(x) for x in range(len(df))]
        p = (
            sg.to_gdf(extract_unique_points(df.geometry.values))
            .explode()
            .assign(wkt=lambda x: [g.wkt for g in x.geometry])
        )

        for tolerance in loop:
            # if i != "df1" or tolerance != 0.25:
            #     continue

            print(tolerance)
            cleaned = sg.coverage_clean(df, tolerance=tolerance)
            # cleaned = sg.coverage_clean(cleaned, tolerance=tolerance)
            gaps = sg.get_gaps(cleaned)
            double = sg.get_intersections(cleaned)
            missing = get_missing(df, cleaned)
            sg.explore(cleaned, gaps, double, missing, p)
            cleaned = pd.concat([df.assign(idx="df"), cleaned])
            # sg.explore(cleaned, column="idx")

            assert (
                gaps.area.sum() == 0
            ), f"tolerance {tolerance} {i}, gaps: {gaps.area.sum()}"
            assert double.area.sum() == 0, (
                sg.explore_locals(browser=True),
                gaps,
                missing,
                f"tolerance {tolerance} {i}, double: {double.area.sum()}",
            )

            if i != "dfm1":
                assert (
                    missing.area.sum() == 0
                ), f"tolerance {tolerance} {i}, missing: {missing.area.sum()}"


def main():

    test_clean_1144()
    test_clean()
    test_clean_dissappearing_polygon()
    test_snappping(_test=False)
    test_clean_closing_hole()
    test_clean_complicated_land_use()
    test_clean_dissexp()
    not_test_spikes()


if __name__ == "__main__":

    # df = cprofile_df("main()")
    # print(df.iloc[:50])
    # print(df.iloc[50:100])

    main()

    import cProfile

    cProfile.run("main()", sort="cumtime")
    _time = time.perf_counter()
    print("seconds passed:", time.perf_counter() - _time)
