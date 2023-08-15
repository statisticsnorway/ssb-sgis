# %%

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from helpers import create_all_geometry_types
from shapely.geometry import Polygon


src = str(Path(__file__).parent.parent) + "/src"

sys.path.insert(0, src)

import sgis as sg


def test_get_common_crs():
    gdf = sg.to_gdf([0, 0], crs=25833)
    gdf2 = sg.to_gdf([0, 0], crs=None)

    with pytest.raises(ValueError):
        sg.get_common_crs([gdf, gdf2], strict=True)

    assert sg.get_common_crs([gdf, gdf2]) == 25833
    assert sg.get_common_crs([gdf, gdf]) == 25833
    assert sg.get_common_crs([gdf, gdf], strict=True) == 25833


def test_drop_inactive():
    gdf = sg.to_gdf([0, 0])
    gdf["geom2"] = sg.to_gdf([0, 0]).geometry
    gdf = sg.drop_inactive_geometry_columns(gdf)
    assert list(gdf.columns) == ["geometry"]


def test_rename_geometry_if():
    gdf = sg.to_gdf([0, 0])
    gdf = gdf.rename_geometry("geom2")
    gdf.columns = ["geom2"]
    gdf = sg.rename_geometry_if(gdf)
    assert list(gdf.columns) == ["geometry"], gdf

    gdf = sg.to_gdf([0, 0])
    gdf.columns = ["geom2"]
    assert list(gdf.columns) == ["geom2"], gdf
    gdf = sg.rename_geometry_if(gdf)
    assert list(gdf.columns) == ["geometry"], gdf


def test_random():
    gdf = sg.random_points("1")
    assert len(gdf) == 1
    gdf = sg.random_points(1.32323)
    assert len(gdf) == 1

    for i in [1, 10, 100]:
        gdf = sg.random_points(i, loc=100)
        assert len(gdf) == i

        sg.buff(gdf, 10)


def test_points_in_bounds():
    from shapely import box

    circle = sg.to_gdf([0, 0]).pipe(sg.buff, 1)
    box_ = sg.to_gdf(box(*circle.total_bounds))
    points = sg.points_in_bounds(circle, 100)
    assert len(points) == 10_000, len(points)

    joined = points.sjoin(box_, how="inner")
    assert len(joined) == 10_000, len(points)


def test_area():
    gdf = create_all_geometry_types()
    gdf = sg.buffdissexp(gdf, 25)
    assert round(gdf.area.sum(), 5) == 6270.69379, round(gdf.area.sum(), 5)
    assert round(gdf.length.sum(), 5) == 332.02674, round(gdf.length.sum(), 5)


def test_clean_clip():
    p = sg.random_points(100)
    buff1 = sg.buff(p, 100)
    buff2 = sg.buff(p, 200)

    clipped = buff1.clip(buff2)
    clipped["geometry"] = clipped.make_valid()
    clipped2 = sg.clean_clip(buff1, buff2)
    assert clipped.equals(clipped2)


def test_clean():
    invalid_geometry = sg.to_gdf(
        Polygon([(0, 1), (0, 0), (0, 1), (1, 1), (1, 2), (0, 2), (0, 0)])
    )

    empty_geometry = sg.to_gdf("POINT (0 0)").pipe(sg.buff, 0)

    missing_geometry = gpd.GeoDataFrame(
        {"geometry": [None]}, geometry="geometry", crs=25833
    )

    problematic_geometries = pd.concat(
        [invalid_geometry, missing_geometry, empty_geometry], ignore_index=True
    )

    assert len(problematic_geometries) == 3
    gdf = sg.clean_geoms(problematic_geometries.copy())
    print(gdf)
    assert list(gdf.index) == [0], list(gdf.index)
    gdf = sg.to_single_geom_type(gdf, "polygon")
    assert list(gdf.index) == [0], list(gdf.index)
    assert len(gdf) == 1
    assert sg.get_geom_type(gdf) == "polygon"
    ser = sg.clean_geoms(problematic_geometries.geometry.copy())
    ser = sg.to_single_geom_type(ser, "polygon")
    assert len(ser) == 1
    assert list(gdf.index) == [0], list(gdf.index)
    assert sg.get_geom_type(ser) == "polygon"

    valid_geometry = sg.to_gdf([0, 1])
    problematic_geometries = pd.concat(
        [problematic_geometries, valid_geometry], ignore_index=True
    )
    gdf = sg.clean_geoms(problematic_geometries.copy())
    assert list(gdf.index) == [0, 3], list(gdf.index)
    gdf = sg.clean_geoms(problematic_geometries.copy(), ignore_index=True)
    assert list(gdf.index) == [0, 1], list(gdf.index)
    assert len(gdf) == 2
    gdf = sg.to_single_geom_type(gdf, "polygon")
    assert sg.get_geom_type(gdf) == "polygon"
    assert len(gdf) == 1
    assert list(gdf.index) == [0], list(gdf.index)

    print(problematic_geometries)
    from shapely import is_valid, make_valid

    print(make_valid(problematic_geometries.geometry))
    problematic_geometries["geometry"] = make_valid(problematic_geometries.geometry)
    print(problematic_geometries.loc[lambda x: x["geometry"].map(bool)])
    print(problematic_geometries.loc[lambda x: x["geometry"].map(is_valid)])
    print(problematic_geometries.loc[lambda x: is_valid(x["geometry"])])


def test_sort():
    points = sg.random_points(5)
    points["idx"] = np.arange(1, 6)
    buffered = sg.buff(points, [1, 2, 3, 4, 5])
    sorted_ = sg.sort_large_first(buffered).iloc[::-1]["idx"]
    assert list(sorted_.index) == [0, 1, 2, 3, 4]
    assert list(sorted_) == [1, 2, 3, 4, 5]
    sorted_ = sg.sort_long_first(buffered).iloc[::-1]["idx"]
    assert list(sorted_.index) == [0, 1, 2, 3, 4]
    assert list(sorted_) == [1, 2, 3, 4, 5]

    sorted_ = sg.sort_large_first(buffered)["idx"]
    assert list(sorted_.index) == [4, 3, 2, 1, 0]
    assert list(sorted_) == [5, 4, 3, 2, 1]
    sorted_ = sg.sort_long_first(buffered)["idx"]
    assert list(sorted_.index) == [4, 3, 2, 1, 0]
    assert list(sorted_) == [5, 4, 3, 2, 1]

    df = sg.random_points(5)
    df.index = [0, 0, 1, 2, 3]
    df.geometry = df.buffer([1, 2, 3, 4, 5])
    df["col"] = [None, 1, 2, None, 1]
    df["col2"] = [None, 1, 2, 3, None]
    df["area"] = df.area

    df.index = [0, 0, 1, 2, 3]
    sorted1 = sg.sort_nans_last(sg.sort_large_first(df))
    df.index = [0, 1, 2, 3, 4]
    sorted2 = sg.sort_nans_last(sg.sort_large_first(df))
    assert sorted1.reset_index(drop=True).equals(sorted2.reset_index(drop=True))

    s = df.sort_values("col2")
    assert s["col2"].dropna().is_monotonic_increasing, s["col2"]

    s = sg.sort_large_first(df)
    area_is_decreasing = s["area"].is_monotonic_decreasing
    assert area_is_decreasing, s["area"]

    s = sg.sort_nans_last(sg.sort_large_first(df))
    s["n_nan"] = s.isna().sum(axis=1).sort_values()

    area_is_not_decreasing = s["area"].is_monotonic_decreasing is False
    assert area_is_not_decreasing, s["area"]

    grouped_area_is_decreasing = s.groupby("n_nan")[
        "area"
    ].is_monotonic_decreasing.all()
    assert grouped_area_is_decreasing, s["area"].tolist()


def main():
    info = """
    The test was created 08.01.2023 with the following package versions.
    From C++: GEOS 3.11.1, PROJ 9.1.0, GDAL 3.6.1.
    From Python: geopandas 0.12.2, shapely 2.0.0, pyproj 3.4.1, pandas 1.5.2, numpy 1.24.
    """
    print(info)

    print("Versjoner nå:")
    from shapely.geos import geos_version

    geos_versjon = ".".join([str(x) for x in geos_version])
    print(f"{gpd.__version__ = }")
    print(f"{geos_versjon    = }")
    print(f"{pd.__version__  = }")
    print(f"{np.__version__  = }")


if __name__ == "__main__":
    main()
    test_sort()
    test_clean()
    test_get_common_crs()
