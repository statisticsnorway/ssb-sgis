# %%

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon


src = str(Path(__file__).parent.parent) + "/src"

sys.path.insert(0, src)

import sgis as sg


def test_random():
    for i in [1, 10, 100]:
        gdf = sg.random_points(i, loc=100)
        assert len(gdf) == i

        buffered = sg.buff(gdf, 10)

        points = sg.random_points_in_polygons(buffered, 100)

        assert len(points) == 100 * i, points
        assert max(points.index) == i - 1, points.index

        points["temp_idx"] = range(len(points))
        joined = points.sjoin(buffered, how="inner")
        assert all(points.temp_idx.isin(joined.temp_idx))

        if i == 10:
            buffered.index = np.random.choice([0, 1, 2], i)
            points = sg.random_points_in_polygons(buffered, 100)
            assert all(points.index.isin([0, 1, 2]))

            points["temp_idx"] = range(len(points))
            joined = points.sjoin(buffered, how="inner")
            assert all(points.temp_idx.isin(joined.temp_idx))


def test_area(gdf_fixture):
    copy = sg.buffdissexp(gdf_fixture, 25)
    assert len(copy) == 4
    assert round(copy.area.sum(), 5) == 1035381.10389
    assert round(copy.length.sum(), 5) == 16689.46148


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
    gdf = sg.clean_geoms(problematic_geometries)
    gdf = sg.to_single_geom_type(gdf, "polygon")
    print(gdf)
    assert len(gdf) == 1
    assert sg.get_geom_type(gdf) == "polygon"
    ser = sg.clean_geoms(problematic_geometries.geometry)
    ser = sg.to_single_geom_type(ser, "polygon")
    assert len(ser) == 1
    assert sg.get_geom_type(ser) == "polygon"

    valid_geometry = sg.to_gdf([0, 1])
    problematic_geometries = pd.concat(
        [problematic_geometries, valid_geometry], ignore_index=True
    )
    gdf = sg.clean_geoms(problematic_geometries)
    assert len(gdf) == 2
    gdf = sg.to_single_geom_type(gdf, "polygon")
    print(gdf)
    assert sg.get_geom_type(gdf) == "polygon"
    assert len(gdf) == 1


def test_to_multipoint(gdf_fixture):
    mp = sg.to_multipoint(gdf_fixture)
    assert mp.length.sum() == 0
    mp = sg.to_multipoint(gdf_fixture.geometry)
    assert mp.length.sum() == 0
    mp = sg.to_multipoint(gdf_fixture.unary_union)
    assert mp.length == 0


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
