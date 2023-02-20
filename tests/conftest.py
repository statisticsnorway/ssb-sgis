import geopandas as gpd
import pytest
from shapely.wkt import loads

from gis_utils import to_gdf


@pytest.fixture(scope="module")
def gdf_fixture() -> gpd.GeoDataFrame:
    """Calling the make_gdf function here so that make_gdf can be
    imported when running test outside of pytest."""
    return make_gdf()


def make_gdf() -> gpd.GeoDataFrame:
    """Making a small geodataframe of points, a line and a polygon."""

    xs = [
        10.7497196,
        10.7484624,
        10.7480624,
        10.7384624,
        10.7374624,
        10.7324624,
        10.7284624,
    ]
    ys = [
        59.9281407,
        59.9275268,
        59.9272268,
        59.9175268,
        59.9165268,
        59.9365268,
        59.9075268,
    ]
    points = [f"POINT ({x} {y})" for x, y in zip(xs, ys)]

    line = [
        "LINESTRING (10.7284623 59.9075267, 10.7184623 59.9175267, 10.7114623 59.9135267, 10.7143623 59.8975267, 10.7384623 59.900000, 10.720000 59.9075200)"
    ]

    polygon = [
        "POLYGON ((10.74 59.92, 10.735 59.915, 10.73 59.91, 10.725 59.905, 10.72 59.9, 10.72 59.91, 10.72 59.91, 10.74 59.92))"
    ]

    geoms = [loads(x) for x in points + line + polygon]

    gdf = gpd.GeoDataFrame(
        {"geometry": gpd.GeoSeries(geoms)}, geometry="geometry", crs=4326
    ).to_crs(25833)

    gdf2 = to_gdf(geoms, crs=4326).to_crs(25833)

    assert gdf.equals(
        gdf2
    ), "to_gdf does not give same results as manual gdf constructor"

    gdf["numcol"] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    gdf["txtcol"] = [*"aaaabbbcc"]

    assert len(gdf) == 9, "wrong number of rows"

    x = round(gdf.dissolve().centroid.x.iloc[0], 5)
    y = round(gdf.dissolve().centroid.y.iloc[0], 5)
    assert (
        f"POINT ({x} {y})" == "POINT (261106.48628 6649101.81219)"
    ), "wrong centerpoints. Have the testdata changed?"

    return gdf
