import geopandas as gpd
import pytest

from gis_utils import testgdf


@pytest.fixture(scope="module")
def gdf_fixture() -> gpd.GeoDataFrame:
    """Calling the testgdf function here so that testgdf can be
    imported when running test outside of pytest."""
    return testgdf()
