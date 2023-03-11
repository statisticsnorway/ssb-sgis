from geopandas import GeoDataFrame

from gis_utils import read_parquet_url


def test_read_parquet_url():
    points = read_parquet_url(
        "https://media.githubusercontent.com/media/"
        "statisticsnorway/ssb-gis-utils/main/tests/testdata/random_points.parquet"
    )
    assert isinstance(points, GeoDataFrame)
