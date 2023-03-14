from geopandas import GeoDataFrame

from sgis import read_parquet_url


def test_read_parquet_url():
    points = read_parquet_url(
        "https://media.githubusercontent.com/media/"
        "statisticsnorway/ssb-sgis/main/tests/testdata/points_oslo.parquet"
    )
    assert isinstance(points, GeoDataFrame)
