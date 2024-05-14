from pathlib import Path

from geopandas import GeoDataFrame
from geopandas import read_parquet


def points_oslo() -> GeoDataFrame:
    return read_parquet(Path(__file__).parent / "testdata" / "points_oslo.parquet")


def roads_oslo() -> GeoDataFrame:
    return read_parquet(Path(__file__).parent / "testdata" / "roads_oslo_2022.parquet")
