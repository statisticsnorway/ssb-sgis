import os

import geopandas as gpd
import numpy as np
import pandas as pd
from dapla import FileClient, details
from geopandas import GeoDataFrame
from geopandas.io.arrow import _geopandas_to_arrow
from pyarrow import parquet


def exists(path: str) -> bool:
    """Returns True if the path exists, and False if it doesn't

    Works in dapla and outside of dapla

    Args:
      path (str): The path to the file or directory.

    Returns:
      A boolean value.
    """
    try:
        details(path)
        return True
    except FileNotFoundError:
        return False
    except ModuleNotFoundError:
        return os.path.exists(path)


def read_geopandas(path: str, **kwargs) -> GeoDataFrame:
    """Reads geoparquet or other geodata* from a file on GCS

    *does not read shapelfiles or filegeodatabases.

    Args:
      path: path to a file on Google Cloud Storage

    Returns:
      A GeoDataFrame

    """

    fs = FileClient.get_gcs_file_system()

    if "parquet" in path:
        with fs.open(path, mode="rb") as file:
            return gpd.read_parquet(file, **kwargs)
    else:
        with fs.open(path, mode="rb") as file:
            return gpd.read_file(file, **kwargs)


def write_geopandas(df: gpd.GeoDataFrame, gcs_path: str, **kwargs) -> None:
    """Writes a GeoDataFrame to the speficied format.

    Does not work for .shp and .gdb.

    Args:
      df (gpd.GeoDataFrame): The GeoDataFrame to write
      gcs_path (str): The path to the file you want to write to.

    Returns:
      None
    """

    pd.io.parquet.BaseImpl.validate_dataframe(df)

    fs = FileClient.get_gcs_file_system()

    if ".parquet" in gcs_path:
        with fs.open(gcs_path, mode="wb") as buffer:
            table = _geopandas_to_arrow(df, index=df.index, schema_version=None)
            parquet.write_table(table, buffer, compression="snappy", **kwargs)
        return

    if ".gpkg" in gcs_path:
        driver = "GPKG"
    elif ".geojson" in gcs_path:
        driver = "GeoJSON"
    elif ".gml" in gcs_path:
        driver = "GML"
    elif ".shp" in gcs_path:
        driver = "ESRI Shapefile"
    else:
        driver = None

    with fs.open(gcs_path, "wb") as file:
        df.to_file(file, driver=driver)
