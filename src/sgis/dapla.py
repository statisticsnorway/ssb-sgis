"""Functions for reading and writing GeoDataFrames in Statistics Norway's GCS Dapla.
"""
import os

import geopandas as gpd
import pandas as pd
from dapla import FileClient, details
from geopandas import GeoDataFrame
from geopandas.io.arrow import _geopandas_to_arrow
from pyarrow import parquet


def exists(path: str) -> bool:
    """Returns True if the path exists, and False if it doesn't.

    Works in dapla and outside of dapla.

    Args:
        path (str): The path to the file or directory.

    Returns:
        True if the path exists, False if not.
    """
    try:
        details(path)
        return True
    except FileNotFoundError:
        return False
    except ModuleNotFoundError:
        return os.path.exists(path)


def read_geopandas(path: str, **kwargs) -> GeoDataFrame:
    """Reads geoparquet or other geodata from a file on GCS.

    Note:
        Does not currently read shapefiles or filegeodatabases.

    Args:
        path: path to a file on Google Cloud Storage.
        **kwargs: Additional keyword arguments passed to geopandas' read_parquet
            or read_file, depending on the file type.

     Returns:
         A GeoDataFrame.
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

    Note:
        Does not currently write to shapelfile or filegeodatabase.

    Args:
        df: The GeoDataFrame to write.
        gcs_path: The path to the file you want to write to.
        **kwargs: Additional keyword arguments passed to parquet.write_table
            (for parquet) or geopandas' to_file method (if not parquet).
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
