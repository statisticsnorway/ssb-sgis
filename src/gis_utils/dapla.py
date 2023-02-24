import os

import geopandas as gpd
import numpy as np
import pandas as pd
from dapla import FileClient, details
from geopandas.io.arrow import _geopandas_to_arrow
from pyarrow import parquet


def exists(path: str) -> bool:
    try:
        details(path)
        return True
    except FileNotFoundError:
        return False
    except ModuleNotFoundError:
        return os.path.exists(path)


def read_geopandas(sti: str, **kwargs) -> gpd.GeoDataFrame:
    fs = FileClient.get_gcs_file_system()

    if "parquet" in sti:
        with fs.open(sti, mode="rb") as file:
            return gpd.read_parquet(file, **kwargs)
    else:
        with fs.open(sti, mode="rb") as file:
            return gpd.read_file(file, **kwargs)


def write_geopandas(df: gpd.GeoDataFrame, gcs_path: str, **kwargs) -> None:
    """funker ikke for shp og gdb"""

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
