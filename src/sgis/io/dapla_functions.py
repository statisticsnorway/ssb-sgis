"""Functions for reading and writing GeoDataFrames in Statistics Norway's GCS Dapla.
"""
from pathlib import Path
from typing import Optional

import dapla as dp
import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame
from geopandas.io.arrow import _geopandas_to_arrow
from pandas import DataFrame
from pyarrow import parquet


def read_geopandas(
    gcs_path: str | Path,
    pandas_fallback: bool = False,
    fs: Optional[dp.gcs.GCSFileSystem] = None,
    **kwargs,
) -> GeoDataFrame | DataFrame:
    """Reads geoparquet or other geodata from a file on GCS.

    If the file has 0 rows, the contents will be returned as a pandas.DataFrame,
    since geopandas does not read and write empty tables.

    Note:
        Does not currently read shapefiles or filegeodatabases.

    Args:
        gcs_path: path to a file on Google Cloud Storage.
        pandas_fallback: If False (default), an exception is raised if the file can
            not be read with geopandas and the number of rows is more than 0. If True,
            the file will be read as
        **kwargs: Additional keyword arguments passed to geopandas' read_parquet
            or read_file, depending on the file type.

     Returns:
         A GeoDataFrame if it has rows. If zero rows, a pandas DataFrame is returned.
    """

    if not isinstance(gcs_path, str):
        try:
            gcs_path = str(gcs_path)
        except TypeError:
            raise TypeError(f"Unexpected type {type(gcs_path)}.")

    if fs is None:
        fs = dp.FileClient.get_gcs_file_system()

    if "parquet" in gcs_path or "prqt" in gcs_path:
        with fs.open(gcs_path, mode="rb") as file:
            try:
                return gpd.read_parquet(file, **kwargs)
            except ValueError as e:
                if "Missing geo metadata" not in str(e) and "geometry" not in str(e):
                    raise e
                df = dp.read_pandas(gcs_path, **kwargs)

                if pandas_fallback or not len(df):
                    return df
                else:
                    raise e
    else:
        with fs.open(gcs_path, mode="rb") as file:
            try:
                return gpd.read_file(file, **kwargs)
            except ValueError as e:
                if "Missing geo metadata" not in str(e) and "geometry" not in str(e):
                    raise e
                df = dp.read_pandas(gcs_path, **kwargs)

                if pandas_fallback or not len(df):
                    return df
                else:
                    raise e


def write_geopandas(
    df: gpd.GeoDataFrame,
    gcs_path: str | Path,
    overwrite: bool = True,
    pandas_fallback: bool = False,
    fs: Optional[dp.gcs.GCSFileSystem] = None,
    **kwargs,
) -> None:
    """Writes a GeoDataFrame to the speficied format.

    Note:
        Does not currently write to shapelfile or filegeodatabase.

    Args:
        df: The GeoDataFrame to write.
        gcs_path: The path to the file you want to write to.
        overwrite: Whether to overwrite the file if it exists. Defaults to True.
        **kwargs: Additional keyword arguments passed to parquet.write_table
            (for parquet) or geopandas' to_file method (if not parquet).
    """

    if not isinstance(gcs_path, str):
        try:
            gcs_path = str(gcs_path)
        except TypeError as e:
            raise TypeError(f"Unexpected type {type(gcs_path)}.") from e

    if not overwrite and exists(gcs_path):
        raise ValueError("File already exists.")

    if fs is None:
        fs = dp.FileClient.get_gcs_file_system()

    pd.io.parquet.BaseImpl.validate_dataframe(df)

    if not len(df):
        if pandas_fallback:
            df.geometry = df.geometry.astype(str)
            df = pd.DataFrame(df)
        dp.write_pandas(df, gcs_path, **kwargs)
        return

    fs = dp.FileClient.get_gcs_file_system()

    if ".parquet" in gcs_path or "prqt" in gcs_path:
        with fs.open(gcs_path, mode="wb") as buffer:
            table = _geopandas_to_arrow(df, index=df.index, schema_version=None)
            parquet.write_table(table, buffer, compression="snappy", **kwargs)
        return

    layer = kwargs.pop("layer", None)
    if ".gpkg" in gcs_path:
        driver = "GPKG"
        layer = Path(gcs_path).stem
    elif ".geojson" in gcs_path:
        driver = "GeoJSON"
    elif ".gml" in gcs_path:
        driver = "GML"
    elif ".shp" in gcs_path:
        driver = "ESRI Shapefile"
    else:
        driver = None

    with fs.open(gcs_path, "wb") as file:
        df.to_file(file, driver=driver, layer=layer)


def exists(path: str | Path) -> bool:
    """Returns True if the path exists, and False if it doesn't.

    Args:
        path (str): The path to the file or directory.

    Returns:
        True if the path exists, False if not.
    """

    fs = dp.FileClient.get_gcs_file_system()
    return fs.exists(path)


def check_files(
    folder: str,
    contains: str | None = None,
    within_minutes: int | None = None,
) -> pd.DataFrame:
    """Returns DataFrame of files in the folder and subfolders with times and sizes.

    Args:
        folder: Google cloud storage folder.
        contains: Optional substring that must be in the file path.
        within_minutes: Optionally include only files that were updated in the
            last n minutes.
    """
    fs = dp.FileClient.get_gcs_file_system()

    # (recursive doesn't work, so doing recursive search below)
    info = fs.ls(folder, detail=True, recursive=True)

    if not info:
        return pd.DataFrame(columns=["kb", "mb", "name", "child", "path"])

    fileinfo = [
        (x["name"], x["size"], x["updated"])
        for x in info
        if x["storageClass"] != "DIRECTORY"
    ]
    folderinfo = [x["name"] for x in info if x["storageClass"] == "DIRECTORY"]

    fileinfo += get_files_in_subfolders(folderinfo)

    df = pd.DataFrame(fileinfo, columns=["path", "kb", "updated"])

    if contains:
        try:
            df = df.loc[df["path"].str.contains(contains)]
        except TypeError:
            for item in contains:
                df = df.loc[df["path"].str.contains(item)]

    if not len(df):
        return df

    df = df.set_index("updated").sort_index()

    df["name"] = df["path"].apply(lambda x: Path(x).name)
    df["child"] = df["path"].apply(lambda x: Path(x).parent.name)

    df["kb"] = df["kb"].round(1)
    df["mb"] = (df["kb"] / 1_000_000).round(1)

    df.index = (
        pd.to_datetime(df.index)
        .round("s")
        .tz_convert("Europe/Oslo")
        .tz_localize(None)
        .round("s")
    )

    df.index.name = None

    if not within_minutes:
        return df[["kb", "mb", "name", "child", "path"]]

    the_time = pd.Timestamp.now() - pd.Timedelta(minutes=within_minutes)
    return df.loc[lambda x: x.index > the_time, ["kb", "mb", "name", "child", "path"]]


def get_files_in_subfolders(folderinfo: list[dict]) -> list[dict]:
    fs = dp.FileClient.get_gcs_file_system()

    if isinstance(folderinfo, (str, Path)):
        folderinfo = [folderinfo]

    fileinfo = []

    while folderinfo:
        new_folderinfo = []
        for m in folderinfo:
            more_info = fs.ls(m, detail=True, recursive=True)
            if not more_info:
                continue

            more_fileinfo = [
                (x["name"], x["size"], x["updated"])
                for x in more_info
                if x["storageClass"] != "DIRECTORY"
            ]
            fileinfo.extend(more_fileinfo)

            more_folderinfo = [
                x["name"]
                for x in more_info
                if x["storageClass"] == "DIRECTORY" and x["name"] not in folderinfo
            ]
            new_folderinfo.extend(more_folderinfo)

        folderinfo = new_folderinfo

    return fileinfo
