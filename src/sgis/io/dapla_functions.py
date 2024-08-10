"""Functions for reading and writing GeoDataFrames in Statistics Norway's GCS Dapla."""

import json
from pathlib import Path

import dapla as dp
import geopandas as gpd
import joblib
import pandas as pd
import shapely
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from geopandas.io.arrow import _geopandas_to_arrow
from pandas import DataFrame
from pyarrow import parquet

from ..geopandas_tools.sfilter import sfilter


def read_geopandas(
    gcs_path: str | Path | list[str | Path],
    pandas_fallback: bool = False,
    file_system: dp.gcs.GCSFileSystem | None = None,
    mask: GeoSeries | GeoDataFrame | shapely.Geometry | tuple | None = None,
    validate_crs: bool = True,
    **kwargs,
) -> GeoDataFrame | DataFrame:
    """Reads geoparquet or other geodata from one or more files on GCS.

    If the file has 0 rows, the contents will be returned as a pandas.DataFrame,
    since geopandas does not read and write empty tables.

    Note:
        Does not currently read shapefiles or filegeodatabases.

    Args:
        gcs_path: path to one or more files on Google Cloud Storage.
            Multiple paths are read with threading.
        pandas_fallback: If False (default), an exception is raised if the file can
            not be read with geopandas and the number of rows is more than 0. If True,
            the file will be read with pandas if geopandas fails.
        file_system: Optional file system.
        mask: Optional geometry mask to keep only intersecting geometries.
            If 'gcs_path' is an iterable of multiple paths, only the files
            with a bbox that intersects the mask are read, then filtered by location.
        validate_crs: If multiple files are to be read and validate_crs is True (default),
            all files in the folder has to have the same coordinate reference system.
        **kwargs: Additional keyword arguments passed to geopandas' read_parquet
            or read_file, depending on the file type.

    Returns:
         A GeoDataFrame if it has rows. If zero rows, a pandas DataFrame is returned.
    """
    if file_system is None:
        file_system = dp.FileClient.get_gcs_file_system()

    if isinstance(gcs_path, (list, tuple)):
        kwargs |= {"file_system": file_system, "pandas_fallback": pandas_fallback}

        if mask is not None:
            in_bounds: GeoSeries = sfilter(
                _get_bounds_series(gcs_path, file_system, validate_crs), mask
            )
            gcs_path: list[str] = list(in_bounds.index)

        # recursive read with threads
        with joblib.Parallel(n_jobs=len(gcs_path), backend="threading") as parallel:
            dfs: list[GeoDataFrame] = parallel(
                joblib.delayed(read_geopandas)(x, **kwargs) for x in gcs_path
            )
        df = pd.concat(dfs)
        if mask is not None:
            return sfilter(df, mask)
        return df

    if not isinstance(gcs_path, str):
        try:
            gcs_path = str(gcs_path)
        except TypeError as e:
            raise TypeError(f"Unexpected type {type(gcs_path)}.") from e

    if "parquet" in gcs_path or "prqt" in gcs_path:
        with file_system.open(gcs_path, mode="rb") as file:
            try:
                df = gpd.read_parquet(file, **kwargs)
            except ValueError as e:
                if "Missing geo metadata" not in str(e) and "geometry" not in str(e):
                    raise e
                df = dp.read_pandas(gcs_path, **kwargs)

                if pandas_fallback or not len(df):
                    return df
                else:
                    raise e
    else:
        with file_system.open(gcs_path, mode="rb") as file:
            try:
                df = gpd.read_file(file, **kwargs)
            except ValueError as e:
                if "Missing geo metadata" not in str(e) and "geometry" not in str(e):
                    raise e
                df = dp.read_pandas(gcs_path, **kwargs)

                if pandas_fallback or not len(df):
                    return df
                else:
                    raise e

    if mask is not None:
        return sfilter(df, mask)
    return df


def _get_bounds_parquet(
    path: str | Path, fs: dp.gcs.GCSFileSystem
) -> tuple[list[float], dict]:
    with fs.open(path) as f:
        meta = json.loads(parquet.read_schema(f).metadata[b"geo"])["columns"][
            "geometry"
        ]
    return meta["bbox"], meta["crs"]


def _get_bounds_series(gcs_path, fs: dp.gcs.GCSFileSystem, validate_crs) -> GeoSeries:
    with joblib.Parallel(n_jobs=len(gcs_path), backend="threading") as parallel:
        bounds: list[tuple[list[float], dict]] = parallel(
            joblib.delayed(_get_bounds_parquet)(path, fs=fs) for path in gcs_path
        )
    if validate_crs and len({json.dumps(x[1]) for x in bounds}) != 1:
        raise ValueError(f"crs mismatch {({json.dumps(x[1]) for x in bounds})}")
    return GeoSeries(
        [shapely.box(*bbox[0]) for bbox in bounds], index=gcs_path, crs=bounds[0][1]
    )


def write_geopandas(
    df: GeoDataFrame,
    gcs_path: str | Path,
    overwrite: bool = True,
    pandas_fallback: bool = False,
    file_system: dp.gcs.GCSFileSystem | None = None,
    write_covering_bbox: bool = False,
    **kwargs,
) -> None:
    """Writes a GeoDataFrame to the speficied format.

    Note:
        Does not currently write to shapelfile or filegeodatabase.

    Args:
        df: The GeoDataFrame to write.
        gcs_path: The path to the file you want to write to.
        overwrite: Whether to overwrite the file if it exists. Defaults to True.
        pandas_fallback: If False (default), an exception is raised if the file can
            not be written with geopandas and the number of rows is more than 0. If True,
            the file will be written without geo-metadata if >0 rows.
        file_system: Optional file sustem.
        write_covering_bbox: Writes the bounding box column for each row entry with column name "bbox".
            Writing a bbox column can be computationally expensive, but allows you to specify
            a bbox in : func:read_parquet for filtered reading.
            Note: this bbox column is part of the newer GeoParquet 1.1 specification and should be
            considered as experimental. While writing the column is backwards compatible, using it
            for filtering may not be supported by all readers.

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

    if not isinstance(df, GeoDataFrame):
        raise ValueError("DataFrame must be GeoDataFrame.")

    if file_system is None:
        file_system = dp.FileClient.get_gcs_file_system()

    if not len(df):
        if pandas_fallback:
            df = pd.DataFrame(df)
            df.geometry = df.geometry.astype(str)
        dp.write_pandas(df, gcs_path, **kwargs)
        return

    file_system = dp.FileClient.get_gcs_file_system()

    if ".parquet" in gcs_path or "prqt" in gcs_path:
        with file_system.open(gcs_path, mode="wb") as buffer:
            table = _geopandas_to_arrow(
                df,
                index=df.index,
                schema_version=None,
                write_covering_bbox=write_covering_bbox,
            )
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

    with file_system.open(gcs_path, "wb") as file:
        df.to_file(file, driver=driver, layer=layer)


def exists(path: str | Path) -> bool:
    """Returns True if the path exists, and False if it doesn't.

    Args:
        path (str): The path to the file or directory.

    Returns:
        True if the path exists, False if not.
    """
    file_system = dp.FileClient.get_gcs_file_system()
    return file_system.exists(path)


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
    file_system = dp.FileClient.get_gcs_file_system()

    # (recursive doesn't work, so doing recursive search below)
    info = file_system.ls(folder, detail=True, recursive=True)

    if not info:
        return pd.DataFrame(columns=["kb", "mb", "name", "child", "path"])

    fileinfo = [
        (x["name"], x["size"], x["updated"])
        for x in info
        if x["storageClass"] != "DIRECTORY"
    ]
    folderinfo = [x["name"] for x in info if x["storageClass"] == "DIRECTORY"]

    fileinfo += _get_files_in_subfolders(folderinfo)

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


def _get_files_in_subfolders(folderinfo: list[dict]) -> list[tuple]:
    file_system = dp.FileClient.get_gcs_file_system()

    fileinfo = []

    while folderinfo:
        new_folderinfo = []
        for m in folderinfo:
            more_info = file_system.ls(m, detail=True, recursive=True)
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
