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
    gcs_path: str | Path | list[str | Path] | tuple[str | Path] | GeoSeries,
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

    if isinstance(gcs_path, (list | tuple | GeoSeries)):
        kwargs |= {"file_system": file_system, "pandas_fallback": pandas_fallback}

        if mask is not None:
            if not isinstance(gcs_path, GeoSeries):
                bounds_series: GeoSeries = get_bounds_series(
                    gcs_path, file_system, validate_crs
                )
            else:
                bounds_series = gcs_path
            bounds_series = sfilter(bounds_series, mask)
            if not len(bounds_series):
                return GeoDataFrame({"geometry": []})
            paths: list[str] = list(bounds_series.index)
        else:
            if isinstance(gcs_path, GeoSeries):
                paths: list[str] = list(gcs_path.index)
            else:
                paths = gcs_path

        # recursive read with threads
        with joblib.Parallel(n_jobs=len(paths), backend="threading") as parallel:
            dfs: list[GeoDataFrame] = parallel(
                joblib.delayed(read_geopandas)(x, **kwargs) for x in paths
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
    path: str | Path, file_system: dp.gcs.GCSFileSystem
) -> tuple[list[float], dict] | tuple[None, None]:
    with file_system.open(path) as f:
        num_rows = parquet.read_metadata(f).num_rows
        if not num_rows:
            return None, None
        meta = parquet.read_schema(f).metadata
    meta = json.loads(meta[b"geo"])["columns"]["geometry"]
    return meta["bbox"], meta["crs"]


def get_bounds_series(
    paths: list[str | Path] | tuple[str | Path],
    file_system: dp.gcs.GCSFileSystem | None = None,
    validate_crs: bool = False,
) -> GeoSeries:
    """Get a GeoSeries with file paths as indexes and the file's bounds as values.

    The returned GeoSeries can be used as the first argument of 'read_geopandas'
    along with the 'mask' keyword.

    Args:
        paths: Iterable of file paths in gcs.
        file_system: Optional instance of dp.gcs.GCSFileSystem.
            If None, an instance is created within the function.
            Note that this is slower in long loops.
        validate_crs: If True, a ValueError is raised if all
            files does not have the same coordinate reference system.
            Defaults to False.

    Returns:
        A geopandas.GeoSeries with file paths as indexes and bounds as values.

    Examples:
    -------
    >>> import sgis as sg
    >>> import dapla as dp
    >>> file_system = dp.FileClient.get_gcs_file_system()
    >>> all_paths = file_system.ls("...")

    Get the bounds of all your file paths, indexed by path.

    >>> bounds_series = sg.get_bounds_series(all_paths, file_system)
    >>> bounds_series
    .../0301.parquet    POLYGON ((273514.334 6638380.233, 273514.334 6...
    .../1101.parquet    POLYGON ((6464.463 6503547.192, 6464.463 65299...
    .../1103.parquet    POLYGON ((-6282.301 6564097.347, -6282.301 660...
    .../1106.parquet    POLYGON ((-46359.891 6622984.385, -46359.891 6...
    .../1108.parquet    POLYGON ((30490.798 6551661.467, 30490.798 658...
                                                                                                            ...
    .../5628.parquet    POLYGON ((1019391.867 7809550.777, 1019391.867...
    .../5630.parquet    POLYGON ((1017907.145 7893398.317, 1017907.145...
    .../5632.parquet    POLYGON ((1075687.587 7887714.263, 1075687.587...
    .../5634.parquet    POLYGON ((1103447.451 7874551.663, 1103447.451...
    .../5636.parquet    POLYGON ((1024129.618 7838961.91, 1024129.618 ...
    Length: 357, dtype: geometry

    Make a grid around the total bounds of the files,
    and read geometries intersecting with the mask in a loop.

    >>> grid = sg.make_grid(bounds_series, 10_000)
    >>> for mask in grid.geometry:
    ...     df = sg.read_geopandas(
    ...         bounds_series,
    ...         mask=mask,
    ...         file_system=file_system,
    ...     )

    """
    if file_system is None:
        file_system = dp.FileClient.get_gcs_file_system()

    with joblib.Parallel(n_jobs=len(paths), backend="threading") as parallel:
        bounds: list[tuple[list[float], dict]] = parallel(
            joblib.delayed(_get_bounds_parquet)(path, file_system=file_system)
            for path in paths
        )
    if validate_crs and len({json.dumps(x[1]) for x in bounds}) != 1:
        raise ValueError(f"crs mismatch {({json.dumps(x[1]) for x in bounds})}")
    return GeoSeries(
        [shapely.box(*bbox[0]) if bbox[0] is not None else None for bbox in bounds],
        index=paths,
        crs=bounds[0][1],
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
