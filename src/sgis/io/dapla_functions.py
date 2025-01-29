"""Functions for reading and writing GeoDataFrames in Statistics Norway's GCS Dapla."""

from __future__ import annotations

import functools
import glob
import json
import multiprocessing
import os
import shutil
import uuid
from collections.abc import Callable
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import joblib
import pandas as pd
import pyarrow
import pyarrow.dataset
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import shapely
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from geopandas.io.arrow import _geopandas_to_arrow
from pandas import DataFrame
from pyarrow import ArrowInvalid

from ..conf import config
from ..geopandas_tools.conversion import to_shapely
from ..geopandas_tools.general import get_common_crs
from ..geopandas_tools.sfilter import sfilter
from ..helpers import _get_file_system

try:
    from gcsfs import GCSFileSystem
except ImportError:
    pass

PANDAS_FALLBACK_INFO = " Set pandas_fallback=True to ignore this error."
NULL_VALUE = "__HIVE_DEFAULT_PARTITION__"


def read_geopandas(
    gcs_path: str | Path | list[str | Path] | tuple[str | Path] | GeoSeries,
    pandas_fallback: bool = False,
    file_system: GCSFileSystem | None = None,
    mask: GeoSeries | GeoDataFrame | shapely.Geometry | tuple | None = None,
    threads: int | None = None,
    filters: pyarrow.dataset.Expression | None = None,
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
        threads: Number of threads to use if reading multiple files. Defaults to
            the number of files to read or the number of available threads (if lower).
        filters: To filter out data. Either a pyarrow.dataset.Expression, or a list in the
            structure [[(column, op, val), …],…] where op is [==, =, >, >=, <, <=, !=, in, not in].
            More details here: https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html
        **kwargs: Additional keyword arguments passed to geopandas' read_parquet
            or read_file, depending on the file type.

    Returns:
         A GeoDataFrame if it has rows. If zero rows, a pandas DataFrame is returned.
    """
    file_system = _get_file_system(file_system, kwargs)

    if not isinstance(gcs_path, (str | Path | os.PathLike)):
        cols = {}
        if mask is not None:
            if not isinstance(gcs_path, GeoSeries):
                bounds_series: GeoSeries = get_bounds_series(
                    gcs_path,
                    file_system,
                    threads=threads,
                    pandas_fallback=pandas_fallback,
                )
            else:
                bounds_series = gcs_path
            new_bounds_series = sfilter(bounds_series, mask)
            if not len(new_bounds_series):
                if isinstance(kwargs.get("columns"), Iterable):
                    cols = {col: [] for col in kwargs["columns"]}
                else:
                    cols = {}
                    for path in bounds_series.index:
                        try:
                            cols |= {col: [] for col in _get_columns(path, file_system)}
                        except ArrowInvalid as e:
                            if file_system.isfile(path):
                                raise ArrowInvalid(e, path) from e

                return GeoDataFrame(cols | {"geometry": []})
            paths = list(new_bounds_series.index)
        else:
            if isinstance(gcs_path, GeoSeries):
                paths = list(gcs_path.index)
            else:
                paths = list(gcs_path)

        if threads is None:
            threads = min(len(paths), int(multiprocessing.cpu_count())) or 1

        # recursive read with threads
        with joblib.Parallel(n_jobs=threads, backend="threading") as parallel:
            dfs: list[GeoDataFrame] = parallel(
                joblib.delayed(read_geopandas)(
                    x,
                    filters=filters,
                    file_system=file_system,
                    pandas_fallback=pandas_fallback,
                    mask=mask,
                    threads=threads,
                    **kwargs,
                )
                for x in paths
            )

        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            try:
                df = GeoDataFrame(df)
            except Exception as e:
                if not pandas_fallback:
                    print(e)
                    raise e
        else:
            df = GeoDataFrame(cols | {"geometry": []})

        if mask is not None:
            return sfilter(df, mask)
        return df

    child_paths = has_partitions(gcs_path, file_system)
    if child_paths:
        return gpd.GeoDataFrame(
            _read_partitioned_parquet(
                gcs_path,
                read_func=_read_geopandas,
                file_system=file_system,
                mask=mask,
                pandas_fallback=pandas_fallback,
                filters=filters,
                child_paths=child_paths,
                **kwargs,
            )
        )

    if "parquet" in gcs_path or "prqt" in gcs_path:
        with file_system.open(gcs_path, mode="rb") as file:
            try:
                df = gpd.read_parquet(
                    file, filters=filters, filesystem=file_system, **kwargs
                )
            except ValueError as e:
                if "Missing geo metadata" not in str(e) and "geometry" not in str(e):
                    raise e.__class__(
                        f"{e.__class__.__name__}: {e} for {gcs_path}."
                    ) from e
                df = pd.read_parquet(
                    file, filters=filters, filesystem=file_system, **kwargs
                )
                if pandas_fallback or not len(df):
                    return df
                else:
                    more_txt = PANDAS_FALLBACK_INFO if not len(df) else ""
                    raise e.__class__(
                        f"{e.__class__.__name__}: {e} for {df}." + more_txt
                    ) from e
            except Exception as e:
                raise e.__class__(f"{e.__class__.__name__}: {e} for {gcs_path}.") from e

    else:
        with file_system.open(gcs_path, mode="rb") as file:
            try:
                df = gpd.read_file(
                    file, filters=filters, filesystem=file_system, **kwargs
                )
            except ValueError as e:
                if "Missing geo metadata" not in str(e) and "geometry" not in str(e):
                    raise e
                file_type: str = Path(gcs_path).suffix.strip(".")
                df = getattr(pd, f"read_{file_type}")(
                    file, filters=filters, filesystem=file_system, **kwargs
                )

                if pandas_fallback or not len(df):
                    return df
                else:
                    more_txt = PANDAS_FALLBACK_INFO if not len(df) else ""
                    raise e.__class__(
                        f"{e.__class__.__name__}: {e} for {df}. " + more_txt
                    ) from e
            except Exception as e:
                raise e.__class__(
                    f"{e.__class__.__name__}: {e} for {gcs_path}." + more_txt
                ) from e

    if mask is not None:
        return sfilter(df, mask)
    return df


def _get_bounds_parquet(
    path: str | Path, file_system: GCSFileSystem, pandas_fallback: bool = False
) -> tuple[list[float], dict] | tuple[None, None]:
    with file_system.open(path, "rb") as file:
        return _get_bounds_parquet_from_open_file(file, file_system)


def _get_bounds_parquet_from_open_file(
    file, file_system
) -> tuple[list[float], dict] | tuple[None, None]:
    geo_metadata = _get_geo_metadata(file, file_system)
    if not geo_metadata:
        return None, None
    return geo_metadata["bbox"], geo_metadata["crs"]


def _get_geo_metadata(file, file_system) -> dict:
    meta = pq.read_schema(file).metadata
    geo_metadata = json.loads(meta[b"geo"])
    try:
        primary_column = geo_metadata["primary_column"]
    except KeyError as e:
        raise KeyError(e, geo_metadata) from e
    try:
        return geo_metadata["columns"][primary_column]
    except KeyError as e:
        try:
            num_rows = pq.read_metadata(file).num_rows
        except ArrowInvalid as e:
            if not file_system.isfile(file):
                return {}
            raise ArrowInvalid(e, file) from e
        if not num_rows:
            return {}
    return {}


def _get_columns(path: str | Path, file_system: GCSFileSystem) -> pd.Index:
    with file_system.open(path, "rb") as f:
        schema = pq.read_schema(f)
        index_cols = _get_index_cols(schema)
        return pd.Index(schema.names).difference(index_cols)


def _get_index_cols(schema: pyarrow.Schema) -> list[str]:
    cols = json.loads(schema.metadata[b"pandas"])["index_columns"]
    return [x for x in cols if not isinstance(x, dict)]


def get_bounds_series(
    paths: list[str | Path] | tuple[str | Path],
    file_system: GCSFileSystem | None = None,
    threads: int | None = None,
    pandas_fallback: bool = False,
) -> GeoSeries:
    """Get a GeoSeries with file paths as indexes and the file's bounds as values.

    The returned GeoSeries can be used as the first argument of 'read_geopandas'
    along with the 'mask' keyword.

    Args:
        paths: Iterable of file paths in gcs.
        file_system: Optional instance of GCSFileSystem.
            If None, an instance is created within the function.
            Note that this is slower in long loops.
        threads: Number of threads to use if reading multiple files. Defaults to
            the number of files to read or the number of available threads (if lower).
        pandas_fallback: If False (default), an exception is raised if the file has
            no geo metadata. If True, the geometry value is set to None for this file.

    Returns:
        A geopandas.GeoSeries with file paths as indexes and bounds as values.

    Examples:
    ---------
    >>> import sgis as sg
    >>> import dapla as dp
    >>> all_paths =  GCSFileSystem().ls("...")

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
    file_system = _get_file_system(file_system, {})

    if threads is None:
        threads = min(len(paths), int(multiprocessing.cpu_count())) or 1

    with joblib.Parallel(n_jobs=threads, backend="threading") as parallel:
        bounds: list[tuple[list[float], dict]] = parallel(
            joblib.delayed(_get_bounds_parquet)(
                path, file_system=file_system, pandas_fallback=pandas_fallback
            )
            for path in paths
        )
    crss = {json.dumps(x[1]) for x in bounds}
    crs = get_common_crs(
        [
            crs
            for crs in crss
            if not any(str(crs).lower() == txt for txt in ["none", "null"])
        ]
    )
    return GeoSeries(
        [shapely.box(*bbox[0]) if bbox[0] is not None else None for bbox in bounds],
        index=paths,
        crs=crs,
    )


def write_geopandas(
    df: GeoDataFrame,
    gcs_path: str | Path,
    overwrite: bool = True,
    pandas_fallback: bool = False,
    file_system: GCSFileSystem | None = None,
    partition_cols=None,
    existing_data_behavior: str = "error",
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
        partition_cols: Column(s) to partition by. Only for parquet files.
        existing_data_behavior : 'error' | 'overwrite_or_ignore' | 'delete_matching'.
            Defaults to 'error'. More info: https://arrow.apache.org/docs/python/generated/pyarrow.dataset.write_dataset.html
        **kwargs: Additional keyword arguments passed to parquet.write_table
            (for parquet) or geopandas' to_file method (if not parquet).
    """
    if not isinstance(gcs_path, str):
        try:
            gcs_path = str(gcs_path)
        except TypeError as e:
            raise TypeError(f"Unexpected type {type(gcs_path)}.") from e

    file_system = _get_file_system(file_system, kwargs)

    if not overwrite and file_system.exists(gcs_path):
        raise ValueError("File already exists.")

    if not isinstance(df, GeoDataFrame):
        raise ValueError(f"DataFrame must be GeoDataFrame. Got {type(df)}.")

    if not len(df) and has_partitions(gcs_path, file_system):
        # no need to write empty df
        return
    elif not len(df):
        if pandas_fallback:
            df = pd.DataFrame(df)
            df.geometry = df.geometry.astype(str)
            df.geometry = None
        try:
            with file_system.open(gcs_path, "wb") as file:
                df.to_parquet(file, **kwargs)
        except Exception as e:
            more_txt = PANDAS_FALLBACK_INFO if not pandas_fallback else ""
            raise e.__class__(
                f"{e.__class__.__name__}: {e} for {df}. " + more_txt
            ) from e
        return

    if ".parquet" in gcs_path or "prqt" in gcs_path:
        if partition_cols is not None:
            return _write_partitioned_geoparquet(
                df,
                gcs_path,
                partition_cols,
                file_system,
                existing_data_behavior=existing_data_behavior,
                write_func=_to_geopandas,
                **kwargs,
            )
        with file_system.open(gcs_path, mode="wb") as file:
            df.to_parquet(file, **kwargs)
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

    with BytesIO() as buffer:
        df.to_file(buffer, driver=driver)
        buffer.seek(0)  # Rewind the buffer to the beginning

        # Upload buffer content to the desired storage
        with file_system.open(gcs_path, "wb") as file:
            file.write(buffer.read())


def _to_geopandas(df, path, **kwargs) -> None:
    table = _geopandas_to_arrow(
        df,
        index=df.index,
        schema_version=None,
    )

    if "schema" in kwargs:
        schema = kwargs.pop("schema")

        # make sure to get the actual metadata
        schema = pyarrow.schema(
            [(schema.field(col).name, schema.field(col).type) for col in schema.names],
            metadata=table.schema.metadata,
        )
        table = table.select(schema.names).cast(schema)

    pq.write_table(table, path, compression="snappy", **kwargs)


def _remove_file(path, file_system) -> None:
    try:
        file_system.rm_file(str(path))
    except (AttributeError, TypeError, PermissionError) as e:
        print(path, type(e), e)
        try:
            shutil.rmtree(path)
        except NotADirectoryError:
            try:
                os.remove(path)
            except PermissionError:
                pass


def _write_partitioned_geoparquet(
    df,
    path,
    partition_cols,
    file_system=None,
    write_func: Callable = _to_geopandas,
    existing_data_behavior: str = "error",
    **kwargs,
):
    if isinstance(partition_cols, str):
        partition_cols = [partition_cols]

    file_system = _get_file_system(file_system, kwargs)

    path = Path(path)
    unique_id = uuid.uuid4()

    for col in partition_cols:
        if df[col].isna().all() and not kwargs.get("schema"):
            raise ValueError("Must specify 'schema' when all rows are NA.")

    try:
        glob_func = functools.partial(file_system.glob, detail=False)
    except AttributeError:
        glob_func = functools.partial(glob.glob, recursive=True)

    args: list[tuple[Path, DataFrame]] = []
    dirs: list[Path] = set()
    for group, rows in df.groupby(partition_cols, dropna=False):
        name = (
            "/".join(
                f"{col}={value if not pd.isna(value) else NULL_VALUE}"
                for col, value in zip(partition_cols, group, strict=True)
            )
            + f"/{unique_id}.parquet"
        )

        dirs.add((path / name).parent)
        args.append((path / name, rows))

    if file_system.exists(path) and file_system.isfile(path):
        _remove_file(path, file_system)

    if kwargs.get("schema"):
        schema = kwargs.pop("schema")
    elif isinstance(df, GeoDataFrame):
        geom_name = df.geometry.name
        pandas_columns = [col for col in df if col != geom_name]
        schema = pyarrow.Schema.from_pandas(df[pandas_columns], preserve_index=True)
        index_columns = _get_index_cols(schema)
        schema = pyarrow.schema(
            [
                (
                    (schema.field(col).name, schema.field(col).type)
                    if col != geom_name
                    else (geom_name, pyarrow.binary())
                )
                for col in [*df.columns, *index_columns]
                # for col in df.columns
            ]
        )
    else:
        schema = pyarrow.Schema.from_pandas(df, preserve_index=True)

    def get_siblings(path: str, paths: list[str]) -> list[str]:
        parts = path.parts
        return {x for x in paths if all(part in parts for part in x.parts)}

    def threaded_write(path_rows):
        new_path, rows = path_rows
        # for sibling_path in get_siblings(new_path, child_paths):
        for sibling_path in glob_func(str(Path(new_path).with_name("**"))):
            if not paths_are_equal(sibling_path, Path(new_path).parent):
                if existing_data_behavior == "delete_matching":
                    _remove_file(sibling_path, file_system)
                elif existing_data_behavior == "error":
                    raise pyarrow.ArrowInvalid(
                        f"Could not write to  {path} as the directory is not empty and existing_data_behavior is to error"
                    )
        try:
            with file_system.open(new_path, mode="wb") as file:
                write_func(rows, file, schema=schema, **kwargs)
        except FileNotFoundError:
            file_system.makedirs(str(Path(new_path).parent), exist_ok=True)
            with file_system.open(new_path, mode="wb") as file:
                write_func(rows, file, schema=schema, **kwargs)

    with ThreadPoolExecutor() as executor:
        list(executor.map(threaded_write, args))


def _filters_to_expression(filters) -> list[ds.Expression]:
    if filters is None:
        return None
    elif isinstance(filters, pyarrow.dataset.Expression):
        return filters

    for filt in filters:
        if "in" in filt and isinstance(filt[-1], str):
            raise ValueError(
                "Using strings with 'in' is ambigous. Use a list of strings."
            )
    try:
        return pq.core.filters_to_expression(filters)
    except ValueError as e:
        raise ValueError(f"{e}: {filters}") from e


def expression_match_path(expression: ds.Expression, path: str) -> bool:
    """Check if a file path match a pyarrow Expression.

    Examples:
    --------
    >>> import pyarrow.compute as pc
    >>> path = 'data/file.parquet/x=1/y=10/name0.parquet'
    >>> expression = (pc.Field("x") == 1) & (pc.Field("y") == 10)
    >>> expression_match_path(path, expression)
    True
    >>> expression = (pc.Field("x") == 1) & (pc.Field("y") == 5)
    >>> expression_match_path(path, expression)
    False
    >>> expression = (pc.Field("x") == 1) & (pc.Field("z") == 10)
    >>> expression_match_path(path, expression)
    False
    """
    if NULL_VALUE in path:
        return True
    # build a one lengthed pyarrow.Table of the partitioning in the file path
    values = []
    names = []
    for part in Path(path).parts:
        if part.count("=") != 1:
            continue
        name, value = part.split("=")
        values.append([value])
        names.append(name)
    table = pyarrow.Table.from_arrays(values, names=names)
    try:
        table = table.filter(expression)
    except pyarrow.ArrowInvalid as e:
        if "No match for FieldRef" not in str(e):
            raise e
        # cannot determine if the expression match without reading the file
        return True
    return bool(len(table))


def _read_geopandas(file, pandas_fallback: bool, **kwargs):
    try:
        return gpd.read_parquet(file, **kwargs)
    except Exception as e:
        if not pandas_fallback:
            raise e
        df = pd.read_parquet(file, **kwargs)
        if len(df):
            raise e
        return df


def _read_pandas(gcs_path: str, **kwargs):
    file_system = _get_file_system(None, kwargs)

    child_paths = has_partitions(gcs_path, file_system)
    if child_paths:
        return gpd.GeoDataFrame(
            _read_partitioned_parquet(
                gcs_path,
                read_func=pd.read_parquet,
                file_system=file_system,
                mask=None,
                child_paths=child_paths,
                **kwargs,
            )
        )

    with file_system.open(gcs_path, "rb") as file:
        return pd.read_parquet(file, **kwargs)


def _read_partitioned_parquet(
    path: str,
    read_func: Callable,
    filters=None,
    file_system=None,
    mask=None,
    child_paths: list[str] | None = None,
    **kwargs,
):
    file_system = _get_file_system(file_system, kwargs)

    if child_paths is None:
        try:
            glob_func = functools.partial(file_system.glob)
        except AttributeError:
            glob_func = functools.partial(glob.glob, recursive=True)
        child_paths = list(glob_func(str(Path(path) / "**/*.parquet")))

    filters = _filters_to_expression(filters)

    def intersects(file, mask) -> bool:
        bbox, _ = _get_bounds_parquet_from_open_file(file, file_system)
        return shapely.box(*bbox).intersects(to_shapely(mask))

    def read(path) -> GeoDataFrame | None:
        with file_system.open(path, "rb") as file:
            if mask is not None and not intersects(file, mask):
                return

            schema = kwargs.get("schema", pq.read_schema(file))
            # copy kwargs because mutable
            new_kwargs = {
                key: value for key, value in kwargs.items() if key != "schema"
            }

            return read_func(file, schema=schema, filters=filters, **new_kwargs)

    with ThreadPoolExecutor() as executor:
        results = [
            x
            for x in (
                executor.map(
                    read,
                    (
                        path
                        for path in child_paths
                        if filters is None or expression_match_path(filters, path)
                    ),
                )
            )
            if x is not None
        ]
    if results:
        if mask is not None:
            return sfilter(pd.concat(results), mask)
        return pd.concat(results)

    # add columns to empty DataFrame
    first_path = next(iter(child_paths + [path]))
    return pd.DataFrame(
        columns=list(dict.fromkeys(_get_columns(first_path, file_system)))
    )


def paths_are_equal(path1: Path | str, path2: Path | str) -> bool:
    return Path(path1).parts == Path(path2).parts


def has_partitions(path, file_system) -> list[str]:
    try:
        glob_func = functools.partial(file_system.glob, detail=False)
    except AttributeError:
        glob_func = functools.partial(glob.glob, recursive=True)

    return [
        x
        for x in glob_func(str(Path(path) / "**/*.parquet"))
        if not paths_are_equal(x, path)
    ]


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
    file_system = config["file_system"]()

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
    file_system = config["file_system"]()

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
