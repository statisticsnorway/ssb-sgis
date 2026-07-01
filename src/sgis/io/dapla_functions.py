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
from pathlib import Path
from typing import Any

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.dataset
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import shapely
from geopandas import GeoDataFrame
from geopandas import GeoSeries
from geopandas.io.arrow import _arrow_to_geopandas
from geopandas.io.arrow import _geopandas_to_arrow
from pandas import DataFrame
from pyarrow import ArrowInvalid

from ..conf import config
from ..geopandas_tools.conversion import to_shapely
from ..geopandas_tools.general import get_common_crs
from ..geopandas_tools.sfilter import sfilter
from ..helpers import _get_file_system
from ..helpers import _standardize_path

try:
    from gcsfs import GCSFileSystem
except ImportError:

    class GCSFileSystem:
        """Placeholder."""


PANDAS_FALLBACK_INFO = " Set pandas_fallback=True to ignore this error."
NULL_VALUE = "__HIVE_DEFAULT_PARTITION__"


def read_geopandas(
    gcs_path: str | Path | list[str | Path] | tuple[str | Path] | GeoSeries,
    pandas_fallback: bool = False,
    file_system: GCSFileSystem | None = None,
    mask: GeoSeries | GeoDataFrame | shapely.Geometry | tuple | None = None,
    use_threads: bool = True,
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
        mask: If gcs_path is a partitioned parquet file or an interable of paths.
            Only files with a bbox intersecting mask will be read.
            Note that the data is not filtered on a row level. You should either
            use clip or sfilter to filter the data after reading.
        use_threads: Defaults to True.
        filters: To filter out data. Either a pyarrow.dataset.Expression, or a list in the
            structure [[(column, op, val), …],…] where op is [==, =, >, >=, <, <=, !=, in, not in].
            More details here: https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html
        **kwargs: Additional keyword arguments passed to geopandas' read_parquet
            or read_file, depending on the file type.

    Returns:
         A GeoDataFrame if it has rows. If zero rows, a pandas DataFrame is returned.
    """
    file_system = _get_file_system(file_system, kwargs)

    if isinstance(gcs_path, (Path | os.PathLike)):
        gcs_path = str(gcs_path)
    elif not isinstance(gcs_path, str):
        return _read_geopandas_from_iterable(
            gcs_path,
            mask=mask,
            file_system=file_system,
            use_threads=use_threads,
            pandas_fallback=pandas_fallback,
            filters=filters,
            **kwargs,
        )

    if isinstance(file_system, GCSFileSystem) and not str(gcs_path).startswith("gs://"):
        gcs_path = "gs://" + str(gcs_path)

    single_eq_filter = (
        isinstance(filters, Iterable)
        and len(filters) == 1
        and ("=" in next(iter(filters)) or "==" in next(iter(filters)))
    )
    # try to read files in subfolder path / "column=value"
    # because glob is slow without GCSFileSystem from the root partition
    if single_eq_filter:
        try:
            expression: list[str] = "".join(
                [str(x) for x in next(iter(filters))]
            ).replace("==", "=")
            paths = get_child_paths(gcs_path, file_system, pattern=f"/{expression}/*")
            if paths:
                return _read_geopandas_from_iterable(
                    paths,
                    mask=mask,
                    file_system=file_system,
                    use_threads=use_threads,
                    pandas_fallback=pandas_fallback,
                    filters=filters,
                    **kwargs,
                )
        except FileNotFoundError:
            pass

    child_paths = get_child_paths(gcs_path, file_system)
    if child_paths:
        return gpd.GeoDataFrame(
            _read_partitioned_parquet(
                gcs_path,
                file_system=file_system,
                mask=mask,
                filters=filters,
                child_paths=child_paths,
                use_threads=use_threads,
                **kwargs,
            )
        )

    if not gcs_path.endswith(".parquet"):
        file_format: str = Path(gcs_path).suffix.lstrip(".")
        return _read_geopandas_single_path(
            gcs_path,
            read_func=gpd.read_file,
            file_format=file_format,
            filters=filters,
            **kwargs,
        )

    table = _read_geopandas_single_path(
        gcs_path,
        read_func=functools.partial(_read_pyarrow, file_system=file_system),
        file_format="parquet",
        filters=filters,
        **kwargs,
    )
    if pandas_fallback and not len(table):
        return GeoDataFrame(table.to_pandas())
    geo_metadata = _get_geo_metadata(gcs_path, file_system)
    return _arrow_to_geopandas(table, geo_metadata)


def get_schema(file) -> pyarrow.Schema:
    try:
        return pq.read_schema(file)
    except (PermissionError, pyarrow.ArrowInvalid, OSError):
        return ds.dataset(file).schema


def _read_geopandas_from_iterable(
    paths,
    mask,
    file_system,
    use_threads,
    pandas_fallback,
    **kwargs,
):
    if isinstance(file_system, GCSFileSystem):
        paths = ["gs://" + str(x).replace("gs://", "") for x in paths]

    cols = {}
    if mask is None and isinstance(paths, GeoSeries):
        # bounds GeoSeries indexed with file paths
        paths = list(paths.index)
    elif mask is None:
        paths = list(paths)
    elif isinstance(paths, GeoSeries):
        bounds_series = sfilter(paths, mask)
        if not len(bounds_series):
            # return GeoDataFrame with correct columns
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
            first_path = next(iter(paths.index))
            _, crs = _get_bounds_parquet(first_path, file_system)
            return GeoDataFrame(cols | {"geometry": []}, crs=crs)
        paths = list(bounds_series.index)

    filters = kwargs.pop("filters")
    filters = _filters_to_expression(filters)
    paths = [
        path
        for path in paths
        if filters is None or expression_match_path(filters, path)
    ]
    try:
        results: list[pyarrow.Table] = _read_pyarrow_with_threads(
            paths,
            file_system=file_system,
            mask=mask,
            use_threads=use_threads,
            **kwargs,
        )
    except _FileIsPartitionedError:
        return pd.concat(
            [
                _read_partitioned_parquet(
                    path,
                    filters=filters,
                    file_system=file_system,
                    mask=mask,
                    use_threads=use_threads,
                    **kwargs,
                )
                for path in paths
            ]
        )

    if results:
        try:
            return _concat_pyarrow_to_geopandas(results, paths, file_system)
        except Exception as e:
            if not pandas_fallback:
                print(e)
                raise e
    else:
        first_path = next(iter(paths))
        _, crs = _get_bounds_parquet(first_path, file_system)
        df = GeoDataFrame(cols | {"geometry": []}, crs=crs)

    return df


def _read_pyarrow_with_threads(
    paths: list[str | Path | os.PathLike],
    use_threads,
    **kwargs,
) -> list[pyarrow.Table]:
    read_partial = functools.partial(_read_pyarrow, **kwargs)
    if not use_threads:
        return [x for x in map(read_partial, paths) if x is not None]
    with ThreadPoolExecutor() as executor:
        return [x for x in executor.map(read_partial, paths) if x is not None]


def _concat_pyarrow_tables(
    tables: list[pyarrow.Table], promote_options: str = "permissive"
) -> pyarrow.Table:
    try:
        return pyarrow.concat_tables(tables, promote_options=promote_options)
    except pyarrow.lib.ArrowTypeError:
        schema = pyarrow.unify_schemas(
            [table.schema for table in tables], promote_options=promote_options
        )
        coerced_tables = [
            table.cast(schema, safe=False) if not table.schema.equals(schema) else table
            for table in tables
        ]
        return pyarrow.concat_tables(coerced_tables, promote_options=promote_options)


def intersects(file, mask, file_system) -> bool:
    bbox, _ = _get_bounds_parquet_from_open_file(file, file_system)
    return shapely.box(*bbox).intersects(to_shapely(mask))


def _read_pyarrow(
    path: str,
    file_system,
    mask=None,
    partition_dtypes: dict[str, pyarrow.DataType] | None = None,
    **kwargs,
) -> pyarrow.Table | None:
    if partition_dtypes is None:
        partition_dtypes = {}
    partition_cols_and_values = {
        part.split("=")[0]: part.split("=")[-1]
        for part in Path(path).parts
        if "=" in part
    }
    try:
        if mask is not None and not intersects(path, mask, file_system):
            return

        columns = None
        try:
            table = pq.read_table(path, **kwargs)
        except pyarrow.lib.ArrowTypeError:
            if "schema" not in kwargs:
                schema = get_schema(path)
                if "columns" in kwargs and hasattr(kwargs["columns"], "__iter__"):
                    columns = list(kwargs["columns"])
                    schema = pyarrow.schema(
                        [
                            (schema.field(col).name, schema.field(col).type)
                            for col in schema.names
                            if col in columns
                        ],
                        metadata=schema.metadata,
                    )
            else:
                schema = kwargs["schema"]

            new_kwargs = {
                key: value for key, value in kwargs.items() if key != "schema"
            }
            table = pq.read_table(path, schema=schema, **new_kwargs)
        for col, value in partition_cols_and_values.items():
            if col in table.schema.names or (
                columns is not None and col not in columns
            ):
                continue
            dtype = partition_dtypes.get(col)
            table = table.append_column(
                col,
                pyarrow.array([value for _ in range(len(table))], dtype),
            )
        return table
    except ArrowInvalid as e:
        glob_func = _get_glob_func(file_system)
        child_paths = {
            x
            for x in glob_func(str(_standardize_path(path) + "/**"))
            if not paths_are_equal(path, x)
        }
        if not len(child_paths):
            raise e
        elif any(x.endswith(".parquet") for x in child_paths):
            raise _FileIsPartitionedError(
                f"Cannot read partitioned files here for {path} with child paths {child_paths}"
            ) from e

        # return None to allow not being able to read empty directories that are hard to delete in gcs
        return None


class _FileIsPartitionedError(ValueError):
    pass


def _get_bounds_parquet(
    path: str | Path, file_system: GCSFileSystem, pandas_fallback: bool = False
) -> tuple[list[float], dict] | tuple[None, None]:
    try:
        return _get_bounds_parquet_from_open_file(path, file_system)
    except KeyError as e:
        if pandas_fallback and "geo" in str(e):
            return None, None
        raise e


def _get_bounds_parquet_from_open_file(
    file, file_system
) -> tuple[list[float], dict] | tuple[None, None]:
    geo_metadata = _get_geo_metadata_primary_column(file, file_system)

    if not geo_metadata:
        return None, None
    return geo_metadata["bbox"], geo_metadata["crs"]


def _get_geo_metadata(file, file_system) -> dict:
    try:
        meta = pq.read_schema(file).metadata
    except FileNotFoundError:
        meta = pq.ParquetDataset(file).schema.metadata
    return json.loads(meta[b"geo"])


def _get_geo_metadata_primary_column(file, file_system) -> dict:
    geo_metadata = _get_geo_metadata(file, file_system)
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
        # allow for 0 lengthed tables not to have geo metadata
        if not num_rows:
            return {}
    return {}


def _get_columns(path: str | Path, file_system: GCSFileSystem) -> pd.Index:
    # with file_system.open(path, "rb") as f:
    schema = pq.read_schema(path)
    index_cols = _get_index_cols(schema)
    return pd.Index(schema.names).difference(index_cols)


def _get_index_cols(schema: pyarrow.Schema) -> list[str]:
    cols = json.loads(schema.metadata[b"pandas"])["index_columns"]
    return [x for x in cols if not isinstance(x, dict)]


def get_bounds_series(
    paths: list[str | Path] | tuple[str | Path],
    file_system: GCSFileSystem | None = None,
    use_threads: bool = True,
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
        use_threads: Default True.
        pandas_fallback: If False (default), an exception is raised if the file has
            no geo metadata. If True, the geometry value is set to None for this file.

    Returns:
        A geopandas.GeoSeries with file paths as indexes and bounds as values.

    Examples:
    ---------
    >>> import sgis as sg
    >>> from gcsfs import GCSFileSystem
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

    if isinstance(paths, (str | Path)):
        paths = [paths]

    threads = (
        min(len(paths), int(multiprocessing.cpu_count() * 1.2)) or 1
        if use_threads
        else 1
    )

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
        raise ValueError(
            f"DataFrame must be GeoDataFrame. Got {type(df)} for {gcs_path}."
        )

    if not len(df) and get_child_paths(gcs_path, file_system):
        # no need to write empty df for partitioned parquet - if root dir exists
        return
    elif not len(df):
        if pandas_fallback:
            df = pd.DataFrame(df)
            df.geometry = df.geometry.astype(str)
            df.geometry = None
        try:
            file_format: str = Path(gcs_path).suffix.lstrip(".")
            write_method: Callable = getattr(df, f"to_{file_format}")
            if file_format == "parquet":
                kwargs["engine"] = "pyarrow"
            # with file_system.open(gcs_path, "wb") as file:
            write_method(gcs_path, **kwargs)

        except Exception as e:
            more_txt = PANDAS_FALLBACK_INFO if not pandas_fallback else ""
            raise e.__class__(
                f"{e.__class__.__name__}: {e} for {df}. " + more_txt
            ) from e
        return

    if isinstance(file_system, GCSFileSystem) and not str(gcs_path).startswith("gs://"):
        gcs_path = "gs://" + str(gcs_path)

    if ".parquet" in gcs_path or "prqt" in gcs_path:
        if partition_cols is not None:
            try:
                return _write_partitioned_geoparquet(
                    df,
                    gcs_path,
                    partition_cols,
                    file_system,
                    existing_data_behavior=existing_data_behavior,
                    write_func=_to_geopandas,
                    **kwargs,
                )
            except Exception as e:
                if (
                    file_system.exists(gcs_path)
                    and not list(file_system.ls(gcs_path))
                    and file_system.isfile(gcs_path)
                ):
                    _remove_file(gcs_path, file_system)
                return _write_partitioned_geoparquet(
                    df,
                    gcs_path,
                    partition_cols,
                    file_system,
                    existing_data_behavior=existing_data_behavior,
                    write_func=_to_geopandas,
                    **kwargs,
                )
        _to_geopandas(df, gcs_path, **kwargs)
        return

    return df.to_file(gcs_path, **kwargs)


def _to_geopandas(df, path, **kwargs) -> None:
    table = _geopandas_to_arrow(
        df,
        index=df.index,
        schema_version=None,
    )

    if "schema" in kwargs:
        schema = kwargs.pop("schema")
        partition_cols = [part.split("=")[0] for part in path if "=" in part]
        # make sure to get the actual metadata
        schema = pyarrow.schema(
            [
                (schema.field(col).name, schema.field(col).type)
                for col in schema.names
                if col not in partition_cols
            ],
            metadata=table.schema.metadata,
        )
        table = table.select(schema.names).cast(schema)
    pq.write_table(
        table,
        path,
        compression="snappy",
        flavor="hive",
        **kwargs,
    )


def _pyarrow_schema_from_geopandas(df: GeoDataFrame) -> pyarrow.Schema:
    geom_name = df.geometry.name
    pandas_columns = [col for col in df if col != geom_name]
    schema = pyarrow.Schema.from_pandas(df[pandas_columns], preserve_index=True)
    index_columns = _get_index_cols(schema)
    return pyarrow.schema(
        [
            (
                (schema.field(col).name, schema.field(col).type)
                if col != geom_name
                else (geom_name, pyarrow.binary())
            )
            for col in [*df.columns, *index_columns]
        ]
    )


def _remove_file(path, file_system) -> None:
    try:
        file_system.rm_file(str(path))
    except FileNotFoundError:
        return
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
    basename_template: str | None = None,
    **kwargs,
):
    if isinstance(partition_cols, str):
        partition_cols = [partition_cols]

    if kwargs.get("schema"):
        schema = kwargs.pop("schema")
    elif isinstance(df, GeoDataFrame):
        schema = _pyarrow_schema_from_geopandas(df)
    else:
        schema = pyarrow.Schema.from_pandas(df, preserve_index=True)

    table = _geopandas_to_arrow(
        df,
        index=df.index,
        schema_version=None,
    )
    # make sure to get the actual metadata
    schema = pyarrow.schema(
        [
            (schema.field(col).name, schema.field(col).type)
            for col in schema.names
            if col not in partition_cols
        ],
        metadata=table.schema.metadata,
    )

    glob_func = _get_glob_func(file_system)

    def as_partition_part(col: str, value: Any) -> str:
        value = value if not pd.isna(value) else NULL_VALUE
        return f"{col}={value}"

    paths: list[Path] = []
    dfs: list[DataFrame] = []
    for group, rows in df.groupby(partition_cols, dropna=False):
        partition_parts = "/".join(
            as_partition_part(col, value)
            for col, value in zip(partition_cols, group, strict=True)
        )
        paths.append(_standardize_path(path) + f"/{partition_parts}")
        dfs.append(rows)

    def threaded_write(rows: DataFrame, path: str) -> None:
        if basename_template is None:
            this_basename = (uuid.uuid4().hex + "-{i}.parquet").replace("-{i}", "0")
        else:
            this_basename = basename_template.replace("-{i}", "0")
        for i, sibling_path in enumerate(
            sorted(glob_func(str(_standardize_path(path) + "/**")))
        ):
            if paths_are_equal(sibling_path, path):
                continue
            if existing_data_behavior == "delete_matching":
                _remove_file(sibling_path, file_system)
            elif existing_data_behavior == "error":
                raise pyarrow.ArrowInvalid(
                    f"Could not write to {path} as the directory is not empty and existing_data_behavior is to error"
                )
            else:
                this_basename = basename_template.replace("-{i}", str(i + 1))

        out_path = str(_standardize_path(path) + "/" + this_basename)
        try:
            write_func(rows, out_path, schema=schema, **kwargs)
        except FileNotFoundError:
            parent = "/".join(out_path.split("/")[:-1])
            file_system.makedirs(parent, exist_ok=True)
            for sibling_path in sorted(glob_func(str(_standardize_path(path) + "/**"))):
                if paths_are_equal(sibling_path, path):
                    continue
                if existing_data_behavior == "delete_matching":
                    _remove_file(sibling_path, file_system)
            write_func(rows, out_path, schema=schema, **kwargs)

    with ThreadPoolExecutor() as executor:
        list(executor.map(threaded_write, dfs, paths))

    a_partition_col_is_string_type_but_all_numeric_values = any(
        func(df[col]) and df[col].dropna().str.replace(".", "").str.isnumeric().all()
        for col in partition_cols
        for func in [
            pd.api.types.is_string_dtype,
            pd.api.types.is_object_dtype,
        ]
    )
    if not a_partition_col_is_string_type_but_all_numeric_values:
        return

    new_path = str(path)
    for col in partition_cols:
        new_path += f"/{col}=this_will_force_str_dtype"
        file_system.makedirs(new_path, exist_ok=True)
        new_path += "/this_will_force_str_dtype.parquet"
        pd.DataFrame({col: [] for col in df}).to_parquet(new_path)


def _get_glob_func(file_system) -> functools.partial:
    try:
        return functools.partial(file_system.glob)
    except AttributeError:
        return functools.partial(glob.glob, recursive=True)


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
    ---------
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
    # keep only the parts in between the two .parquet parts
    path = str(path).split(".parquet")[1]
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
        if "No match for FieldRef" in str(e):
            # if a non-partition col is used in 'filters',
            # we cannot determine if the expression match without reading the file
            return True
        raise e
    return bool(len(table))


def _read_geopandas_single_path(
    file,
    read_func: Callable,
    file_format: str,
    **kwargs,
):
    try:
        return read_func(file, **kwargs)
    except ValueError as e:
        if "Missing geo metadata" not in str(e) and "geometry" not in str(e):
            raise e.__class__(f"{e.__class__.__name__}: {e} for {file}. ") from e
        df = getattr(pd, f"read_{file_format}")(file, **kwargs)
        if not len(df):
            return GeoDataFrame(df)
        raise e.__class__(f"{e.__class__.__name__}: {e} for {df}. ") from e
    except Exception as e:
        raise e.__class__(f"{e.__class__.__name__}: {e} for {file}.") from e


def _read_partitioned_parquet(
    path: str,
    filters=None,
    file_system=None,
    mask=None,
    child_paths: list[str] | None = None,
    use_threads: bool = True,
    to_geopandas: bool = True,
    **kwargs,
):
    file_system = _get_file_system(file_system, kwargs)

    partition_cols = [
        part.split("=")[0]
        for part in Path(child_paths[0]).parts
        if "=" in part and part not in path
    ]
    base_parts = Path(path).parts
    partition_parts = [
        [Path(path).parts[i] for path in child_paths]
        for i, part in enumerate(Path(child_paths[0]).parts)
        if "=" in part and part not in base_parts
    ]

    def infer_dtype(values: list[str]) -> pyarrow.DataType:
        values = np.array(values)
        try:
            values = values.astype(np.int32)
            return pyarrow.int32()
        except ValueError:
            try:
                values = values.astype(np.float32)
                return pyarrow.float32()
            except Exception:
                return pyarrow.string()

    partition_dtypes: dict[str, pyarrow.DataType] = {
        next(iter(parts)).split("=")[0]: infer_dtype([x.split("=")[-1] for x in parts])
        for parts in partition_parts
    }
    partitioning = pyarrow.dataset.partitioning(
        pyarrow.schema(
            [pyarrow.field(col, dtype) for col, dtype in partition_dtypes.items()]
        ),
        flavor="hive",
    )

    filters = _filters_to_expression(filters)

    filtered_child_paths = [
        path
        for path in child_paths
        if filters is None or expression_match_path(filters, path)
    ]

    base_parts = Path(path).parts

    if mask is not None:
        intersections = sfilter(
            get_bounds_series(filtered_child_paths, pandas_fallback=True)[
                lambda x: (x.notna()) & (~x.is_empty)
            ],
            mask,
        )
        if not len(intersections):
            # add columns to empty DataFrame
            first_path = next(iter(filtered_child_paths + [path]))
            _, crs = _get_bounds_parquet(first_path, file_system)
            df = GeoDataFrame(columns=_get_columns(first_path, file_system), crs=crs)
            if kwargs.get("columns"):
                return df[list(kwargs["columns"])]
            return df

        filtered_child_paths = list(intersections.index)
        filters_from_mask = [
            (
                part.split("=")[0],
                "in",
                [Path(path).parts[i].split("=")[-1] for path in filtered_child_paths],
            )
            for i, part in enumerate(Path(filtered_child_paths[0]).parts)
            if "=" in part and part not in base_parts
        ]
        filters_from_mask = _filters_to_expression(filters_from_mask)
        if filters is not None:
            filters &= filters_from_mask
        else:
            filters = filters_from_mask

    schema = kwargs.pop("schema", get_schema(path))
    if not any(col in partition_cols for col in schema.names):
        # Note that schema is not passed to read_parquet because the partition_cols are not part of the schema, meaning they get left out if specified
        return gpd.read_parquet(
            path,
            filters=filters,
            use_threads=use_threads,
            partitioning=partitioning,
            **kwargs,
        )

    results: list[pyarrow.Table] = _read_pyarrow_with_threads(
        (
            path
            for path in filtered_child_paths
            if filters is None or expression_match_path(filters, path)
        ),
        file_system=file_system,
        filters=filters,
        use_threads=use_threads,
        schema=schema,
        **kwargs,
    )

    if results and to_geopandas:
        return _concat_pyarrow_to_geopandas(results, filtered_child_paths, file_system)
    elif results:
        return pyarrow.concat_tables(results, promote_options="permissive").to_pandas()

    # add columns to empty DataFrame
    first_path = next(iter(child_paths + [path]))
    _, crs = _get_bounds_parquet(first_path, file_system)
    df = GeoDataFrame(columns=_get_columns(first_path, file_system), crs=crs)
    if kwargs.get("columns"):
        return df[list(kwargs["columns"])]
    return df


def _concat_pyarrow_to_geopandas(
    results: list[pyarrow.Table], paths: list[str], file_system: Any
):
    dfs = [x for x in results if isinstance(x, pd.DataFrame)]
    results = _concat_pyarrow_tables(
        [x for x in results if not isinstance(x, pd.DataFrame)],
        promote_options="permissive",
    )
    geo_metadata = _get_geo_metadata(next(iter(paths)), file_system)
    df = _arrow_to_geopandas(results, geo_metadata)
    if dfs:
        return pd.concat([df, *dfs])
    return df


def paths_are_equal(path1: Path | str, path2: Path | str) -> bool:
    return Path(path1).parts == Path(path2).parts


def get_child_paths(path, file_system, pattern: str = "/**/*.parquet") -> list[str]:
    glob_func = _get_glob_func(file_system)
    paths = [x for x in glob_func(str(_standardize_path(path) + pattern))]
    if str(path).startswith("gs://"):
        paths = ["gs://" + str(x).replace("gs://", "") for x in paths]
    paths = [x for x in paths if not paths_are_equal(x, path)]
    return paths


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
