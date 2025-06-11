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
from typing import Any

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
from geopandas.io.arrow import _arrow_to_geopandas
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

    if not isinstance(gcs_path, (str | Path | os.PathLike)):
        return _read_geopandas_from_iterable(
            gcs_path,
            mask=mask,
            file_system=file_system,
            use_threads=use_threads,
            pandas_fallback=pandas_fallback,
            filters=filters,
            **kwargs,
        )

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
            glob_func = _get_glob_func(file_system)
            suffix: str = Path(gcs_path).suffix
            paths = glob_func(str(Path(gcs_path) / expression / f"*{suffix}"))
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

    if gcs_path.endswith(".parquet"):
        file_format: str = "parquet"
        read_func = gpd.read_parquet
    else:
        file_format: str = Path(gcs_path).suffix.lstrip(".")
        read_func = gpd.read_file

    with file_system.open(gcs_path, mode="rb") as file:
        return _read_geopandas_single_path(
            file,
            read_func=read_func,
            file_format=file_format,
            filters=filters,
            **kwargs,
        )


def _read_geopandas_from_iterable(
    paths, mask, file_system, use_threads, pandas_fallback, **kwargs
):
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

    results: list[pyarrow.Table] = _read_pyarrow_with_treads(
        paths,
        file_system=file_system,
        mask=mask,
        use_threads=use_threads,
        **kwargs,
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


def _read_pyarrow_with_treads(
    paths: list[str | Path | os.PathLike],
    file_system,
    use_threads,
    mask,
    filters,
    **kwargs,
) -> list[pyarrow.Table]:
    read_partial = functools.partial(
        _read_pyarrow, filters=filters, mask=mask, file_system=file_system, **kwargs
    )
    if not use_threads:
        return [x for x in map(read_partial, paths) if x is not None]
    with ThreadPoolExecutor() as executor:
        return [x for x in executor.map(read_partial, paths) if x is not None]


def intersects(file, mask, file_system) -> bool:
    bbox, _ = _get_bounds_parquet_from_open_file(file, file_system)
    return shapely.box(*bbox).intersects(to_shapely(mask))


def _read_pyarrow(path: str, file_system, mask=None, **kwargs) -> pyarrow.Table | None:
    try:
        with file_system.open(path, "rb") as file:
            if mask is not None and not intersects(file, mask, file_system):
                return

            # 'get' instead of 'pop' because dict is mutable
            schema = kwargs.get("schema", pq.read_schema(file))
            new_kwargs = {
                key: value for key, value in kwargs.items() if key != "schema"
            }

            return pq.read_table(file, schema=schema, **new_kwargs)
    except ArrowInvalid as e:
        glob_func = _get_glob_func(file_system)
        if not len(
            {
                x
                for x in glob_func(str(Path(path) / "**"))
                if not paths_are_equal(path, x)
            }
        ):
            raise e
        # allow not being able to read empty directories that are hard to delete in gcs


def _get_bounds_parquet(
    path: str | Path, file_system: GCSFileSystem, pandas_fallback: bool = False
) -> tuple[list[float], dict] | tuple[None, None]:
    with file_system.open(path, "rb") as file:
        return _get_bounds_parquet_from_open_file(file, file_system)


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
        try:
            with file_system.open(file, "rb") as f:
                meta = pq.read_schema(f).metadata
        except Exception as e:
            raise e.__class__(f"{file}: {e}") from e
    except Exception as e:
        raise e.__class__(f"{file}: {e}") from e

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

    threads = (
        min(len(paths), int(multiprocessing.cpu_count())) or 1 if use_threads else 1
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
    file_system = _get_file_system(file_system, kwargs)

    if basename_template is None:
        basename_template = uuid.uuid4().hex + "-{i}.parquet"

    if isinstance(partition_cols, str):
        partition_cols = [partition_cols]

    for col in partition_cols:
        if df[col].isna().all() and not kwargs.get("schema"):
            raise ValueError("Must specify 'schema' when all rows are NA.")

    glob_func = _get_glob_func(file_system)

    if file_system.exists(path) and file_system.isfile(path):
        _remove_file(path, file_system)

    if kwargs.get("schema"):
        schema = kwargs.pop("schema")
    elif isinstance(df, GeoDataFrame):
        schema = _pyarrow_schema_from_geopandas(df)
    else:
        schema = pyarrow.Schema.from_pandas(df, preserve_index=True)

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
        paths.append(Path(path) / partition_parts)
        dfs.append(rows)

    def threaded_write(rows: DataFrame, path: str) -> None:
        this_basename = basename_template.replace("-{i}", "0")
        for i, sibling_path in enumerate(sorted(glob_func(str(Path(path) / "**")))):
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

        out_path = str(Path(path) / this_basename)
        try:
            with file_system.open(out_path, mode="wb") as file:
                write_func(rows, file, schema=schema, **kwargs)
        except FileNotFoundError:
            file_system.makedirs(str(path), exist_ok=True)
            with file_system.open(out_path, mode="wb") as file:
                write_func(rows, file, schema=schema, **kwargs)

    with ThreadPoolExecutor() as executor:
        executor.map(threaded_write, dfs, paths)


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


def _read_pandas(gcs_path: str, use_threads: bool = True, **kwargs):
    file_system = _get_file_system(None, kwargs)

    if not isinstance(gcs_path, (str | Path | os.PathLike)):
        results: list[pyarrow.Table] = _read_pyarrow_with_treads(
            gcs_path,
            file_system=file_system,
            mask=None,
            use_threads=use_threads,
            **kwargs,
        )
        results = pyarrow.concat_tables(results, promote_options="permissive")
        return results.to_pandas()

    child_paths = get_child_paths(gcs_path, file_system)
    if child_paths:
        return _read_partitioned_parquet(
            gcs_path,
            file_system=file_system,
            mask=None,
            child_paths=child_paths,
            use_threads=use_threads,
            to_geopandas=False,
            **kwargs,
        )

    with file_system.open(gcs_path, "rb") as file:
        return pd.read_parquet(file, **kwargs)


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
    glob_func = _get_glob_func(file_system)

    if child_paths is None:
        child_paths = list(glob_func(str(Path(path) / "**/*.parquet")))

    filters = _filters_to_expression(filters)

    results: list[pyarrow.Table] = _read_pyarrow_with_treads(
        (
            path
            for path in child_paths
            if filters is None or expression_match_path(filters, path)
        ),
        file_system=file_system,
        mask=mask,
        filters=filters,
        use_threads=use_threads,
        **kwargs,
    )

    if results and to_geopandas:
        return _concat_pyarrow_to_geopandas(results, child_paths, file_system)
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
    results = pyarrow.concat_tables(
        results,
        promote_options="permissive",
    )
    geo_metadata = _get_geo_metadata(next(iter(paths)), file_system)
    return _arrow_to_geopandas(results, geo_metadata)


def paths_are_equal(path1: Path | str, path2: Path | str) -> bool:
    return Path(path1).parts == Path(path2).parts


def get_child_paths(path, file_system) -> list[str]:
    glob_func = _get_glob_func(file_system)
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
