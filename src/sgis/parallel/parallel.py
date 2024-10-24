import functools
import inspect
import itertools
import multiprocessing
import pickle
import warnings
from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pandas.api.types import is_array_like

try:
    import dapla as dp
except ImportError:
    pass

import joblib
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from pandas import DataFrame
from pandas import Series

from ..geopandas_tools.neighbors import get_neighbor_indices
from ..geopandas_tools.overlay import clean_overlay
from ..helpers import LocalFunctionError
from ..helpers import dict_zip_union
from ..helpers import in_jupyter

try:
    from ..io.dapla_functions import read_geopandas
    from ..io.dapla_functions import write_geopandas

except ImportError:
    pass


try:
    from dapla import read_pandas
    from dapla import write_pandas
except ImportError:
    pass


def parallel_overlay(
    df1: GeoDataFrame,
    df2: GeoDataFrame,
    processes: int,
    how: str = "intersection",
    max_rows_per_chunk: int | None = None,
    backend: str = "loky",
    to_print: str | None = None,
    **kwargs,
) -> GeoDataFrame:
    """Perform spatial overlay operations on two GeoDataFrames in parallel.

    This function splits the first GeoDataFrame into chunks, processes each chunk in parallel using the specified
    overlay operation with the second GeoDataFrame, and then concatenates the results.

    Note that this function is most useful if df2 has few and simple geometries.

    Args:
        df1: The first GeoDataFrame for the overlay operation.
        df2: The second GeoDataFrame for the overlay operation.
        how: Type of overlay operation ('intersection', 'union', etc.).
        processes: Number of parallel processes to use.
        max_rows_per_chunk: Maximum number of rows per chunk for processing. This helps manage memory usage.
        backend: The parallelization backend to use ('loky', 'multiprocessing', 'threading').
        to_print: Optional text to print to see progression.
        **kwargs: Additional keyword arguments to pass to the overlay function.

    Returns:
        A GeoDataFrame containing the result of the overlay operation.
    """
    return pd.concat(
        chunkwise(
            _clean_overlay_with_print,
            df1,
            kwargs={
                "df2": df2,
                # "to_print": to_print,
                "how": how,
            }
            | kwargs,
            processes=processes,
            max_rows_per_chunk=max_rows_per_chunk,
            backend=backend,
        ),
        ignore_index=True,
    )


def parallel_overlay_rowwise(
    df1: GeoDataFrame,
    df2: GeoDataFrame,
    processes: int,
    max_rows_per_chunk: int | None = None,
    backend: str = "loky",
    to_print: str | None = None,
    **kwargs,
) -> GeoDataFrame:
    """Perform spatial clip on two GeoDataFrames in parallel.

    This function splits the first GeoDataFrame into chunks, processes each chunk in parallel using the specified
    overlay operation with the second GeoDataFrame, and then concatenates the results.

    Note that this function is most useful if df2 has few and simple geometries.

    Args:
        df1: The first GeoDataFrame for the overlay operation.
        df2: The second GeoDataFrame for the overlay operation.
        how: Type of overlay operation ('intersection', 'union', etc.).
        processes: Number of parallel processes to use.
        max_rows_per_chunk: Maximum number of rows per chunk for processing. This helps manage memory usage.
        backend: The parallelization backend to use ('loky', 'multiprocessing', 'threading').
        to_print: Optional text to print to see progression.
        **kwargs: Additional keyword arguments to pass to the overlay function.

    Returns:
        A GeoDataFrame containing the result of the overlay operation.
    """
    return pd.concat(
        chunkwise(
            _clip_rowwise,
            df1,
            kwargs={
                "df2": df2,
                "to_print": to_print,
            }
            | kwargs,
            processes=processes,
            max_rows_per_chunk=max_rows_per_chunk,
            backend=backend,
        ),
        ignore_index=True,
    )


def parallel_sjoin(
    df1: GeoDataFrame,
    df2: GeoDataFrame,
    processes: int,
    max_rows_per_chunk: int | None = None,
    backend: str = "loky",
    to_print: str | None = None,
    **kwargs,
) -> GeoDataFrame:
    """Perform spatial clip on two GeoDataFrames in parallel.

    This function splits the first GeoDataFrame into chunks, processes each chunk in parallel using the specified
    overlay operation with the second GeoDataFrame, and then concatenates the results.

    Note that this function is most useful if df2 has few and simple geometries.

    Args:
        df1: The first GeoDataFrame for the overlay operation.
        df2: The second GeoDataFrame for the overlay operation.
        how: Type of overlay operation ('intersection', 'union', etc.).
        processes: Number of parallel processes to use.
        max_rows_per_chunk: Maximum number of rows per chunk for processing. This helps manage memory usage.
        backend: The parallelization backend to use ('loky', 'multiprocessing', 'threading').
        to_print: Optional text to print to see progression.
        **kwargs: Additional keyword arguments to pass to the overlay function.

    Returns:
        A GeoDataFrame containing the result of the overlay operation.
    """
    return pd.concat(
        chunkwise(
            _sjoin_within_first,
            df1,
            kwargs={
                "df2": df2,
                "to_print": to_print,
            }
            | kwargs,
            processes=processes,
            max_rows_per_chunk=max_rows_per_chunk,
            backend=backend,
        ),
        ignore_index=True,
    )


def _sjoin_within_first(
    df1, df2, to_print: str | None = None, predicate: str = "intersects", **kwargs
):
    if to_print:
        print(to_print, "- sjoin chunk len:", len(df1))

    df2 = df2.reset_index(drop=True)
    df2["_from_df2"] = 1
    df1["_range_idx"] = range(len(df1))
    joined = df1.sjoin(df2, predicate="within", how="left")
    within = joined.loc[joined["_from_df2"].notna()].drop(
        columns=["_from_df2", "_range_idx", "index_right"], errors="raise"
    )
    not_within = df1.loc[
        df1["_range_idx"].isin(joined.loc[joined["_from_df2"].isna(), "_range_idx"])
    ]

    return pd.concat(
        [
            within,
            not_within.sjoin(df2, predicate=predicate, **kwargs),
        ],
        ignore_index=True,
    )


def _clip_rowwise(df1, df2, to_print: str | None = None):
    geom_col = df2.geometry.name

    def clip_by_one_row(i):
        this: pd.Series = df2.iloc[i]
        clipped = df1.clip(this[geom_col])
        without_geom_col = this.drop(geom_col)
        clipped.loc[:, without_geom_col.index] = without_geom_col.values
        if to_print:
            print(i, to_print, len(clipped))
        return clipped

    return pd.concat([clip_by_one_row(i) for i in range(len(df2))], ignore_index=True)


def _clean_overlay_with_print(
    df1: GeoDataFrame,
    df2: GeoDataFrame,
    how: str = "intersection",
    to_print: str | None = None,
    **kwargs,
) -> GeoDataFrame:
    if to_print:
        print(to_print, f"- {how} chunk len:", len(df1))
    return clean_overlay(df1, df2, how=how, **kwargs)


class Parallel:
    """Run functions in parallell.

    The main method is 'map', which runs a single function for
    each item of an iterable. If the items of the iterable also are iterables,
    starmap can be used.

    The class also provides functions for reading and writing files in parallell
    in dapla.

    Note that nothing gets printed during execution if running in a notebook.
    Tip for debugging: set processes=1 to run without parallelization.

    Note that when using the default backend 'multiprocessing', all code except for
    imports and functions should be guarded by 'if __name__ == "__main__"' to not cause
    an eternal loop. This is not the case if setting backend to 'loky'. See joblib's
    documentation: https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation

    Args:
        processes: Number of parallel processes. Set to 1 to run without
            parallelization.
        backend: Defaults to "multiprocessing". Other options are 'loky' and 'threading',
            offered through joblib's Parallel class.
        context: Start method for the processes. Defaults to 'spawn'
            to avoid frozen processes.
        maxtasksperchild: Number of tasks a worker process can complete before
            it will exit and be replaced with a fresh worker process, to enable
            unused resources to be freed. Defaults to 10 to
        **kwargs: Keyword arguments to be passed to either
            multiprocessing.Pool or joblib.Parallel, depending
            on the backend. Not to be confused with the kwargs passed to functions in
            the map and starmap methods.
    """

    def __init__(
        self,
        processes: int,
        backend: str = "multiprocessing",
        context: str = "spawn",
        maxtasksperchild: int = 10,
        chunksize: int = 1,
        **kwargs,
    ) -> None:
        """Initialize a Parallel instance with specified settings for parallel execution.

        Args:
            processes: Number of parallel processes. Set to 1 to run without parallelization.
            backend: The backend to use for parallel execution. Defaults to 'multiprocessing'.
            context: The context setting for multiprocessing. Defaults to 'spawn'.
            maxtasksperchild: The maximum number of tasks a worker process can complete
                before it is replaced. Defaults to 10.
            chunksize: The size of the chunks of the iterable to distribute to workers.
            **kwargs: Additional keyword arguments passed to the underlying parallel execution backend.
        """
        self.processes = int(processes)
        self.maxtasksperchild = maxtasksperchild
        self.chunksize = chunksize
        self.backend = backend
        self.context = context
        self.kwargs = kwargs
        self.funcs: list[functools.partial] = []
        self.results: list[Any] = []

    def map(
        self,
        func: Callable,
        iterable: Collection,
        args: tuple | None = None,
        kwargs: dict | None = None,
    ) -> list[Any]:
        """Run functions in parallel with items of an iterable as 0th arguemnt.

        Args:
            func: Function to be run.
            iterable: An iterable where each item will be passed to func as
                0th positional argument.
            args: Positional arguments passed to 'func' starting from the 1st argument.
                The 0th argument will be reserved for the values of 'iterable'.
            kwargs: Keyword arguments passed to 'func'. Must be passed as a dict,
                not unpacked into separate keyword arguments.

        Returns:
            A list of the return values of the function, one for each item in
            'iterable'.

        Examples:
        ---------
        Multiply each list element by 2.

        >>> iterable = [1, 2, 3]
        >>> def x2(x):
        ...     return x * 2
        >>> p = sg.Parallel(4, backend="loky")
        >>> results = p.map(x2, iterable)
        >>> results
        [2, 4, 6]

        With args and kwargs.

        >>> iterable = [1, 2, 3]
        >>> def x2(x, plus, minus):
        ...     return x * 2 + plus - minus
        >>> p = sg.Parallel(4, backend="loky")
        ...
        >>> # these three are the same
        >>> results1 = p.map(x2, iterable, args=(2, 1))
        >>> results2 = p.map(x2, iterable, kwargs=dict(plus=2, minus=1))
        >>> results3 = p.map(x2, iterable, args=(2,), kwargs=dict(minus=1))
        >>> assert results1 == results2 == results3
        ...
        >>> results1
        [3, 5, 7]

        If in Jupyter the function should be defined in another module.
        And if using the multiprocessing backend, the code should be
        guarded by if __name__ == "__main__".

        >>> from .file import x2
        >>> if __name__ == "__main__":
        ...     p = sg.Parallel(4, backend="loky")
        ...     results = p.map(x2, iterable)
        ...     print(results)
        [2, 4, 6]
        """
        if args:
            # start at index 1, meaning the 0th argument (the iterable) is still available
            args_as_kwargs = _turn_args_into_kwargs(func, args, index_start=1)
        else:
            args_as_kwargs = {}

        self._validate_execution(func)

        kwargs = self._validate_kwargs(kwargs) | args_as_kwargs

        func_with_kwargs = functools.partial(func, **kwargs)

        if self.processes == 1:
            return [func_with_kwargs(item) for item in iterable]

        iterable = list(iterable)

        # don't use unnecessary processes
        processes = min(self.processes, len(iterable))

        if not processes:
            return []
        elif processes == 1:
            return [func_with_kwargs(item) for item in iterable]

        try:
            if self.backend == "multiprocessing":
                with multiprocessing.get_context(self.context).Pool(
                    processes, maxtasksperchild=self.maxtasksperchild, **self.kwargs
                ) as pool:
                    try:
                        return pool.map(
                            func_with_kwargs, iterable, chunksize=self.chunksize
                        )
                    except Exception as e:
                        pool.terminate()
                        raise e

            with joblib.Parallel(
                n_jobs=processes, backend=self.backend, **self.kwargs
            ) as parallel:
                return parallel(
                    joblib.delayed(func)(item, **kwargs) for item in iterable
                )
        except pickle.PickleError as e:
            unpicklable = []
            for k, v in locals().items():
                try:
                    pickle.dumps(v)
                except pickle.PickleError:
                    unpicklable.append(k)
                except TypeError:
                    pass
            if unpicklable:
                raise pickle.PickleError(
                    f"Cannot unpickle objects: {unpicklable}"
                ) from e
            raise e

    def starmap(
        self,
        func: Callable,
        iterable: Collection[Iterable[Any]],
        args: tuple | None = None,
        kwargs: dict | None = None,
    ) -> list[Any]:
        """Run functions in parallel where items of the iterable are unpacked.

        This requires the items of the iterable to be iterables as well. See
        https://docs.python.org/3/library/itertools.html#itertools.starmap

        Args:
            func: Function to be run.
            iterable: An iterable of iterables, where each item will be
                unpacked as positional argument to the function.
            args: Positional arguments passed to 'func' starting at argument position
                n + 1, where n is the length of the iterables inside the iterable.
            kwargs: Keyword arguments passed to 'func'. Must be passed as a dict,
                not unpacked into separate keyword arguments.

        Returns:
            A list of the return values of the function, one for each item in
            'iterable'.

        Examples:
        ---------
        Multiply each list element by 2.

        >>> iterable = [(1, 2), (2, 3), (3, 4)]
        >>> def add(x, y):
        ...     return x + y
        >>> p = sg.Parallel(3, backend="loky")
        >>> results = p.starmap(add, iterable)
        >>> results
        [3, 5, 7]

        With args and kwargs. Since the iterables inside 'iterable' are of length 2,
        'args' will start at argument number three, e.i. 'c'.

        >>> iterable = [(1, 2), (2, 3), (3, 4)]
        >>> def add(a, b, c, *, d):
        ...     return a + b + c + d
        >>> p = sg.Parallel(3, backend="loky")
        >>> results = p.starmap(add, iterable, args=(1,), kwargs={"d": 0.1})
        >>> results
        [4.1, 6.1, 8.1]

        If in Jupyter the function should be defined in another module.
        And if using the multiprocessing backend, the code should be
        guarded by if __name__ == "__main__".

        >>> from .file import x2
        >>> if __name__ == "__main__":
        ...     p = sg.Parallel(4, backend="loky")
        ...     results = p.starmap(add, iterable)
        ...     print(results)
        [3, 5, 7]

        """
        if args:
            # starting the count at the length of the iterables inside the iterables
            iterable = list(iterable)
            args_as_kwargs = _turn_args_into_kwargs(
                func, args, index_start=len(iterable[0])
            )
        else:
            args_as_kwargs = {}

        self._validate_execution(func)

        kwargs = self._validate_kwargs(kwargs) | args_as_kwargs

        func_with_kwargs = functools.partial(func, **kwargs)

        if self.processes == 1:
            return list(itertools.starmap(func_with_kwargs, iterable))

        iterable = list(iterable)

        # don't use unnecessary processes
        processes = min(self.processes, len(iterable))

        if not processes:
            return []

        if self.backend == "multiprocessing":
            with multiprocessing.get_context(self.context).Pool(
                processes, maxtasksperchild=self.maxtasksperchild, **self.kwargs
            ) as pool:
                try:
                    return pool.starmap(
                        func_with_kwargs, iterable, chunksize=self.chunksize
                    )
                except Exception as e:
                    pool.terminate()
                    raise e

        with joblib.Parallel(
            n_jobs=processes, backend=self.backend, **self.kwargs
        ) as parallel:
            return parallel(joblib.delayed(func)(*item, **kwargs) for item in iterable)

    def read_pandas(
        self,
        files: list[str],
        concat: bool = True,
        ignore_index: bool = True,
        strict: bool = True,
        **kwargs,
    ) -> DataFrame | list[DataFrame]:
        """Read tabular files from a list in parallel.

        Args:
            files: List of file paths.
            concat: Whether to concat the results to a DataFrame.
            ignore_index: Defaults to True.
            strict: If True (default), all files must exist.
            **kwargs: Keyword arguments passed to dapla.read_pandas.

        Returns:
            A DataFrame, or a list of DataFrames if concat is False.
        """
        if strict:
            res = self.map(read_pandas, files, kwargs=kwargs)
        else:
            res = self.map(_try_to_read_pandas, files, kwargs=kwargs)

        return pd.concat(res, ignore_index=ignore_index) if concat else res

    def read_geopandas(
        self,
        files: list[str],
        concat: bool = True,
        ignore_index: bool = True,
        strict: bool = True,
        **kwargs,
    ) -> GeoDataFrame | list[GeoDataFrame]:
        """Read geospatial files from a list in parallel.

        Args:
            files: List of file paths.
            concat: Whether to concat the results to a GeoDataFrame.
            ignore_index: Defaults to True.
            strict: If True (default), all files must exist.
            chunksize: The size of the chunks of the iterable to distribute to workers.
            **kwargs: Keyword arguments passed to sgis.read_geopandas.

        Returns:
            A GeoDataFrame, or a list of GeoDataFrames if concat is False.
        """
        if "file_system" not in kwargs:
            kwargs["file_system"] = dp.FileClient.get_gcs_file_system()

        if strict:
            res = self.map(read_geopandas, files, kwargs=kwargs)
        else:
            res = self.map(_try_to_read_geopandas, files, kwargs=kwargs)

        return pd.concat(res, ignore_index=ignore_index) if concat else res

    def write_municipality_data(
        self,
        in_data: dict[str, str | GeoDataFrame],
        out_data: str | dict[str, str],
        municipalities: GeoDataFrame,
        with_neighbors: bool = False,
        funcdict: dict[str, Callable] | None = None,
        file_type: str = "parquet",
        muni_number_col: str = "KOMMUNENR",
        strict: bool = False,
        write_empty: bool = False,
        id_assign_func: Callable | functools.partial = clean_overlay,
        verbose: bool = True,
    ) -> None:
        """Split multiple datasets into municipalities and write as separate files.

        The files will be named as the municipality number.
        Each dataset in 'in_data' is intersected with 'municipalities'
        in parallel. The intersections themselves can also be run in parallel
        with the 'processes_in_clip' argument.


        Args:
            in_data: Dictionary with dataset names as keys and file paths or
                (Geo)DataFrames as values. Note that the files will be read
                in parallel if file paths are used.
            out_data: Either a single folder path or a dictionary with same keys as
                'in_data' and folder paths as values. If a single folder is passed,
                the 'in_data' keys will be used as subfolders.
            municipalities: GeoDataFrame of municipalities (or similar) of which to
                split the data by.
            with_neighbors: If True, the resulting data will include
                neighbor municipalities, as well as the munipality itself.
                Defaults to False.
            funcdict: Dictionary with the keys of 'in_data' and functions as values.
                The functions should take a GeoDataFrame as input and return a
                GeoDataFrame. The function will be excecuted before the right after
                the data is read.
            file_type: Defaults to parquet.
            muni_number_col: String column name with municipality
                number/identifier. Defaults to KOMMUNENR. If the column is not present
                in the data to be split, the data will be intersected with the
                municipalities.
            strict: If False (default), the dictionaries 'out_data' and 'funcdict' does
                not have to have the same length as 'in_data'.
            write_empty: If False (default), municipalities with no data will be skipped.
                If True, an empty parquet file will be written.
            id_assign_func: Function to assign ids (e.g. municipality number) to
                the dataframe for missing values.
            verbose: Whether to print during execution.

        """
        shared_kwds = {
            "municipalities": municipalities,
            "file_type": file_type,
            "muni_number_col": muni_number_col,
            "write_empty": write_empty,
            "with_neighbors": with_neighbors,
            "strict": strict,
            "verbose": verbose,
            "id_assign_func": id_assign_func,
        }

        if isinstance(out_data, (str, Path)):
            out_data = {name: Path(out_data) / name for name in in_data}

        if funcdict is None:
            funcdict = {}

        fs = dp.FileClient.get_gcs_file_system()

        for _, data, folder, postfunc in dict_zip_union(in_data, out_data, funcdict):
            if data is None or (
                not strict and isinstance(data, (str | Path)) and not fs.exists(data)
            ):
                continue

            kwds = shared_kwds | {
                "data": data,
                "func": postfunc,
                "out_folder": folder,
            }
            partial_func = functools.partial(write_municipality_data, **kwds)
            self.funcs.append(partial_func)

        return self._execute()

    def chunkwise(
        self,
        func: Callable,
        iterable: Collection[Iterable[Any]],
        args: tuple | None = None,
        kwargs: dict | None = None,
        max_rows_per_chunk: int | None = None,
    ) -> Collection[Iterable[Any]]:
        """Run a function in parallel on chunks of a (Geo)DataFrame.

        Args:
            func: Function to run chunkwise. It should take
                (a chunk of) the iterable as first argument.
            iterable: Iterable to split in chunks and passed
                as first argument to 'func'.
            args: Positional arguments in 'func' after the DataFrame.
            kwargs: Additional keyword arguments in 'func'.
            max_rows_per_chunk: Alternatively decide number of chunks
                by a maximum number of rows per chunk.
        """
        return chunkwise(
            func,
            iterable,
            args=args,
            kwargs=kwargs,
            processes=self.processes,
            max_rows_per_chunk=max_rows_per_chunk,
            backend=self.backend,
        )

    def _validate_execution(self, func: Callable) -> None:
        """Multiprocessing doesn't work with local variables in interactive interpreter.

        Raising Exception to avoid confusion.
        """
        if (
            func.__module__ == "__main__"
            and self.context == "spawn"
            and self.backend == "multiprocessing"
            and in_jupyter()
        ):
            raise LocalFunctionError(func)

    @staticmethod
    def _validate_kwargs(kwargs: dict) -> dict:
        """Make sure kwargs is a dict (not ** unpacked or None)."""
        if kwargs is None:
            kwargs = {}
        elif not isinstance(kwargs, dict):
            raise TypeError("kwargs must be a dict")
        return kwargs

    def _execute(self) -> list[Any]:
        [self._validate_execution(func) for func in self.funcs]

        if self.processes == 1:
            return [func() for func in self.funcs]

        # don't use unnecessary processes
        if self.processes > len(self.funcs):
            processes = len(self.funcs)
        else:
            processes = self.processes

        if not processes:
            return []

        if self.backend != "multiprocessing":
            with joblib.Parallel(
                n_jobs=processes, backend=self.backend, **self.kwargs
            ) as parallel:
                return parallel(joblib.delayed(func)() for func in self.funcs)

        with multiprocessing.get_context(self.context).Pool(
            processes, **self.kwargs
        ) as pool:
            results = [pool.apply_async(func) for func in self.funcs]
            return [result.get() for result in results]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}(processes={self.processes}, "
            f"backend='{self.backend}', context='{self.context}')"
        )


def write_municipality_data(
    data: str | GeoDataFrame | DataFrame,
    out_folder: str,
    municipalities: GeoDataFrame | list[str] | None = None,
    with_neighbors: bool = False,
    muni_number_col: str = "KOMMUNENR",
    file_type: str = "parquet",
    func: Callable | None = None,
    write_empty: bool = False,
    id_assign_func: Callable = clean_overlay,
    strict: bool = True,
    verbose: bool = True,
) -> None:
    """Splits and writes data into municipality-specific files.

    Args:
        data: Path to the data file or a GeoDataFrame.
        out_folder: Path to the output directory where the municipality data
            is written.
        municipalities: Either a sequence of municipality numbers or a GeoDataFrame
            of municipality polygons and municipality numbers in the column 'muni_number_col'.
            Defaults to None.
        with_neighbors: If True, include data from neighboring municipalities
            for each municipality.
        muni_number_col: Column name for municipality codes in 'municipalities'.
        file_type: Format of the output file.
        func: Function to process data before writing.
        write_empty: If True, write empty files for municipalities without data.
        clip: If True, clip the data to municipality boundaries. If False
            the data is spatial joined.
        max_rows_per_chunk: Maximum number of rows in each processed chunk.
        processes_in_clip: Number of processes to use for clipping.
        strict: If True (default) and the data has a municipality column,
            all municipality numbers in 'data' must be present in 'municipalities'.
        id_assign_func: Function to assign ids (e.g. municipality number) to
            the dataframe for missing values.
        verbose: Whether to print during execution.

    Returns:
        None. The function writes files directly.
    """
    write_func = (
        _write_neighbor_municipality_data
        if with_neighbors
        else _write_municipality_data
    )
    return write_func(
        data=data,
        out_folder=out_folder,
        municipalities=municipalities,
        muni_number_col=muni_number_col,
        file_type=file_type,
        func=func,
        write_empty=write_empty,
        strict=strict,
        id_assign_func=id_assign_func,
        verbose=verbose,
    )


def _validate_data(
    data: str | list[str] | DataFrame | GeoDataFrame,
) -> DataFrame | GeoDataFrame:
    if hasattr(data, "__iter__") and len(data) == 1:
        data = data[0]
    if isinstance(data, (str, Path)):
        try:
            return read_geopandas(str(data))
        except ValueError as e:
            try:
                return read_pandas(str(data))
            except ValueError as e2:
                raise e.__class__(e, data) from e2
    return data


def _get_out_path(out_folder: str | Path, muni: str, file_type: str) -> str:
    return str(Path(out_folder) / f"{muni}.{file_type.strip('.')}")


def _write_municipality_data(
    data: str | GeoDataFrame | DataFrame,
    out_folder: str,
    municipalities: GeoDataFrame | list[str] | None = None,
    muni_number_col: str = "KOMMUNENR",
    file_type: str = "parquet",
    func: Callable | None = None,
    write_empty: bool = False,
    processes: int = 1,
    strict: bool = True,
    verbose: bool = True,
    id_assign_func: Callable = clean_overlay,
) -> None:
    if verbose:
        to_print = out_folder
        print(to_print)
    else:
        to_print = None

    gdf = _validate_data(data)

    if func is not None:
        gdf = func(gdf)

    gdf = _fix_missing_muni_numbers(
        gdf,
        municipalities,
        muni_number_col,
        strict=strict,
        id_assign_func=id_assign_func,
    )

    if municipalities is None:
        muni_numbers = gdf[muni_number_col].unique()
    elif not isinstance(municipalities, DataFrame):
        muni_numbers = set(municipalities)
    else:
        muni_numbers = municipalities[muni_number_col].unique()

    muni_numbers = list(sorted(muni_numbers))

    # hardcode this to threading for efficiency in io bound task
    Parallel(processes, backend="threading").map(
        _write_one_muni,
        muni_numbers,
        kwargs=dict(
            gdf=gdf,
            out_folder=out_folder,
            muni_number_col=muni_number_col,
            file_type=file_type,
            write_empty=write_empty,
            to_print=to_print,
        ),
    )


def _write_neighbor_municipality_data(
    data: str | GeoDataFrame | DataFrame,
    out_folder: str,
    municipalities: GeoDataFrame,
    muni_number_col: str = "KOMMUNENR",
    file_type: str = "parquet",
    func: Callable | None = None,
    write_empty: bool = False,
    processes: int = 1,
    strict: bool = True,
    verbose: bool = True,
    id_assign_func: Callable = clean_overlay,
) -> None:
    if verbose:
        to_print = out_folder
        print("out_folder:", to_print)
    else:
        to_print = None

    gdf = _validate_data(data)

    if func is not None:
        gdf = func(gdf)

    gdf = _fix_missing_muni_numbers(
        gdf,
        municipalities,
        muni_number_col,
        strict=strict,
        id_assign_func=id_assign_func,
    )

    if municipalities.index.name != muni_number_col:
        municipalities = municipalities.set_index(muni_number_col)

    neighbor_munis = get_neighbor_indices(
        municipalities, municipalities, max_distance=1
    )

    # hardcode this to threading for efficiency in io bound task
    Parallel(processes, backend="threading").map(
        _write_one_muni_with_neighbors,
        municipalities.index,
        kwargs=dict(
            gdf=gdf,
            neighbor_munis=neighbor_munis,
            out_folder=out_folder,
            muni_number_col=muni_number_col,
            file_type=file_type,
            write_empty=write_empty,
            to_print=to_print,
        ),
    )


def _write_one_muni(
    muni_number: Any,
    gdf: GeoDataFrame | DataFrame,
    out_folder: str | Path,
    muni_number_col: str,
    file_type: str,
    write_empty: bool,
    to_print: str | None = None,
) -> None:
    out = _get_out_path(out_folder, muni_number, file_type)

    if to_print:
        print("writing:", out)

    gdf_muni = gdf.loc[gdf[muni_number_col] == muni_number]

    if not len(gdf_muni):
        if write_empty:
            try:
                geom_col = gdf.geometry.name
            except AttributeError:
                geom_col = "geometry"
            gdf_muni = gdf_muni.drop(columns=geom_col, errors="ignore")
            gdf_muni["geometry"] = None
            write_pandas(gdf_muni, out)
        return

    write_geopandas(gdf_muni, out)


def _write_one_muni_with_neighbors(
    muni_number: Any,
    gdf: GeoDataFrame | DataFrame,
    neighbor_munis: Series,
    out_folder: str | Path,
    muni_number_col: str,
    file_type: str,
    write_empty: bool,
    to_print: str | None = None,
) -> None:
    out = _get_out_path(out_folder, muni_number, file_type)

    if to_print:
        print("writing:", out)

    muni_and_neighbors: Series = neighbor_munis.loc[[muni_number]]
    gdf_neighbor = gdf.loc[gdf[muni_number_col].isin(muni_and_neighbors)]

    if not len(gdf_neighbor):
        if write_empty:
            try:
                geom_col = gdf.geometry.name
            except AttributeError:
                geom_col = "geometry"
            gdf_neighbor = gdf_neighbor.drop(columns=geom_col, errors="ignore")
            gdf_neighbor["geometry"] = None
            write_pandas(gdf_neighbor, out)
        return

    write_geopandas(gdf_neighbor, out)


def _fix_missing_muni_numbers(
    gdf: GeoDataFrame,
    municipalities: GeoDataFrame,
    muni_number_col: str,
    strict: bool,
    id_assign_func: Callable,
) -> GeoDataFrame:
    if muni_number_col in gdf and gdf[muni_number_col].notna().all():
        if municipalities is None:
            return gdf
        if diffs := set(gdf[muni_number_col].values).difference(
            set(municipalities[muni_number_col].values)
        ):
            message = f"Different municipality numbers: {diffs}. Set 'strict=False' to ignore."
            if strict:
                raise ValueError(message)
            else:
                warnings.warn(message, stacklevel=1)
        return gdf

    if municipalities is None:
        if muni_number_col not in gdf:
            raise ValueError(
                f"Cannot find column {muni_number_col}. "
                "Specify another column or a municipality GeoDataFrame to clip "
                "the geometries by."
            )
        assert gdf[muni_number_col].isna().any()
        raise ValueError(
            f"Column {muni_number_col} has missing values. Make sure gdf has "
            "correct municipality number info or specify a municipality "
            "GeoDataFrame to clip the geometries by."
        )

    municipalities = municipalities[
        [muni_number_col, municipalities.geometry.name]
    ].to_crs(gdf.crs)

    if muni_number_col in gdf and gdf[muni_number_col].isna().any():
        notna = gdf[gdf[muni_number_col].notna()]

        isna = gdf[gdf[muni_number_col].isna()].drop(muni_number_col, axis=1)

        notna_anymore = id_assign_func(
            isna,
            municipalities[[muni_number_col, municipalities._geometry_column_name]],
        )

        return pd.concat([notna, notna_anymore], ignore_index=True)

    return id_assign_func(
        gdf,
        municipalities[[muni_number_col, municipalities._geometry_column_name]],
    )


def chunkwise(
    func: Callable,
    iterable: Collection[Iterable[Any]],
    args: tuple | None = None,
    kwargs: dict | None = None,
    processes: int = 1,
    max_rows_per_chunk: int | None = None,
    backend: str = "loky",
) -> Collection[Iterable[Any]]:
    """Run a function in parallel on chunks of a DataFrame.

    This method is used to process large (Geo)DataFrames in manageable pieces,
    optionally in parallel.

    Args:
        func: The function to apply to each chunk. This function must accept a DataFrame as
            its first argument and return a DataFrame.
        iterable: Iterable to be chunked and processed.
        args: Additional positional arguments to pass to 'func'.
        kwargs: Keyword arguments to pass to 'func'.
        processes: The number of parallel jobs to run. Defaults to 1 (no parallel execution).
        max_rows_per_chunk: The maximum number of rows each chunk should contain.
        backend: The backend to use for parallel execution (e.g., 'loky', 'multiprocessing').

    Returns:
        Iterable of iterable.

    """
    args = args or ()
    kwargs = kwargs or {}

    if max_rows_per_chunk is None:
        n_chunks: int = processes
    else:
        n_chunks: int = len(iterable) // max_rows_per_chunk

    if n_chunks <= 1:
        return [func(iterable, *args, **kwargs)]

    chunks = np.array_split(np.arange(len(iterable)), n_chunks)

    if hasattr(iterable, "iloc"):
        iterable_chunked: list[pd.DataFrame | pd.Series] = [
            iterable.iloc[chunk] for chunk in chunks
        ]
    elif is_array_like(iterable):
        iterable_chunked: list[np.ndarray] = [iterable[chunk] for chunk in chunks]
    else:
        to_type: type = iterable.__class__
        iterable_chunked: list[Iterable] = [
            to_type(chunk) for chunk in np.array_split(list(iterable), n_chunks)
        ]

    return Parallel(processes, backend=backend).map(
        func,
        iterable_chunked,
        args=args,
        kwargs=kwargs,
    )


def _turn_args_into_kwargs(func: Callable, args: tuple, index_start: int) -> dict:
    if not isinstance(args, tuple):
        raise TypeError("args should be a tuple (it should not be unpacked with *)")
    argnames = inspect.getfullargspec(func).args[index_start:]
    return {name: value for value, name in zip(args, argnames, strict=False)}


def _try_to_read_geopandas(path: str, **kwargs) -> GeoDataFrame | DataFrame | None:
    """Read with try/except because it's faster than checking exists first."""
    try:
        return read_geopandas(path, **kwargs)
    except FileNotFoundError:
        return None


def _try_to_read_pandas(path: str, **kwargs) -> DataFrame | None:
    """Read with try/except because it's faster than checking exists first."""
    try:
        return read_pandas(path, **kwargs)
    except FileNotFoundError:
        return None
